use crate::tabular::{read_table, write_table, Table, TableValue};
use anyhow::{anyhow, bail, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_COLUMNS: &[&str] = &["frame_i", "Cell_ID"];
const REQUIRED_EDIT_COLUMNS: &[&str] = &["Cell_ID"];
const BASE_LINEAGE_COLUMNS: &[&str] = &[
    "Cell_ID_tree",
    "generation_num_tree",
    "parent_ID_tree",
    "root_ID_tree",
];
const HISTORY_COLUMN: &str = "is_history_known";
const SISTER_BASE_COLUMN: &str = "sister_ID_tree";

type Row = BTreeMap<String, TableValue>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageBuildConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageUpdateConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub frame_i: i64,
    pub edits_table_path: Option<PathBuf>,
    pub edits_json_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineagePropagateConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub frame_i: i64,
    pub cell_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageInfoConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub frame_i: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageOutputPaths {
    pub primary_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LineageFrameInfo {
    pub cells_with_parent: Vec<(i64, i64)>,
    pub orphan_cells: Vec<i64>,
    pub lost_cells: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageFrameEdit {
    pub frame_i: i64,
    pub cell_id: i64,
    pub parent_id: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageCandidateSet {
    pub cell_id: i64,
    pub candidates: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LineageState {
    headers: Vec<String>,
    rows: Vec<Row>,
    sister_columns: Vec<String>,
    row_lookup: BTreeMap<(i64, i64), usize>,
    frames: BTreeMap<i64, Vec<usize>>,
    first_frame_by_cell: BTreeMap<i64, i64>,
}

impl LineageState {
    pub fn to_table(&self) -> Table {
        rows_to_table(&self.headers, &self.rows)
    }

    pub fn cell_ids_in_frame(&self, frame_i: i64) -> Vec<i64> {
        self.frames
            .get(&frame_i)
            .into_iter()
            .flatten()
            .filter_map(|idx| self.rows[*idx].get("Cell_ID").and_then(TableValue::as_i64))
            .collect()
    }

    pub fn sister_columns(&self) -> &[String] {
        &self.sister_columns
    }
}

pub fn load_lineage_state(table: &Table) -> Result<LineageState> {
    build_lineage_state(table)
}

pub fn build_lineage_state(table: &Table) -> Result<LineageState> {
    ensure_required_columns(table)?;
    let mut headers = table.headers.clone();
    let mut rows = table_to_rows(table);
    let sister_columns = detect_sister_columns(&headers);
    let mut normalized_sister_columns = sister_columns.clone();
    if normalized_sister_columns.is_empty() {
        normalized_sister_columns.push(SISTER_BASE_COLUMN.to_string());
    }

    if has_fully_populated_lineage_columns(table, &normalized_sister_columns) {
        let max_sisters = rows
            .iter()
            .map(|row| sister_values(row, &normalized_sister_columns).len())
            .max()
            .unwrap_or(0)
            .max(1);
        normalized_sister_columns = normalized_sister_headers(max_sisters);
        ensure_headers(
            &mut headers,
            normalized_lineage_headers(&normalized_sister_columns),
        );
        for row in &mut rows {
            normalize_loaded_lineage_row(row, &normalized_sister_columns)?;
        }
    } else {
        normalized_sister_columns = normalized_sister_headers(1);
        ensure_headers(
            &mut headers,
            normalized_lineage_headers(&normalized_sister_columns),
        );
        initialize_root_lineage(&mut rows, &normalized_sister_columns)?;
    }

    Ok(LineageState::new(headers, rows, normalized_sister_columns))
}

pub fn update_lineage_frame(
    mut state: LineageState,
    frame_i: i64,
    edited_rows: &Table,
) -> Result<LineageState> {
    ensure_table_columns(edited_rows, REQUIRED_EDIT_COLUMNS)?;
    let edit_rows = table_to_rows(edited_rows);
    let mut edited_cell_ids = BTreeSet::new();
    let sister_columns = state.sister_columns.clone();
    for edit in edit_rows {
        let cell_id = get_required_i64(&edit, "Cell_ID")?;
        let row_idx = *state
            .row_lookup
            .get(&(frame_i, cell_id))
            .ok_or_else(|| anyhow!("Missing row for frame {frame_i}, Cell_ID {cell_id}"))?;
        apply_lineage_edit(&mut state.rows[row_idx], &edit, &sister_columns)?;
        edited_cell_ids.insert(cell_id);
    }

    let edited_cell_ids = edited_cell_ids.into_iter().collect::<Vec<_>>();
    let sibling_roots = collect_affected_sibling_roots(&state, &edited_cell_ids);
    repair_local_frame(&mut state, frame_i, &edited_cell_ids)?;
    recompute_sister_groups(&mut state, &sibling_roots)?;
    Ok(LineageState::new(
        state.headers,
        state.rows,
        state.sister_columns,
    ))
}

pub fn propagate_lineage(
    mut state: LineageState,
    frame_i: i64,
    relevant_cell_ids: &[i64],
) -> Result<LineageState> {
    let sister_columns = state.sister_columns.clone();
    let mut parent_aliases = BTreeMap::<i64, BTreeSet<i64>>::new();
    for cell_id in relevant_cell_ids {
        let Some(source_idx) = state.row_lookup.get(&(frame_i, *cell_id)).copied() else {
            continue;
        };
        let source_values = lineage_values(&state.rows[source_idx], state.sister_columns());
        let alias_values = lineage_aliases_for_cell(&state, *cell_id, frame_i);
        parent_aliases.insert(*cell_id, alias_values);
        for (&(row_frame, row_cell_id), &row_idx) in &state.row_lookup {
            if row_frame <= frame_i || row_cell_id != *cell_id {
                continue;
            }
            apply_lineage_values(&mut state.rows[row_idx], &source_values, &sister_columns);
        }
    }

    let mut repaired_children = BTreeSet::new();
    for (&parent_cell_id, aliases) in &parent_aliases {
        for frame in state
            .frames
            .keys()
            .copied()
            .filter(|candidate| *candidate > frame_i)
        {
            let Some(parent_idx) = state.row_lookup.get(&(frame, parent_cell_id)).copied() else {
                continue;
            };
            let parent_tree_id =
                get_or_default_i64(&state.rows[parent_idx], "Cell_ID_tree", parent_cell_id);
            let child_ids = frame_rows(&state, frame)
                .filter_map(|(_, row)| {
                    let child_parent = get_optional_i64(row, "parent_ID_tree")?;
                    if aliases.contains(&child_parent) || child_parent == parent_tree_id {
                        row.get("Cell_ID").and_then(TableValue::as_i64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            for child_id in child_ids {
                let child_idx = *state
                    .row_lookup
                    .get(&(frame, child_id))
                    .ok_or_else(|| anyhow!("Missing propagated child row"))?;
                let parent_row = state.rows[parent_idx].clone();
                repair_child_from_parent(
                    &mut state.rows[child_idx],
                    &parent_row,
                    parent_tree_id,
                    &sister_columns,
                );
                repaired_children.insert(child_id);
            }
        }
    }

    let mut sibling_roots = collect_affected_sibling_roots(&state, relevant_cell_ids);
    sibling_roots.extend(collect_affected_sibling_roots(
        &state,
        &repaired_children.into_iter().collect::<Vec<_>>(),
    ));
    recompute_sister_groups(&mut state, &sibling_roots)?;

    Ok(LineageState::new(
        state.headers,
        state.rows,
        state.sister_columns,
    ))
}

pub fn export_lineage_frame(state: &LineageState, frame_i: i64) -> Result<Table> {
    let headers = state.headers.clone();
    let rows = frame_rows(state, frame_i)
        .map(|(_, row)| row.clone())
        .collect::<Vec<_>>();
    Ok(rows_to_table(&headers, &rows))
}

pub fn export_lineage_info(state: &LineageState, frame_i: i64) -> Result<LineageFrameInfo> {
    if frame_i <= 0 {
        return Ok(LineageFrameInfo {
            cells_with_parent: Vec::new(),
            orphan_cells: Vec::new(),
            lost_cells: Vec::new(),
        });
    }
    let prev_ids = state
        .cell_ids_in_frame(frame_i - 1)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let curr_ids = state
        .cell_ids_in_frame(frame_i)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let new_ids = curr_ids.difference(&prev_ids).copied().collect::<Vec<_>>();
    let mut cells_with_parent = Vec::new();
    let mut orphan_cells = Vec::new();
    let mut parent_ids = BTreeSet::new();
    for cell_id in new_ids {
        let Some(row_idx) = state.row_lookup.get(&(frame_i, cell_id)).copied() else {
            continue;
        };
        let parent_id = get_optional_i64(&state.rows[row_idx], "parent_ID_tree").unwrap_or(-1);
        if parent_id > 0 {
            cells_with_parent.push((cell_id, parent_id));
            parent_ids.insert(parent_id);
        } else {
            orphan_cells.push(cell_id);
        }
    }
    let mut lost_cells = prev_ids.difference(&curr_ids).copied().collect::<Vec<_>>();
    lost_cells.retain(|cell_id| !parent_ids.contains(cell_id));
    cells_with_parent.sort_unstable();
    orphan_cells.sort_unstable();
    lost_cells.sort_unstable();
    Ok(LineageFrameInfo {
        cells_with_parent,
        orphan_cells,
        lost_cells,
    })
}

pub fn lineage_mother_candidates(
    state: &LineageState,
    frame_i: i64,
    cell_id: i64,
) -> Result<LineageCandidateSet> {
    let mut candidates = state
        .cell_ids_in_frame(frame_i)
        .into_iter()
        .filter(|candidate| *candidate != cell_id)
        .collect::<Vec<_>>();
    if candidates.is_empty() && frame_i > 0 {
        candidates = state
            .cell_ids_in_frame(frame_i - 1)
            .into_iter()
            .filter(|candidate| *candidate != cell_id)
            .collect::<Vec<_>>();
    }
    candidates.sort_unstable();
    candidates.dedup();
    Ok(LineageCandidateSet { cell_id, candidates })
}

pub fn set_lineage_parent(
    state: LineageState,
    frame_i: i64,
    cell_id: i64,
    parent_id: i64,
) -> Result<LineageState> {
    let edits = Table {
        headers: vec![
            "Cell_ID".into(),
            "parent_ID_tree".into(),
            HISTORY_COLUMN.into(),
        ],
        rows: vec![vec![
            TableValue::Number(cell_id as f64),
            TableValue::Number(parent_id as f64),
            TableValue::Bool(true),
        ]],
    };
    update_lineage_frame(state, frame_i, &edits)
}

pub fn set_lineage_unknown(
    state: LineageState,
    frame_i: i64,
    cell_id: i64,
) -> Result<LineageState> {
    let edits = Table {
        headers: vec![
            "Cell_ID".into(),
            "parent_ID_tree".into(),
            HISTORY_COLUMN.into(),
        ],
        rows: vec![vec![
            TableValue::Number(cell_id as f64),
            TableValue::Number(-1.0),
            TableValue::Bool(false),
        ]],
    };
    update_lineage_frame(state, frame_i, &edits)
}

pub fn propagate_lineage_from_frame(
    state: LineageState,
    frame_i: i64,
    relevant_cell_ids: &[i64],
) -> Result<LineageState> {
    propagate_lineage(state, frame_i, relevant_cell_ids)
}

pub fn build_lineage_state_file(config: LineageBuildConfig) -> Result<LineageOutputPaths> {
    let table = read_table(&config.input_path)?;
    let state = build_lineage_state(&table)?;
    write_table(&config.output_path, &state.to_table())?;
    Ok(LineageOutputPaths {
        primary_path: config.output_path,
    })
}

pub fn update_lineage_frame_file(config: LineageUpdateConfig) -> Result<LineageOutputPaths> {
    let source = read_table(&config.input_path)?;
    let state = build_lineage_state(&source)?;
    let edits = read_lineage_edits(
        config.edits_table_path.as_deref(),
        config.edits_json_path.as_deref(),
    )?;
    let updated = update_lineage_frame(state, config.frame_i, &edits)?;
    write_table(&config.output_path, &updated.to_table())?;
    Ok(LineageOutputPaths {
        primary_path: config.output_path,
    })
}

pub fn propagate_lineage_file(config: LineagePropagateConfig) -> Result<LineageOutputPaths> {
    let source = read_table(&config.input_path)?;
    let state = build_lineage_state(&source)?;
    let cell_ids = config
        .cell_ids
        .clone()
        .unwrap_or_else(|| state.cell_ids_in_frame(config.frame_i));
    let propagated = propagate_lineage(state, config.frame_i, &cell_ids)?;
    write_table(&config.output_path, &propagated.to_table())?;
    Ok(LineageOutputPaths {
        primary_path: config.output_path,
    })
}

pub fn export_lineage_info_file(config: LineageInfoConfig) -> Result<LineageOutputPaths> {
    let source = read_table(&config.input_path)?;
    let state = build_lineage_state(&source)?;
    let info = export_lineage_info(&state, config.frame_i)?;
    let json = serde_json::json!({
        "cells_with_parent": info.cells_with_parent
            .iter()
            .map(|(cell_id, parent_id)| serde_json::json!({
                "cell_id": cell_id,
                "parent_id": parent_id,
            }))
            .collect::<Vec<_>>(),
        "orphan_cells": info.orphan_cells,
        "lost_cells": info.lost_cells,
    });
    let parent = config.output_path.parent().ok_or_else(|| {
        anyhow!(
            "Output path has no parent: {}",
            config.output_path.display()
        )
    })?;
    fs::create_dir_all(parent)?;
    fs::write(&config.output_path, serde_json::to_vec_pretty(&json)?)?;
    Ok(LineageOutputPaths {
        primary_path: config.output_path,
    })
}

fn read_lineage_edits(
    edits_table_path: Option<&Path>,
    edits_json_path: Option<&Path>,
) -> Result<Table> {
    match (edits_table_path, edits_json_path) {
        (Some(table_path), None) => read_table(table_path),
        (None, Some(json_path)) => read_lineage_edits_json(json_path),
        (Some(_), Some(_)) => bail!("Provide only one of edits table or edits JSON"),
        (None, None) => bail!("Lineage frame update requires an edits table or edits JSON"),
    }
}

fn read_lineage_edits_json(path: &Path) -> Result<Table> {
    let text = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let rows = value
        .as_array()
        .ok_or_else(|| anyhow!("Lineage edits JSON must be an array of objects"))?;
    let mut headers = Vec::new();
    let mut mapped_rows = Vec::new();
    for item in rows {
        let object = item
            .as_object()
            .ok_or_else(|| anyhow!("Each lineage edit must be a JSON object"))?;
        let mut row = Row::new();
        for (key, value) in object {
            extend_header_order(&mut headers, [key.clone()]);
            row.insert(key.clone(), json_value_to_table_value(value));
        }
        mapped_rows.push(row);
    }
    Ok(rows_to_table(&headers, &mapped_rows))
}

fn has_fully_populated_lineage_columns(table: &Table, sister_columns: &[String]) -> bool {
    let required = BASE_LINEAGE_COLUMNS
        .iter()
        .map(|value| (*value).to_string())
        .chain(std::iter::once(HISTORY_COLUMN.to_string()))
        .chain(sister_columns.iter().cloned())
        .collect::<Vec<_>>();
    required
        .iter()
        .all(|column| table.headers.iter().any(|header| header == column))
        && table.rows.iter().all(|row| {
            required.iter().all(|column| {
                table
                    .maybe_header_index(column)
                    .and_then(|idx| row.get(idx))
                    .map(|value| !matches!(value, TableValue::Empty))
                    .unwrap_or(false)
            })
        })
}

fn initialize_root_lineage(rows: &mut [Row], sister_columns: &[String]) -> Result<()> {
    for row in rows {
        let cell_id = get_required_i64(row, "Cell_ID")?;
        row.insert("Cell_ID_tree".into(), TableValue::Number(cell_id as f64));
        row.insert("generation_num_tree".into(), TableValue::Number(1.0));
        row.insert("parent_ID_tree".into(), TableValue::Number(-1.0));
        row.insert("root_ID_tree".into(), TableValue::Number(cell_id as f64));
        row.insert(HISTORY_COLUMN.into(), TableValue::Bool(false));
        set_sister_values(row, &[]);
        ensure_sister_capacity(row, sister_columns);
    }
    Ok(())
}

fn normalize_loaded_lineage_row(row: &mut Row, sister_columns: &[String]) -> Result<()> {
    let cell_id = get_required_i64(row, "Cell_ID")?;
    let cell_id_tree = get_or_default_i64(row, "Cell_ID_tree", cell_id);
    let parent_id = get_or_default_i64(row, "parent_ID_tree", -1);
    row.insert(
        "Cell_ID_tree".into(),
        TableValue::Number(cell_id_tree as f64),
    );
    row.insert(
        "parent_ID_tree".into(),
        TableValue::Number(parent_id as f64),
    );
    if parent_id <= 0 {
        row.insert("generation_num_tree".into(), TableValue::Number(1.0));
        row.insert("root_ID_tree".into(), TableValue::Number(cell_id as f64));
        row.insert(HISTORY_COLUMN.into(), TableValue::Bool(false));
        set_sister_values(row, &[]);
    } else {
        let generation = get_or_default_i64(row, "generation_num_tree", 2);
        let root_id = get_or_default_i64(row, "root_ID_tree", parent_id);
        row.insert(
            "generation_num_tree".into(),
            TableValue::Number(generation as f64),
        );
        row.insert("root_ID_tree".into(), TableValue::Number(root_id as f64));
        row.insert(HISTORY_COLUMN.into(), TableValue::Bool(true));
        let sisters = sister_values(row, sister_columns);
        set_sister_values(row, &sisters);
    }
    ensure_sister_capacity(row, sister_columns);
    Ok(())
}

fn repair_local_frame(state: &mut LineageState, frame_i: i64, cell_ids: &[i64]) -> Result<()> {
    let sister_columns = state.sister_columns.clone();
    for cell_id in cell_ids {
        let Some(row_idx) = state.row_lookup.get(&(frame_i, *cell_id)).copied() else {
            continue;
        };
        let parent_id = get_or_default_i64(&state.rows[row_idx], "parent_ID_tree", -1);
        if parent_id <= 0 {
            state.rows[row_idx].insert("generation_num_tree".into(), TableValue::Number(1.0));
            state.rows[row_idx].insert("root_ID_tree".into(), TableValue::Number(*cell_id as f64));
            state.rows[row_idx].insert(HISTORY_COLUMN.into(), TableValue::Bool(false));
            set_sister_values(&mut state.rows[row_idx], &[]);
            continue;
        }
        let Some(parent_idx) = state.row_lookup.get(&(frame_i, parent_id)).copied() else {
            state.rows[row_idx].insert("parent_ID_tree".into(), TableValue::Number(-1.0));
            state.rows[row_idx].insert("generation_num_tree".into(), TableValue::Number(1.0));
            state.rows[row_idx].insert("root_ID_tree".into(), TableValue::Number(*cell_id as f64));
            state.rows[row_idx].insert(HISTORY_COLUMN.into(), TableValue::Bool(false));
            set_sister_values(&mut state.rows[row_idx], &[]);
            continue;
        };
        let parent_row = state.rows[parent_idx].clone();
        let parent_tree_id = get_or_default_i64(&parent_row, "Cell_ID_tree", parent_id);
        repair_child_from_parent(
            &mut state.rows[row_idx],
            &parent_row,
            parent_tree_id,
            &sister_columns,
        );
    }
    Ok(())
}

fn repair_child_from_parent(
    row: &mut Row,
    parent_row: &Row,
    parent_tree_id: i64,
    sister_columns: &[String],
) {
    let parent_generation = get_or_default_i64(parent_row, "generation_num_tree", 1);
    let parent_root = get_or_default_i64(
        parent_row,
        "root_ID_tree",
        get_or_default_i64(parent_row, "Cell_ID_tree", -1),
    );
    row.insert(
        "parent_ID_tree".into(),
        TableValue::Number(parent_tree_id as f64),
    );
    row.insert(
        "generation_num_tree".into(),
        TableValue::Number((parent_generation + 1) as f64),
    );
    row.insert(
        "root_ID_tree".into(),
        TableValue::Number(parent_root as f64),
    );
    row.insert(HISTORY_COLUMN.into(), TableValue::Bool(true));
    ensure_sister_capacity(row, sister_columns);
}

fn recompute_sister_groups(state: &mut LineageState, roots: &BTreeSet<(i64, i64)>) -> Result<()> {
    for &(first_frame, anchor_cell_id) in roots {
        let Some(anchor_idx) = state
            .row_lookup
            .get(&(first_frame, anchor_cell_id))
            .copied()
        else {
            continue;
        };
        let parent_id = get_or_default_i64(&state.rows[anchor_idx], "parent_ID_tree", -1);
        if parent_id <= 0 {
            set_sister_values(&mut state.rows[anchor_idx], &[]);
            continue;
        }
        let sibling_ids = frame_rows(state, first_frame)
            .filter_map(|(_, row)| {
                if get_or_default_i64(row, "parent_ID_tree", -1) == parent_id {
                    row.get("Cell_ID").and_then(TableValue::as_i64)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let needed_columns = normalized_sister_headers(sibling_ids.len().max(1));
        if needed_columns.len() > state.sister_columns.len() {
            for column in needed_columns.iter().skip(state.sister_columns.len()) {
                state.headers.push(column.clone());
            }
            state.sister_columns = needed_columns;
            let sister_columns = state.sister_columns.clone();
            for row in &mut state.rows {
                ensure_sister_capacity(row, &sister_columns);
            }
        }
        for sibling_id in &sibling_ids {
            let sisters = sibling_ids
                .iter()
                .copied()
                .filter(|candidate| candidate != sibling_id)
                .collect::<Vec<_>>();
            if let Some(idx) = state.row_lookup.get(&(first_frame, *sibling_id)).copied() {
                set_sister_values(&mut state.rows[idx], &sisters);
            }
        }
    }
    Ok(())
}

fn collect_affected_sibling_roots(state: &LineageState, cell_ids: &[i64]) -> BTreeSet<(i64, i64)> {
    cell_ids
        .iter()
        .filter_map(|cell_id| {
            state
                .first_frame_by_cell
                .get(cell_id)
                .copied()
                .map(|frame| (frame, *cell_id))
        })
        .collect()
}

fn normalized_lineage_headers(sister_columns: &[String]) -> Vec<String> {
    let mut headers = BASE_LINEAGE_COLUMNS
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();
    headers.extend(sister_columns.iter().cloned());
    if !headers.iter().any(|header| header == HISTORY_COLUMN) {
        headers.push(HISTORY_COLUMN.to_string());
    }
    headers
}

fn detect_sister_columns(headers: &[String]) -> Vec<String> {
    let mut columns = headers
        .iter()
        .filter(|header| {
            *header == SISTER_BASE_COLUMN
                || header
                    .strip_prefix(&format!("{SISTER_BASE_COLUMN}_"))
                    .and_then(|suffix| suffix.parse::<usize>().ok())
                    .is_some()
        })
        .cloned()
        .collect::<Vec<_>>();
    columns.sort_by_key(|column| sister_column_order(column));
    columns
}

fn normalized_sister_headers(count: usize) -> Vec<String> {
    (0..count)
        .map(|idx| {
            if idx == 0 {
                SISTER_BASE_COLUMN.to_string()
            } else {
                format!("{SISTER_BASE_COLUMN}_{idx}")
            }
        })
        .collect()
}

fn sister_column_order(column: &str) -> usize {
    if column == SISTER_BASE_COLUMN {
        0
    } else {
        column
            .strip_prefix(&format!("{SISTER_BASE_COLUMN}_"))
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    }
}

fn apply_lineage_edit(row: &mut Row, edit: &Row, sister_columns: &[String]) -> Result<()> {
    let mut sister_values_override = None;
    for column in BASE_LINEAGE_COLUMNS {
        if let Some(value) = edit.get(*column).cloned() {
            row.insert((*column).to_string(), value);
        }
    }
    if let Some(value) = edit.get(HISTORY_COLUMN).cloned() {
        row.insert(HISTORY_COLUMN.to_string(), normalize_bool_value(value)?);
    }
    let incoming_sisters = detect_sister_columns(&edit.keys().cloned().collect::<Vec<_>>());
    if !incoming_sisters.is_empty() {
        sister_values_override = Some(sister_values(edit, &incoming_sisters));
    }
    if let Some(values) = sister_values_override {
        set_sister_values(row, &values);
        ensure_sister_capacity(row, sister_columns);
    }
    Ok(())
}

fn lineage_values(row: &Row, sister_columns: &[String]) -> LineageValues {
    LineageValues {
        cell_id_tree: get_or_default_i64(
            row,
            "Cell_ID_tree",
            get_or_default_i64(row, "Cell_ID", -1),
        ),
        generation_num_tree: get_or_default_i64(row, "generation_num_tree", 1),
        parent_id_tree: get_or_default_i64(row, "parent_ID_tree", -1),
        root_id_tree: get_or_default_i64(
            row,
            "root_ID_tree",
            get_or_default_i64(row, "Cell_ID", -1),
        ),
        sister_ids: sister_values(row, sister_columns),
        is_history_known: get_or_default_bool(row, HISTORY_COLUMN, false),
    }
}

fn apply_lineage_values(row: &mut Row, values: &LineageValues, sister_columns: &[String]) {
    row.insert(
        "Cell_ID_tree".into(),
        TableValue::Number(values.cell_id_tree as f64),
    );
    row.insert(
        "generation_num_tree".into(),
        TableValue::Number(values.generation_num_tree as f64),
    );
    row.insert(
        "parent_ID_tree".into(),
        TableValue::Number(values.parent_id_tree as f64),
    );
    row.insert(
        "root_ID_tree".into(),
        TableValue::Number(values.root_id_tree as f64),
    );
    row.insert(
        HISTORY_COLUMN.into(),
        TableValue::Bool(values.is_history_known),
    );
    set_sister_values(row, &values.sister_ids);
    ensure_sister_capacity(row, sister_columns);
}

fn lineage_aliases_for_cell(state: &LineageState, cell_id: i64, frame_i: i64) -> BTreeSet<i64> {
    let mut aliases = BTreeSet::new();
    for (&(row_frame, row_cell_id), &row_idx) in &state.row_lookup {
        if row_cell_id != cell_id || row_frame < frame_i {
            continue;
        }
        aliases.insert(get_or_default_i64(
            &state.rows[row_idx],
            "Cell_ID_tree",
            cell_id,
        ));
    }
    aliases.insert(cell_id);
    aliases
}

fn frame_rows<'a>(
    state: &'a LineageState,
    frame_i: i64,
) -> impl Iterator<Item = (usize, &'a Row)> + 'a {
    state
        .frames
        .get(&frame_i)
        .into_iter()
        .flatten()
        .map(|idx| (*idx, &state.rows[*idx]))
}

fn sister_values(row: &Row, sister_columns: &[String]) -> Vec<i64> {
    let mut values = sister_columns
        .iter()
        .filter_map(|column| get_optional_i64(row, column))
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn set_sister_values(row: &mut Row, sister_ids: &[i64]) {
    let mut keys = row
        .keys()
        .filter(|key| {
            *key == SISTER_BASE_COLUMN
                || key
                    .strip_prefix(&format!("{SISTER_BASE_COLUMN}_"))
                    .and_then(|suffix| suffix.parse::<usize>().ok())
                    .is_some()
        })
        .cloned()
        .collect::<Vec<_>>();
    keys.sort_by_key(|column| sister_column_order(column));
    for key in keys {
        row.remove(&key);
    }
    let columns = normalized_sister_headers(sister_ids.len().max(1));
    for (idx, column) in columns.into_iter().enumerate() {
        let value = sister_ids.get(idx).copied().unwrap_or(-1);
        row.insert(column, TableValue::Number(value as f64));
    }
}

fn ensure_sister_capacity(row: &mut Row, sister_columns: &[String]) {
    let sisters = sister_values(
        row,
        &detect_sister_columns(&row.keys().cloned().collect::<Vec<_>>()),
    );
    for column in detect_sister_columns(&row.keys().cloned().collect::<Vec<_>>()) {
        row.remove(&column);
    }
    for (idx, column) in sister_columns.iter().enumerate() {
        let value = sisters.get(idx).copied().unwrap_or(-1);
        row.insert(column.clone(), TableValue::Number(value as f64));
    }
}

fn normalize_bool_value(value: TableValue) -> Result<TableValue> {
    let parsed = match value {
        TableValue::Bool(value) => value,
        TableValue::Number(value) => value != 0.0,
        TableValue::Text(value) => match value.to_ascii_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            other => bail!("Invalid boolean lineage value {other:?}"),
        },
        TableValue::Empty => false,
    };
    Ok(TableValue::Bool(parsed))
}

fn ensure_required_columns(table: &Table) -> Result<()> {
    ensure_table_columns(table, REQUIRED_COLUMNS)
}

fn ensure_table_columns(table: &Table, columns: &[&str]) -> Result<()> {
    for column in columns {
        table.header_index(column)?;
    }
    Ok(())
}

fn ensure_headers(headers: &mut Vec<String>, extra: Vec<String>) {
    for column in extra {
        extend_header_order(headers, [column]);
    }
}

fn extend_header_order(headers: &mut Vec<String>, values: impl IntoIterator<Item = String>) {
    for value in values {
        if !headers.iter().any(|header| header == &value) {
            headers.push(value);
        }
    }
}

fn get_required_i64(row: &Row, column: &str) -> Result<i64> {
    row.get(column)
        .and_then(TableValue::as_i64)
        .ok_or_else(|| anyhow!("Missing or invalid integer column {column:?}"))
}

fn get_optional_i64(row: &Row, column: &str) -> Option<i64> {
    row.get(column).and_then(TableValue::as_i64)
}

fn get_or_default_i64(row: &Row, column: &str, default: i64) -> i64 {
    get_optional_i64(row, column).unwrap_or(default)
}

fn get_or_default_bool(row: &Row, column: &str, default: bool) -> bool {
    match row.get(column) {
        Some(TableValue::Bool(value)) => *value,
        Some(TableValue::Number(value)) => *value != 0.0,
        Some(TableValue::Text(value)) => match value.to_ascii_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => default,
        },
        _ => default,
    }
}

fn table_to_rows(table: &Table) -> Vec<Row> {
    table
        .rows
        .iter()
        .map(|row| {
            table
                .headers
                .iter()
                .cloned()
                .zip(row.iter().cloned())
                .collect::<Row>()
        })
        .collect()
}

fn rows_to_table(headers: &[String], rows: &[Row]) -> Table {
    let mut table = Table::new(headers.to_vec());
    for row in rows {
        table.rows.push(
            headers
                .iter()
                .map(|header| row.get(header).cloned().unwrap_or(TableValue::Empty))
                .collect(),
        );
    }
    table
}

fn json_value_to_table_value(value: &serde_json::Value) -> TableValue {
    match value {
        serde_json::Value::Null => TableValue::Empty,
        serde_json::Value::Bool(value) => TableValue::Bool(*value),
        serde_json::Value::Number(value) => TableValue::Number(value.as_f64().unwrap_or(f64::NAN)),
        serde_json::Value::String(value) => TableValue::Text(value.clone()),
        other => TableValue::Text(other.to_string()),
    }
}

impl LineageState {
    fn new(headers: Vec<String>, rows: Vec<Row>, sister_columns: Vec<String>) -> Self {
        let mut row_lookup = BTreeMap::new();
        let mut frames = BTreeMap::<i64, Vec<usize>>::new();
        let mut first_frame_by_cell = BTreeMap::<i64, i64>::new();
        for (idx, row) in rows.iter().enumerate() {
            let frame_i = row
                .get("frame_i")
                .and_then(TableValue::as_i64)
                .expect("validated frame_i");
            let cell_id = row
                .get("Cell_ID")
                .and_then(TableValue::as_i64)
                .expect("validated Cell_ID");
            row_lookup.insert((frame_i, cell_id), idx);
            frames.entry(frame_i).or_default().push(idx);
            first_frame_by_cell
                .entry(cell_id)
                .and_modify(|current| *current = (*current).min(frame_i))
                .or_insert(frame_i);
        }
        Self {
            headers,
            rows,
            sister_columns,
            row_lookup,
            frames,
            first_frame_by_cell,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LineageValues {
    cell_id_tree: i64,
    generation_num_tree: i64,
    parent_id_tree: i64,
    root_id_tree: i64,
    sister_ids: Vec<i64>,
    is_history_known: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn n(value: f64) -> TableValue {
        TableValue::Number(value)
    }

    fn b(value: bool) -> TableValue {
        TableValue::Bool(value)
    }

    #[test]
    fn builds_root_lineage_when_tree_columns_missing() -> Result<()> {
        let table = Table {
            headers: vec![
                "frame_i".into(),
                "Cell_ID".into(),
                "cell_cycle_stage".into(),
                "generation_num".into(),
                "relative_ID".into(),
                "relationship".into(),
                "is_history_known".into(),
            ],
            rows: vec![
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(4.0),
                    TableValue::Text("G1".into()),
                    TableValue::Number(1.0),
                    TableValue::Number(-1.0),
                    TableValue::Text("mother".into()),
                    TableValue::Bool(true),
                ],
                vec![
                    TableValue::Number(1.0),
                    TableValue::Number(4.0),
                    TableValue::Text("G1".into()),
                    TableValue::Number(1.0),
                    TableValue::Number(-1.0),
                    TableValue::Text("mother".into()),
                    TableValue::Bool(true),
                ],
            ],
        };
        let state = build_lineage_state(&table)?;
        let exported = state.to_table();
        let parent_idx = exported.header_index("parent_ID_tree")?;
        let root_idx = exported.header_index("root_ID_tree")?;
        let hist_idx = exported.header_index("is_history_known")?;
        assert_eq!(exported.rows[0][parent_idx].as_i64(), Some(-1));
        assert_eq!(exported.rows[0][root_idx].as_i64(), Some(4));
        assert_eq!(exported.rows[0][hist_idx], TableValue::Bool(false));
        Ok(())
    }

    #[test]
    fn update_lineage_frame_repairs_parented_rows_and_sisters() -> Result<()> {
        let source = Table {
            headers: vec![
                "frame_i".into(),
                "Cell_ID".into(),
                "Cell_ID_tree".into(),
                "generation_num_tree".into(),
                "parent_ID_tree".into(),
                "root_ID_tree".into(),
                "sister_ID_tree".into(),
                "is_history_known".into(),
            ],
            rows: vec![
                vec![
                    n(0.0),
                    n(1.0),
                    n(1.0),
                    n(1.0),
                    n(-1.0),
                    n(1.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(0.0),
                    n(2.0),
                    n(2.0),
                    n(1.0),
                    n(-1.0),
                    n(2.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(0.0),
                    n(3.0),
                    n(3.0),
                    n(1.0),
                    n(-1.0),
                    n(3.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(1.0),
                    n(1.0),
                    n(1.0),
                    n(1.0),
                    n(-1.0),
                    n(1.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(1.0),
                    n(2.0),
                    n(2.0),
                    n(1.0),
                    n(-1.0),
                    n(2.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(1.0),
                    n(3.0),
                    n(3.0),
                    n(1.0),
                    n(-1.0),
                    n(3.0),
                    n(-1.0),
                    b(false),
                ],
            ],
        };
        let state = build_lineage_state(&source)?;
        let edits = Table {
            headers: vec!["Cell_ID".into(), "parent_ID_tree".into()],
            rows: vec![vec![n(2.0), n(1.0)], vec![n(3.0), n(1.0)]],
        };
        let updated = update_lineage_frame(state, 0, &edits)?;
        let exported = updated.to_table();
        let frame0_cell2 = exported
            .rows
            .iter()
            .find(|row| row[0].as_i64() == Some(0) && row[1].as_i64() == Some(2))
            .cloned()
            .expect("cell 2");
        let frame0_cell3 = exported
            .rows
            .iter()
            .find(|row| row[0].as_i64() == Some(0) && row[1].as_i64() == Some(3))
            .cloned()
            .expect("cell 3");
        assert_eq!(frame0_cell2[4].as_i64(), Some(1));
        assert_eq!(frame0_cell2[3].as_i64(), Some(2));
        assert_eq!(frame0_cell2[5].as_i64(), Some(1));
        assert_eq!(frame0_cell2[6].as_i64(), Some(3));
        assert_eq!(frame0_cell3[6].as_i64(), Some(2));
        Ok(())
    }

    #[test]
    fn propagate_lineage_updates_future_rows_and_children() -> Result<()> {
        let table = Table {
            headers: vec![
                "frame_i".into(),
                "Cell_ID".into(),
                "Cell_ID_tree".into(),
                "generation_num_tree".into(),
                "parent_ID_tree".into(),
                "root_ID_tree".into(),
                "sister_ID_tree".into(),
                "is_history_known".into(),
                "signal".into(),
            ],
            rows: vec![
                vec![
                    n(0.0),
                    n(1.0),
                    n(10.0),
                    n(1.0),
                    n(-1.0),
                    n(1.0),
                    n(-1.0),
                    b(false),
                    n(2.0),
                ],
                vec![
                    n(0.0),
                    n(2.0),
                    n(20.0),
                    n(2.0),
                    n(10.0),
                    n(1.0),
                    n(-1.0),
                    b(true),
                    n(3.0),
                ],
                vec![
                    n(1.0),
                    n(1.0),
                    n(99.0),
                    n(1.0),
                    n(-1.0),
                    n(1.0),
                    n(-1.0),
                    b(false),
                    n(4.0),
                ],
                vec![
                    n(1.0),
                    n(2.0),
                    n(20.0),
                    n(5.0),
                    n(99.0),
                    n(8.0),
                    n(-1.0),
                    b(true),
                    n(5.0),
                ],
            ],
        };
        let state = build_lineage_state(&table)?;
        let edits = Table {
            headers: vec![
                "Cell_ID".into(),
                "Cell_ID_tree".into(),
                "root_ID_tree".into(),
            ],
            rows: vec![vec![n(1.0), n(10.0), n(1.0)]],
        };
        let updated = update_lineage_frame(state, 0, &edits)?;
        let propagated = propagate_lineage(updated, 0, &[1])?;
        let exported = propagated.to_table();
        let parent_row = exported
            .rows
            .iter()
            .find(|row| row[0].as_i64() == Some(1) && row[1].as_i64() == Some(1))
            .expect("parent frame 1");
        let child_row = exported
            .rows
            .iter()
            .find(|row| row[0].as_i64() == Some(1) && row[1].as_i64() == Some(2))
            .expect("child frame 1");
        assert_eq!(parent_row[2].as_i64(), Some(10));
        assert_eq!(child_row[4].as_i64(), Some(10));
        assert_eq!(child_row[3].as_i64(), Some(2));
        assert_eq!(child_row[5].as_i64(), Some(1));
        assert_eq!(child_row[8].as_i64(), Some(5));
        Ok(())
    }

    #[test]
    fn exports_lineage_info_for_new_and_lost_cells() -> Result<()> {
        let table = Table {
            headers: vec![
                "frame_i".into(),
                "Cell_ID".into(),
                "Cell_ID_tree".into(),
                "generation_num_tree".into(),
                "parent_ID_tree".into(),
                "root_ID_tree".into(),
                "sister_ID_tree".into(),
                "is_history_known".into(),
            ],
            rows: vec![
                vec![
                    n(0.0),
                    n(1.0),
                    n(1.0),
                    n(1.0),
                    n(-1.0),
                    n(1.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(0.0),
                    n(2.0),
                    n(2.0),
                    n(1.0),
                    n(-1.0),
                    n(2.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(1.0),
                    n(2.0),
                    n(2.0),
                    n(1.0),
                    n(-1.0),
                    n(2.0),
                    n(-1.0),
                    b(false),
                ],
                vec![
                    n(1.0),
                    n(3.0),
                    n(3.0),
                    n(2.0),
                    n(1.0),
                    n(1.0),
                    n(-1.0),
                    b(true),
                ],
                vec![
                    n(1.0),
                    n(4.0),
                    n(4.0),
                    n(1.0),
                    n(-1.0),
                    n(4.0),
                    n(-1.0),
                    b(false),
                ],
            ],
        };
        let state = build_lineage_state(&table)?;
        let info = export_lineage_info(&state, 1)?;
        assert_eq!(info.cells_with_parent, vec![(3, 1)]);
        assert_eq!(info.orphan_cells, vec![4]);
        assert_eq!(info.lost_cells, Vec::<i64>::new());
        Ok(())
    }

    #[test]
    fn updates_lineage_from_json_file() -> Result<()> {
        let temp = tempdir()?;
        let input = temp.path().join("acdc_output.csv");
        let edits = temp.path().join("edits.json");
        let output = temp.path().join("updated.csv");
        write_table(
            &input,
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into()],
                rows: vec![vec![n(0.0), n(1.0)]],
            },
        )?;
        fs::write(
            &edits,
            r#"[{"Cell_ID": 1, "parent_ID_tree": -1, "is_history_known": false}]"#,
        )?;
        update_lineage_frame_file(LineageUpdateConfig {
            input_path: input,
            output_path: output.clone(),
            frame_i: 0,
            edits_table_path: None,
            edits_json_path: Some(edits),
        })?;
        assert!(output.exists());
        Ok(())
    }
}
