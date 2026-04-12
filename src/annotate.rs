use crate::lineage::{
    build_lineage_state, export_lineage_info, lineage_mother_candidates,
    propagate_lineage_from_frame, set_lineage_parent, set_lineage_unknown, LineageFrameEdit,
    LineageState,
};
use crate::mask_io::{load_mask_data, save_mask_data, SegmentationLayout};
use crate::measure::{measure_position, MeasurementRunConfig};
use crate::runner::OverwritePolicy;
use crate::session::open_position_session;
use crate::tabular::{read_table, write_table, Table, TableValue};
use crate::tracking::{track_sequence, TrackingConfig};
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};

const REQUIRED_CCA_COLUMNS: &[&str] = &[
    "frame_i",
    "Cell_ID",
    "cell_cycle_stage",
    "generation_num",
    "relative_ID",
    "relationship",
    "emerg_frame_i",
    "division_frame_i",
    "is_history_known",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiModeKind {
    Viewer,
    SegmentationAndTracking,
    CellCycleAnalysis,
    NormalDivisionLineageTree,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellCycleAnnotationRecord {
    pub frame_i: i64,
    pub cell_id: i64,
    pub cell_cycle_stage: String,
    pub generation_num: i64,
    pub relative_id: i64,
    pub relationship: String,
    pub emerg_frame_i: i64,
    pub division_frame_i: i64,
    pub is_history_known: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellCycleAnnotationTable {
    pub path: PathBuf,
    pub headers: Vec<String>,
    pub records: Vec<CellCycleAnnotationRecord>,
    pub source_table: Table,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellCycleEdit {
    pub frame_i: i64,
    pub cell_id: i64,
    pub cell_cycle_stage: Option<String>,
    pub generation_num: Option<i64>,
    pub relative_id: Option<i64>,
    pub relationship: Option<String>,
    pub emerg_frame_i: Option<i64>,
    pub division_frame_i: Option<i64>,
    pub is_history_known: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellCyclePropagationConfig {
    pub start_frame_i: i64,
    pub end_frame_i: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageEditAction {
    SetParent { frame_i: i64, cell_id: i64, parent_id: i64 },
    SetUnknown { frame_i: i64, cell_id: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageReview {
    pub frame_i: i64,
    pub cells_with_parent: Vec<(i64, i64)>,
    pub orphan_cells: Vec<i64>,
    pub lost_cells: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualTrackingEdit {
    pub frame_index: usize,
    pub source_label: u32,
    pub target_label: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualTrackingPreview {
    pub changed_pixels: usize,
    pub source_label: u32,
    pub target_label: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackingRunScope {
    CurrentPosition,
    CurrentFrameToEnd { start_frame: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackingRunReport {
    pub output_segmentation_path: PathBuf,
    pub measurement_table_path: Option<PathBuf>,
    pub frames_processed: usize,
}

pub fn load_cell_cycle_annotations(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
) -> Result<CellCycleAnnotationTable> {
    let position = open_position_session(position_dir.as_ref())?;
    let path = acdc_output_path(&position.spec.images_dir, &position.spec.basename, segm_endname);
    let table = read_table(&path)
        .with_context(|| format!("Failed to load Cell-ACDC output {}", path.display()))?;
    ensure_required_columns(&table, REQUIRED_CCA_COLUMNS)?;
    Ok(cell_cycle_table_from_table(path, table)?)
}

pub fn save_cell_cycle_annotations(table: &CellCycleAnnotationTable) -> Result<PathBuf> {
    write_table(&table.path, &table.source_table)?;
    Ok(table.path.clone())
}

pub fn apply_cell_cycle_edits(
    table: &CellCycleAnnotationTable,
    edits: &[CellCycleEdit],
) -> Result<CellCycleAnnotationTable> {
    let mut updated = table.source_table.clone();
    for edit in edits {
        validate_cell_cycle_edit(edit)?;
        let Some(row_idx) = find_row_index(&updated, edit.frame_i, edit.cell_id) else {
            bail!(
                "Missing cell-cycle row for frame {} and Cell_ID {}",
                edit.frame_i,
                edit.cell_id
            );
        };
        apply_cell_cycle_edit_to_row(&mut updated, row_idx, edit)?;
    }
    cell_cycle_table_from_table(table.path.clone(), updated)
}

pub fn propagate_cell_cycle_edits(
    table: &CellCycleAnnotationTable,
    edits: &[CellCycleEdit],
    config: &CellCyclePropagationConfig,
) -> Result<CellCycleAnnotationTable> {
    let mut updated = apply_cell_cycle_edits(table, edits)?.source_table;
    for edit in edits {
        for row_idx in 0..updated.rows.len() {
            let frame_i = row_i64(&updated, row_idx, "frame_i")?;
            let cell_id = row_i64(&updated, row_idx, "Cell_ID")?;
            if cell_id != edit.cell_id {
                continue;
            }
            if frame_i <= config.start_frame_i {
                continue;
            }
            if let Some(end_frame_i) = config.end_frame_i {
                if frame_i > end_frame_i {
                    continue;
                }
            }
            apply_cell_cycle_edit_to_row(&mut updated, row_idx, edit)?;
        }
    }
    cell_cycle_table_from_table(table.path.clone(), updated)
}

pub fn repeat_tracking_current_position(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    tracking: &TrackingConfig,
    scope: TrackingRunScope,
) -> Result<TrackingRunReport> {
    let position = open_position_session(position_dir.as_ref())?;
    let asset = position
        .segmentation_asset(segm_endname)
        .ok_or_else(|| anyhow!("No segmentation selected for tracking"))?;
    let mut mask_data = load_mask_data(&asset.path, None)?;
    if mask_data.layout != SegmentationLayout::TYX {
        bail!(
            "Interactive repeat tracking currently supports TYX segmentations, got {:?}",
            mask_data.layout
        );
    }
    let shape = mask_data.values.shape().to_vec();
    let size_t = shape[0];
    let height = shape[1];
    let width = shape[2];
    let plane_len = height * width;
    let values = mask_data
        .values
        .as_slice_memory_order_mut()
        .ok_or_else(|| anyhow!("Segmentation data is not contiguous"))?;

    let start_frame = match scope {
        TrackingRunScope::CurrentPosition => 0,
        TrackingRunScope::CurrentFrameToEnd { start_frame } => start_frame.min(size_t.saturating_sub(1)),
    };
    let anchor = start_frame.saturating_sub(1);
    let frames = (anchor..size_t)
        .map(|frame_i| {
            let start = frame_i * plane_len;
            values[start..start + plane_len].to_vec()
        })
        .collect::<Vec<_>>();
    let tracked = track_sequence(&frames, height, width, tracking);
    for (offset, frame_i) in (anchor..size_t).enumerate() {
        if start_frame > 0 && frame_i == anchor {
            continue;
        }
        let start = frame_i * plane_len;
        values[start..start + plane_len].copy_from_slice(&tracked.frames[offset]);
    }

    save_mask_data(&asset.path, &mask_data)?;
    let measurement = measure_position(MeasurementRunConfig {
        position_path: position.spec.position_dir.clone(),
        segm_endname: segm_endname.map(str::to_string),
        overwrite_policy: OverwritePolicy::Overwrite,
        stop_frame: None,
    })?;
    Ok(TrackingRunReport {
        output_segmentation_path: asset.path.clone(),
        measurement_table_path: Some(measurement.outputs.acdc_output_csv_path),
        frames_processed: size_t.saturating_sub(start_frame),
    })
}

pub fn apply_manual_tracking_edit(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    edit: &ManualTrackingEdit,
) -> Result<ManualTrackingPreview> {
    if edit.source_label == 0 {
        bail!("Manual tracking requires a non-zero source label");
    }
    if edit.target_label == 0 {
        bail!("Manual tracking requires a non-zero target label");
    }

    let position = open_position_session(position_dir.as_ref())?;
    let table_path = acdc_output_path(&position.spec.images_dir, &position.spec.basename, segm_endname);
    if !table_path.exists() {
        return Ok(ManualTrackingPreview {
            changed_pixels: 0,
            source_label: edit.source_label,
            target_label: edit.target_label,
        });
    }
    let table = load_cell_cycle_annotations(position_dir.as_ref(), segm_endname)?;
    let mut updated = table.source_table.clone();
    let source_row = find_row_index(&updated, edit.frame_index as i64, edit.source_label as i64);
    let target_row = find_row_index(&updated, edit.frame_index as i64, edit.target_label as i64);
    if let Some(source_row) = source_row {
        if target_row.is_some() && edit.source_label != edit.target_label {
            updated.rows.remove(source_row);
        } else {
            write_row_number(&mut updated, source_row, "Cell_ID", edit.target_label as i64)?;
        }
        let updated_table = cell_cycle_table_from_table(table.path.clone(), updated)?;
        save_cell_cycle_annotations(&updated_table)?;
    }

    Ok(ManualTrackingPreview {
        changed_pixels: 0,
        source_label: edit.source_label,
        target_label: edit.target_label,
    })
}

pub fn assign_mother_bud(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
    bud_id: i64,
    mother_id: i64,
) -> Result<CellCycleAnnotationTable> {
    let table = load_cell_cycle_annotations(position_dir, segm_endname)?;
    let updated = apply_cell_cycle_edits(
        &table,
        &[
            CellCycleEdit {
                frame_i,
                cell_id: bud_id,
                cell_cycle_stage: Some("S".to_string()),
                relative_id: Some(mother_id),
                relationship: Some("bud".to_string()),
                emerg_frame_i: Some(frame_i),
                ..Default::default()
            },
            CellCycleEdit {
                frame_i,
                cell_id: mother_id,
                relationship: Some("mother".to_string()),
                is_history_known: Some(true),
                ..Default::default()
            },
        ],
    )?;
    save_cell_cycle_annotations(&updated)?;
    Ok(updated)
}

pub fn mark_unknown_lineage(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
    cell_id: i64,
) -> Result<LineageState> {
    let (lineage_path, mut state) = load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    state = set_lineage_unknown(state, frame_i, cell_id)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok(state)
}

pub fn find_next_mother_candidate(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
    cell_id: i64,
) -> Result<Option<i64>> {
    let (_, state) = load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    let candidates = lineage_mother_candidates(&state, frame_i, cell_id)?;
    Ok(candidates.candidates.into_iter().next())
}

pub fn review_lineage_frame(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
) -> Result<LineageReview> {
    let (_, state) = load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    let info = export_lineage_info(&state, frame_i)?;
    Ok(LineageReview {
        frame_i,
        cells_with_parent: info.cells_with_parent,
        orphan_cells: info.orphan_cells,
        lost_cells: info.lost_cells,
    })
}

pub fn set_lineage_parent_for_position(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    edit: LineageFrameEdit,
) -> Result<LineageState> {
    let (lineage_path, mut state) = load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    state = set_lineage_parent(state, edit.frame_i, edit.cell_id, edit.parent_id)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok(state)
}

pub fn propagate_lineage_for_position(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
    cell_ids: &[i64],
) -> Result<LineageState> {
    let (lineage_path, mut state) = load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    state = propagate_lineage_from_frame(state, frame_i, cell_ids)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok(state)
}

fn load_or_initialize_lineage(
    position_dir: &Path,
    segm_endname: Option<&str>,
) -> Result<(PathBuf, LineageState)> {
    let position = open_position_session(position_dir)?;
    let acdc_path = acdc_output_path(&position.spec.images_dir, &position.spec.basename, segm_endname);
    let lineage_path = lineage_output_path(&acdc_path);
    if lineage_path.exists() {
        let table = read_table(&lineage_path)?;
        let state = build_lineage_state(&table)?;
        return Ok((lineage_path, state));
    }
    let source = read_table(&acdc_path)?;
    let state = build_lineage_state(&source)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok((lineage_path, state))
}

fn cell_cycle_table_from_table(path: PathBuf, table: Table) -> Result<CellCycleAnnotationTable> {
    ensure_required_columns(&table, REQUIRED_CCA_COLUMNS)?;
    let mut records = Vec::with_capacity(table.rows.len());
    for row_idx in 0..table.rows.len() {
        records.push(CellCycleAnnotationRecord {
            frame_i: row_i64(&table, row_idx, "frame_i")?,
            cell_id: row_i64(&table, row_idx, "Cell_ID")?,
            cell_cycle_stage: row_string(&table, row_idx, "cell_cycle_stage"),
            generation_num: row_i64(&table, row_idx, "generation_num")?,
            relative_id: row_i64(&table, row_idx, "relative_ID")?,
            relationship: row_string(&table, row_idx, "relationship"),
            emerg_frame_i: row_i64(&table, row_idx, "emerg_frame_i")?,
            division_frame_i: row_i64(&table, row_idx, "division_frame_i")?,
            is_history_known: row_bool(&table, row_idx, "is_history_known")?,
        });
    }
    Ok(CellCycleAnnotationTable {
        path,
        headers: table.headers.clone(),
        records,
        source_table: table,
    })
}

fn validate_cell_cycle_edit(edit: &CellCycleEdit) -> Result<()> {
    if let Some(generation_num) = edit.generation_num {
        if generation_num < 0 {
            bail!("generation_num must be >= 0");
        }
    }
    if let Some(relative_id) = edit.relative_id {
        if relative_id < -1 {
            bail!("relative_ID must be >= -1");
        }
    }
    if let Some(relationship) = &edit.relationship {
        match relationship.as_str() {
            "" | "mother" | "bud" | "NA" | "none" => {}
            other => bail!("Unsupported relationship value {other:?}"),
        }
        if relationship == "bud" && edit.relative_id == Some(-1) {
            bail!("Bud relationships require a positive relative_ID");
        }
    }
    Ok(())
}

fn apply_cell_cycle_edit_to_row(table: &mut Table, row_idx: usize, edit: &CellCycleEdit) -> Result<()> {
    if let Some(value) = &edit.cell_cycle_stage {
        write_row_text(table, row_idx, "cell_cycle_stage", value)?;
    }
    if let Some(value) = edit.generation_num {
        write_row_number(table, row_idx, "generation_num", value)?;
    }
    if let Some(value) = edit.relative_id {
        write_row_number(table, row_idx, "relative_ID", value)?;
    }
    if let Some(value) = &edit.relationship {
        write_row_text(table, row_idx, "relationship", value)?;
    }
    if let Some(value) = edit.emerg_frame_i {
        write_row_number(table, row_idx, "emerg_frame_i", value)?;
    }
    if let Some(value) = edit.division_frame_i {
        write_row_number(table, row_idx, "division_frame_i", value)?;
    }
    if let Some(value) = edit.is_history_known {
        write_row_bool(table, row_idx, "is_history_known", value)?;
    }
    Ok(())
}

fn acdc_output_path(images_dir: &Path, basename: &str, segm_endname: Option<&str>) -> PathBuf {
    let suffix = segm_endname
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!("_{value}"))
        .unwrap_or_default();
    images_dir.join(format!("{basename}acdc_output{suffix}.csv"))
}

fn lineage_output_path(acdc_output_path: &Path) -> PathBuf {
    let stem = acdc_output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("acdc_output");
    acdc_output_path.with_file_name(format!("{stem}_lineage.csv"))
}

fn ensure_required_columns(table: &Table, columns: &[&str]) -> Result<()> {
    for column in columns {
        if table.maybe_header_index(column).is_none() {
            bail!("Missing table column {column:?}");
        }
    }
    Ok(())
}

fn find_row_index(table: &Table, frame_i: i64, cell_id: i64) -> Option<usize> {
    (0..table.rows.len()).find(|row_idx| {
        row_i64(table, *row_idx, "frame_i").ok() == Some(frame_i)
            && row_i64(table, *row_idx, "Cell_ID").ok() == Some(cell_id)
    })
}

fn row_i64(table: &Table, row_idx: usize, column: &str) -> Result<i64> {
    let col_idx = table.header_index(column)?;
    table.rows[row_idx][col_idx]
        .as_i64()
        .ok_or_else(|| anyhow!("Missing numeric value in column {column:?} at row {row_idx}"))
}

fn row_bool(table: &Table, row_idx: usize, column: &str) -> Result<bool> {
    let col_idx = table.header_index(column)?;
    match &table.rows[row_idx][col_idx] {
        TableValue::Bool(value) => Ok(*value),
        TableValue::Text(value) => Ok(matches!(value.to_ascii_lowercase().as_str(), "true" | "1")),
        TableValue::Number(value) => Ok(*value != 0.0),
        TableValue::Empty => Ok(false),
    }
}

fn row_string(table: &Table, row_idx: usize, column: &str) -> String {
    table
        .header_index(column)
        .ok()
        .map(|col_idx| table.rows[row_idx][col_idx].as_string_lossy())
        .unwrap_or_default()
}

fn write_row_text(table: &mut Table, row_idx: usize, column: &str, value: &str) -> Result<()> {
    let col_idx = table.header_index(column)?;
    table.rows[row_idx][col_idx] = TableValue::Text(value.to_string());
    Ok(())
}

fn write_row_number(table: &mut Table, row_idx: usize, column: &str, value: i64) -> Result<()> {
    let col_idx = table.header_index(column)?;
    table.rows[row_idx][col_idx] = TableValue::Number(value as f64);
    Ok(())
}

fn write_row_bool(table: &mut Table, row_idx: usize, column: &str, value: bool) -> Result<()> {
    let col_idx = table.header_index(column)?;
    table.rows[row_idx][col_idx] = TableValue::Bool(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> CellCycleAnnotationTable {
        let headers = REQUIRED_CCA_COLUMNS.iter().map(|value| value.to_string()).collect::<Vec<_>>();
        let source_table = Table {
            headers: headers.clone(),
            rows: vec![
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Text("G1".into()),
                    TableValue::Number(1.0),
                    TableValue::Number(-1.0),
                    TableValue::Text("mother".into()),
                    TableValue::Number(-1.0),
                    TableValue::Number(-1.0),
                    TableValue::Bool(false),
                ],
                vec![
                    TableValue::Number(1.0),
                    TableValue::Number(1.0),
                    TableValue::Text("G1".into()),
                    TableValue::Number(1.0),
                    TableValue::Number(-1.0),
                    TableValue::Text("mother".into()),
                    TableValue::Number(-1.0),
                    TableValue::Number(-1.0),
                    TableValue::Bool(false),
                ],
            ],
        };
        cell_cycle_table_from_table(PathBuf::from("demo.csv"), source_table).unwrap()
    }

    #[test]
    fn applies_and_propagates_cell_cycle_edits() {
        let table = sample_table();
        let edited = apply_cell_cycle_edits(
            &table,
            &[CellCycleEdit {
                frame_i: 0,
                cell_id: 1,
                generation_num: Some(2),
                ..Default::default()
            }],
        )
        .unwrap();
        assert_eq!(edited.records[0].generation_num, 2);

        let propagated = propagate_cell_cycle_edits(
            &edited,
            &[CellCycleEdit {
                frame_i: 0,
                cell_id: 1,
                relationship: Some("bud".into()),
                relative_id: Some(4),
                ..Default::default()
            }],
            &CellCyclePropagationConfig {
                start_frame_i: 0,
                end_frame_i: None,
            },
        )
        .unwrap();
        assert_eq!(propagated.records[1].relationship, "bud");
        assert_eq!(propagated.records[1].relative_id, 4);
    }
}
