use crate::lineage::{
    build_lineage_state, export_lineage_info, lineage_mother_candidates,
    propagate_lineage_from_frame, set_lineage_parent, set_lineage_unknown, LineageFrameEdit,
    LineageState,
};
use crate::mask_io::{load_mask_data, save_mask_data, MaskPathResolution, SegmentationLayout};
use crate::measure::{measure_position, MeasurementRunConfig};
use crate::metadata::read_metadata_summary;
use crate::runner::OverwritePolicy;
use crate::session::{open_position_session, ViewPlane};
use crate::tabular::{read_table, write_table, Table, TableValue};
use crate::tracking::{track_sequence, TrackingConfig};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
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
    CustomAnnotations,
    Snapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CustomAnnotationKind {
    SingleTimePoint,
    MultipleTimePoints,
    MultipleValuesClass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomAnnotationDefinition {
    pub name: String,
    pub kind: CustomAnnotationKind,
    pub symbol: String,
    pub shortcut: Option<String>,
    pub description: String,
    pub keep_active: bool,
    pub hide_when_inactive: bool,
    pub symbol_color_rgba: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CustomAnnotationStore {
    pub definitions: BTreeMap<String, CustomAnnotationDefinition>,
    pub annotated_ids_by_position:
        BTreeMap<String, BTreeMap<String, BTreeMap<usize, BTreeSet<u32>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CustomAnnotationMutation {
    ToggleObject {
        position_key: String,
        annotation_name: String,
        frame_index: usize,
        object_id: u32,
    },
    RenameDefinition {
        old_name: String,
        definition: CustomAnnotationDefinition,
    },
    RemoveDefinitionKeepColumn {
        annotation_name: String,
    },
    RemoveDefinitionAndColumn {
        annotation_name: String,
    },
    UpdateDefinition {
        old_name: String,
        definition: CustomAnnotationDefinition,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotSaveScope {
    CurrentPosition,
    SelectedPositions(BTreeSet<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotProfile {
    pub is_snapshot: bool,
    pub is_3d_snapshot: bool,
    pub editing_allowed_on_current_plane: bool,
    pub save_scope_required: bool,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CellCycleIntegrityReport {
    pub is_valid: bool,
    pub frames_checked: Vec<i64>,
    pub issues: Vec<CellCycleIntegrityIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CellCycleIntegrityIssue {
    pub frame_i: i64,
    pub category: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cell_ids: Vec<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cycles: Vec<CellCycleCycleId>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mother_ids: Vec<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub relation_mismatches: Vec<CellCycleRelationMismatch>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CellCycleCycleId {
    pub cell_id: i64,
    pub generation_num: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CellCycleRelationMismatch {
    pub cell_id: i64,
    pub relative_id: i64,
    pub relative_relative_id: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageEditAction {
    SetParent {
        frame_i: i64,
        cell_id: i64,
        parent_id: i64,
    },
    SetUnknown {
        frame_i: i64,
        cell_id: i64,
    },
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

pub fn global_custom_annotation_definitions_path() -> PathBuf {
    let base_dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cellacdc-rs");
    base_dir.join("custom_annotations.json")
}

pub fn load_custom_annotation_definitions(
    path: impl AsRef<Path>,
) -> Result<BTreeMap<String, CustomAnnotationDefinition>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let content = fs::read_to_string(path).with_context(|| {
        format!(
            "Failed to read custom annotation definitions {}",
            path.display()
        )
    })?;
    let raw_definitions =
        serde_json::from_str::<BTreeMap<String, Value>>(&content).with_context(|| {
            format!(
                "Failed to parse custom annotation definitions {}",
                path.display()
            )
        })?;
    let mut definitions = BTreeMap::new();
    for (name, value) in raw_definitions {
        let definition =
            custom_annotation_definition_from_json(&name, &value).with_context(|| {
                format!(
                    "Failed to parse custom annotation definition {name:?} in {}",
                    path.display()
                )
            })?;
        definitions.insert(definition.name.clone(), definition);
    }
    Ok(definitions)
}

pub fn save_custom_annotation_definitions(
    path: impl AsRef<Path>,
    definitions: &BTreeMap<String, CustomAnnotationDefinition>,
) -> Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let python_definitions = definitions
        .iter()
        .map(|(name, definition)| {
            (
                name.clone(),
                PythonCustomAnnotationDefinition::from(definition),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let content = serde_json::to_string_pretty(&python_definitions)?;
    fs::write(path, content).with_context(|| {
        format!(
            "Failed to save custom annotation definitions {}",
            path.display()
        )
    })?;
    Ok(path.to_path_buf())
}

#[derive(Debug, Serialize)]
struct PythonCustomAnnotationDefinition {
    #[serde(rename = "type")]
    type_label: String,
    name: String,
    symbol: String,
    shortcut: String,
    description: String,
    #[serde(rename = "keepActive")]
    keep_active: bool,
    #[serde(rename = "isHideChecked")]
    is_hide_checked: bool,
    #[serde(rename = "symbolColor")]
    symbol_color: [u8; 4],
}

impl From<&CustomAnnotationDefinition> for PythonCustomAnnotationDefinition {
    fn from(definition: &CustomAnnotationDefinition) -> Self {
        Self {
            type_label: custom_annotation_kind_python_label(definition.kind).to_string(),
            name: definition.name.clone(),
            symbol: custom_annotation_python_symbol(&definition.symbol),
            shortcut: definition.shortcut.clone().unwrap_or_default(),
            description: definition.description.clone(),
            keep_active: definition.keep_active,
            is_hide_checked: definition.hide_when_inactive,
            symbol_color: definition.symbol_color_rgba,
        }
    }
}

fn custom_annotation_definition_from_json(
    map_key: &str,
    value: &Value,
) -> Result<CustomAnnotationDefinition> {
    if let Ok(definition) = serde_json::from_value::<CustomAnnotationDefinition>(value.clone()) {
        return Ok(definition);
    }

    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("custom annotation definition must be a JSON object"))?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(map_key)
        .to_string();
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .map(custom_annotation_kind_from_python_label)
        .transpose()?
        .unwrap_or(CustomAnnotationKind::SingleTimePoint);
    let symbol = object
        .get("symbol")
        .and_then(Value::as_str)
        .map(custom_annotation_symbol_from_python)
        .unwrap_or_else(|| "o".to_string());
    let shortcut = object
        .get("shortcut")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|shortcut| !shortcut.is_empty())
        .map(str::to_string);
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let keep_active = object
        .get("keepActive")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let hide_when_inactive = object
        .get("isHideChecked")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let symbol_color_rgba = object
        .get("symbolColor")
        .and_then(custom_annotation_color_from_json)
        .unwrap_or([255, 0, 0, 255]);

    Ok(CustomAnnotationDefinition {
        name,
        kind,
        symbol,
        shortcut,
        description,
        keep_active,
        hide_when_inactive,
        symbol_color_rgba,
    })
}

fn custom_annotation_kind_from_python_label(label: &str) -> Result<CustomAnnotationKind> {
    match label {
        "Single time-point" => Ok(CustomAnnotationKind::SingleTimePoint),
        "Multiple time-points" => Ok(CustomAnnotationKind::MultipleTimePoints),
        "Multiple values class" => Ok(CustomAnnotationKind::MultipleValuesClass),
        other => bail!("Unsupported custom annotation type {other:?}"),
    }
}

fn custom_annotation_kind_python_label(kind: CustomAnnotationKind) -> &'static str {
    match kind {
        CustomAnnotationKind::SingleTimePoint => "Single time-point",
        CustomAnnotationKind::MultipleTimePoints => "Multiple time-points",
        CustomAnnotationKind::MultipleValuesClass => "Multiple values class",
    }
}

fn custom_annotation_symbol_from_python(symbol: &str) -> String {
    symbol
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string()
}

fn custom_annotation_python_symbol(symbol: &str) -> String {
    let trimmed = symbol.trim();
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') {
        trimmed.to_string()
    } else {
        format!("'{trimmed}'")
    }
}

fn custom_annotation_color_from_json(value: &Value) -> Option<[u8; 4]> {
    let values = value.as_array()?;
    if values.len() < 3 {
        return None;
    }
    let mut color = [255, 0, 0, 255];
    for (idx, channel) in values.iter().take(4).enumerate() {
        color[idx] = channel.as_u64()?.min(u8::MAX as u64) as u8;
    }
    Some(color)
}

pub fn derive_custom_annotation_memberships(
    position_paths: &[PathBuf],
    segm_endname: Option<&str>,
) -> Result<CustomAnnotationStore> {
    let mut store = CustomAnnotationStore::default();
    for position_path in position_paths {
        let position = open_position_session(position_path)?;
        let position_key = position.position_key();
        let local_definitions =
            load_custom_annotation_definitions(position.custom_annotation_params_path())?;
        for (name, definition) in local_definitions {
            store.definitions.entry(name).or_insert(definition);
        }
        let acdc_output_path = position.acdc_output_path(segm_endname);
        if !acdc_output_path.exists() {
            continue;
        }
        let table = read_table(&acdc_output_path)?;
        let frame_idx = table.header_index("frame_i")?;
        let id_idx = table.header_index("Cell_ID")?;
        for (name, definition) in &store.definitions {
            if definition.kind != CustomAnnotationKind::SingleTimePoint {
                continue;
            }
            let Some(column_idx) = table.maybe_header_index(name) else {
                continue;
            };
            let per_annotation = store
                .annotated_ids_by_position
                .entry(position_key.clone())
                .or_default()
                .entry(name.clone())
                .or_default();
            for row in &table.rows {
                let frame = row[frame_idx].as_i64().unwrap_or(0).max(0) as usize;
                let cell_id = row[id_idx].as_i64().unwrap_or(0);
                if cell_id <= 0 {
                    continue;
                }
                if table_value_is_true(&row[column_idx]) {
                    per_annotation
                        .entry(frame)
                        .or_default()
                        .insert(cell_id as u32);
                }
            }
        }
    }
    Ok(store)
}

pub fn apply_custom_annotation_mutation(
    store: &CustomAnnotationStore,
    mutation: CustomAnnotationMutation,
) -> Result<CustomAnnotationStore> {
    let mut updated = store.clone();
    match mutation {
        CustomAnnotationMutation::ToggleObject {
            position_key,
            annotation_name,
            frame_index,
            object_id,
        } => {
            let per_frame = updated
                .annotated_ids_by_position
                .entry(position_key)
                .or_default()
                .entry(annotation_name)
                .or_default();
            let ids = per_frame.entry(frame_index).or_default();
            if !ids.insert(object_id) {
                ids.remove(&object_id);
            }
        }
        CustomAnnotationMutation::RenameDefinition {
            old_name,
            definition,
        }
        | CustomAnnotationMutation::UpdateDefinition {
            old_name,
            definition,
        } => {
            validate_custom_annotation_definition(&definition)?;
            if definition.kind != CustomAnnotationKind::SingleTimePoint {
                bail!("Only Single time-point custom annotations are currently supported");
            }
            updated.definitions.remove(&old_name);
            for per_position in updated.annotated_ids_by_position.values_mut() {
                if let Some(existing) = per_position.remove(&old_name) {
                    per_position.insert(definition.name.clone(), existing);
                }
            }
            updated
                .definitions
                .insert(definition.name.clone(), definition);
        }
        CustomAnnotationMutation::RemoveDefinitionKeepColumn { annotation_name }
        | CustomAnnotationMutation::RemoveDefinitionAndColumn { annotation_name } => {
            updated.definitions.remove(&annotation_name);
            for per_position in updated.annotated_ids_by_position.values_mut() {
                per_position.remove(&annotation_name);
            }
        }
    }
    Ok(updated)
}

pub fn write_custom_annotations_to_acdc_output(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    store: &CustomAnnotationStore,
    delete_missing_columns: bool,
) -> Result<PathBuf> {
    let position = open_position_session(position_dir.as_ref())?;
    let position_key = position.position_key();
    let path = position.acdc_output_path(segm_endname);
    if !path.exists() {
        bail!(
            "Cell-ACDC output table not found for custom annotations: {}",
            path.display()
        );
    }
    let mut table = read_table(&path)?;
    let frame_idx = table.header_index("frame_i")?;
    let id_idx = table.header_index("Cell_ID")?;
    let active = store
        .annotated_ids_by_position
        .get(&position_key)
        .cloned()
        .unwrap_or_default();

    let annotation_columns = table
        .headers
        .iter()
        .filter(|header| store.definitions.contains_key(*header) || active.contains_key(*header))
        .cloned()
        .collect::<Vec<_>>();
    if delete_missing_columns {
        let keep_indices = table
            .headers
            .iter()
            .enumerate()
            .filter_map(|(idx, header)| {
                (!annotation_columns.contains(header) || store.definitions.contains_key(header))
                    .then_some(idx)
            })
            .collect::<Vec<_>>();
        if keep_indices.len() != table.headers.len() {
            table.headers = keep_indices
                .iter()
                .map(|idx| table.headers[*idx].clone())
                .collect();
            table.rows = table
                .rows
                .into_iter()
                .map(|row| keep_indices.iter().map(|idx| row[*idx].clone()).collect())
                .collect();
        }
    }

    for name in store.definitions.keys() {
        if table.maybe_header_index(name).is_none() {
            table.with_column(
                name.clone(),
                vec![TableValue::Number(0.0); table.rows.len()],
            )?;
        }
        let col_idx = table.header_index(name)?;
        let memberships = active.get(name);
        for row in &mut table.rows {
            let frame = row[frame_idx].as_i64().unwrap_or(0).max(0) as usize;
            let cell_id = row[id_idx].as_i64().unwrap_or(0).max(0) as u32;
            let value = memberships
                .and_then(|per_frame| per_frame.get(&frame))
                .map(|ids| ids.contains(&cell_id))
                .unwrap_or(false);
            row[col_idx] = TableValue::Number(if value { 1.0 } else { 0.0 });
        }
    }
    write_table(&path, &table)?;
    Ok(path)
}

pub fn build_snapshot_profile(
    size_t: usize,
    size_z: usize,
    view_plane: ViewPlane,
) -> SnapshotProfile {
    let is_snapshot = size_t <= 1;
    SnapshotProfile {
        is_snapshot,
        is_3d_snapshot: is_snapshot && size_z > 1,
        editing_allowed_on_current_plane: !is_snapshot
            || size_z <= 1
            || view_plane == ViewPlane::XY,
        save_scope_required: is_snapshot,
    }
}

pub fn resolve_snapshot_save_scope(
    selected_positions: &BTreeSet<String>,
    current_position_key: &str,
) -> Result<SnapshotSaveScope> {
    if selected_positions.is_empty() {
        return Ok(SnapshotSaveScope::CurrentPosition);
    }
    if selected_positions.len() == 1 && selected_positions.contains(current_position_key) {
        return Ok(SnapshotSaveScope::CurrentPosition);
    }
    Ok(SnapshotSaveScope::SelectedPositions(
        selected_positions.clone(),
    ))
}

pub fn load_cell_cycle_annotations(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
) -> Result<CellCycleAnnotationTable> {
    let (path, _) = lineage_table_paths_for_position(position_dir, segm_endname)?;
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

pub fn check_cell_cycle_integrity(
    table: &CellCycleAnnotationTable,
    frame_i: Option<i64>,
) -> CellCycleIntegrityReport {
    let mut frames = table
        .records
        .iter()
        .map(|record| record.frame_i)
        .collect::<BTreeSet<_>>();
    if let Some(frame_i) = frame_i {
        frames = frames
            .into_iter()
            .filter(|candidate| *candidate == frame_i)
            .collect();
    }
    let frames_checked = frames.iter().copied().collect::<Vec<_>>();
    let mut issues = Vec::new();
    for frame_i in &frames_checked {
        let records = table
            .records
            .iter()
            .filter(|record| record.frame_i == *frame_i)
            .collect::<Vec<_>>();
        collect_cell_cycle_frame_issues(*frame_i, &records, &mut issues);
    }
    if frame_i.is_none() {
        collect_cell_cycle_global_issues(table, &mut issues);
    }
    CellCycleIntegrityReport {
        is_valid: issues.is_empty(),
        frames_checked,
        issues,
    }
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
    let resolution = MaskPathResolution {
        size_t: Some(position.spec.size_t),
        size_z: Some(position.spec.size_z),
        layout: Some(SegmentationLayout::TYX),
    };
    let mut mask_data = load_mask_data(&asset.path, Some(&resolution))?;
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
        TrackingRunScope::CurrentFrameToEnd { start_frame } => {
            start_frame.min(size_t.saturating_sub(1))
        }
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
        channel_names: None,
        metric_options: None,
        save_object_counts_table: false,
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
    let table_path = acdc_output_path(
        &position.spec.images_dir,
        &position.spec.basename,
        segm_endname,
    );
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
            write_row_number(
                &mut updated,
                source_row,
                "Cell_ID",
                edit.target_label as i64,
            )?;
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
                generation_num: Some(0),
                relative_id: Some(mother_id),
                relationship: Some("bud".to_string()),
                emerg_frame_i: Some(frame_i),
                ..Default::default()
            },
            CellCycleEdit {
                frame_i,
                cell_id: mother_id,
                cell_cycle_stage: Some("S".to_string()),
                relative_id: Some(bud_id),
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
    let (lineage_path, mut state) =
        load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
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
    let (lineage_path, mut state) =
        load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    state = set_lineage_parent(state, edit.frame_i, edit.cell_id, edit.parent_id)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok(state)
}

pub fn lineage_table_paths_for_position(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
) -> Result<(PathBuf, PathBuf)> {
    let (images_dir, basename) = resolve_lineage_position_basename(position_dir.as_ref())?;
    let acdc_path = acdc_output_path(&images_dir, &basename, segm_endname);
    let lineage_path = lineage_output_path(&acdc_path);
    Ok((acdc_path, lineage_path))
}

pub fn propagate_lineage_for_position(
    position_dir: impl AsRef<Path>,
    segm_endname: Option<&str>,
    frame_i: i64,
    cell_ids: &[i64],
) -> Result<LineageState> {
    let (lineage_path, mut state) =
        load_or_initialize_lineage(position_dir.as_ref(), segm_endname)?;
    state = propagate_lineage_from_frame(state, frame_i, cell_ids)?;
    write_table(&lineage_path, &state.to_table())?;
    Ok(state)
}

fn load_or_initialize_lineage(
    position_dir: &Path,
    segm_endname: Option<&str>,
) -> Result<(PathBuf, LineageState)> {
    let (acdc_path, lineage_path) = lineage_table_paths_for_position(position_dir, segm_endname)?;
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

fn resolve_lineage_position_basename(position_dir: &Path) -> Result<(PathBuf, String)> {
    let images_dir = if position_dir.file_name().and_then(|name| name.to_str()) == Some("Images") {
        position_dir.to_path_buf()
    } else {
        position_dir.join("Images")
    };
    if !images_dir.is_dir() {
        bail!(
            "Expected a Cell-ACDC position directory or Images directory, got {}",
            position_dir.display()
        );
    }

    let mut files = Vec::new();
    for entry in fs::read_dir(&images_dir)
        .with_context(|| format!("Failed to read {}", images_dir.display()))?
    {
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(should_skip_python_listdir_entry)
        {
            continue;
        }
        files.push(path);
    }
    files.sort();

    let metadata_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("metadata.csv"))
                .unwrap_or(false)
        })
        .cloned();
    if let Some(path) = metadata_path {
        if let Some(basename) = read_metadata_summary(&path)?.basename {
            if !basename.is_empty() {
                return Ok((images_dir, basename));
            }
        }
    }

    infer_basename_from_acdc_output(&files)
        .map(|basename| (images_dir.clone(), basename))
        .ok_or_else(|| {
            anyhow!(
                "Failed to determine Cell-ACDC basename in {}",
                images_dir.display()
            )
        })
}

fn infer_basename_from_acdc_output(files: &[PathBuf]) -> Option<String> {
    let mut basenames = files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            if name.contains("_lineage") {
                return None;
            }
            name.strip_suffix("acdc_output.csv")
                .map(|basename| basename.to_string())
        })
        .collect::<Vec<_>>();
    basenames.sort_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    basenames.into_iter().next()
}

fn should_skip_python_listdir_entry(name: &str) -> bool {
    name.starts_with('.')
        || name == "desktop.ini"
        || name == "recovery"
        || name.ends_with(".new.npz")
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

fn collect_cell_cycle_frame_issues(
    frame_i: i64,
    records: &[&CellCycleAnnotationRecord],
    issues: &mut Vec<CellCycleIntegrityIssue>,
) {
    let cell_ids = records
        .iter()
        .map(|record| record.cell_id)
        .collect::<BTreeSet<_>>();
    let by_cell_id = records
        .iter()
        .map(|record| (record.cell_id, *record))
        .collect::<BTreeMap<_, _>>();
    let s_records = records
        .iter()
        .copied()
        .filter(|record| record.cell_cycle_stage == "S")
        .collect::<Vec<_>>();

    let lonely_cells = s_records
        .iter()
        .filter(|record| !cell_ids.contains(&record.relative_id))
        .map(|record| record.cell_id)
        .collect::<Vec<_>>();
    if !lonely_cells.is_empty() {
        issues.push(cell_cycle_issue(
            frame_i,
            "S-phase cells whose relative_ID is missing",
            lonely_cells,
        ));
    }

    let num_buds = s_records
        .iter()
        .filter(|record| record.relationship == "bud")
        .count();
    let num_mothers = s_records
        .iter()
        .filter(|record| record.relationship == "mother")
        .count();
    if num_buds != num_mothers {
        let mut counts = BTreeMap::new();
        counts.insert("buds".to_string(), num_buds);
        counts.insert("mothers_in_s".to_string(), num_mothers);
        issues.push(CellCycleIntegrityIssue {
            frame_i,
            category: "number of buds different from number of mothers in S phase".to_string(),
            cell_ids: Vec::new(),
            cycles: Vec::new(),
            mother_ids: Vec::new(),
            relation_mismatches: Vec::new(),
            counts,
        });
    }

    let mut buds_per_mother = BTreeMap::<i64, usize>::new();
    for record in s_records
        .iter()
        .filter(|record| record.relationship == "bud")
    {
        *buds_per_mother.entry(record.relative_id).or_default() += 1;
    }
    let mother_ids = buds_per_mother
        .into_iter()
        .filter_map(|(mother_id, count)| (count > 1).then_some(mother_id))
        .collect::<Vec<_>>();
    if !mother_ids.is_empty() {
        issues.push(CellCycleIntegrityIssue {
            frame_i,
            category: "mother cells with multiple buds".to_string(),
            cell_ids: Vec::new(),
            cycles: Vec::new(),
            mother_ids,
            relation_mismatches: Vec::new(),
            counts: BTreeMap::new(),
        });
    }

    let bud_ids_gen_num_nonzero = s_records
        .iter()
        .filter(|record| record.relationship == "bud" && record.generation_num != 0)
        .map(|record| record.cell_id)
        .collect::<Vec<_>>();
    if !bud_ids_gen_num_nonzero.is_empty() {
        issues.push(cell_cycle_issue(
            frame_i,
            "buds whose generation number is not zero",
            bud_ids_gen_num_nonzero,
        ));
    }

    let mother_ids_gen_num_less_one = s_records
        .iter()
        .filter(|record| record.relationship == "mother" && record.generation_num < 1)
        .map(|record| record.cell_id)
        .collect::<Vec<_>>();
    if !mother_ids_gen_num_less_one.is_empty() {
        issues.push(cell_cycle_issue(
            frame_i,
            "mothers whose generation number is < 1",
            mother_ids_gen_num_less_one,
        ));
    }

    let buds_g1 = records
        .iter()
        .filter(|record| record.relationship == "bud" && record.cell_cycle_stage == "G1")
        .map(|record| record.cell_id)
        .collect::<Vec<_>>();
    if !buds_g1.is_empty() {
        issues.push(cell_cycle_issue(frame_i, "buds in G1", buds_g1));
    }

    let s_cells_without_relative = s_records
        .iter()
        .filter(|record| record.relative_id < 1)
        .map(|record| record.cell_id)
        .collect::<Vec<_>>();
    if !s_cells_without_relative.is_empty() {
        issues.push(cell_cycle_issue(
            frame_i,
            "S-phase cells without positive relative_ID",
            s_cells_without_relative,
        ));
    }

    let relation_mismatches = s_records
        .iter()
        .filter_map(|record| {
            let relative = by_cell_id.get(&record.relative_id)?;
            (relative.relative_id != record.cell_id).then_some(CellCycleRelationMismatch {
                cell_id: record.cell_id,
                relative_id: record.relative_id,
                relative_relative_id: relative.relative_id,
            })
        })
        .collect::<Vec<_>>();
    if !relation_mismatches.is_empty() {
        issues.push(CellCycleIntegrityIssue {
            frame_i,
            category: "ID-relative_ID mismatches".to_string(),
            cell_ids: Vec::new(),
            cycles: Vec::new(),
            mother_ids: Vec::new(),
            relation_mismatches,
            counts: BTreeMap::new(),
        });
    }
}

fn collect_cell_cycle_global_issues(
    table: &CellCycleAnnotationTable,
    issues: &mut Vec<CellCycleIntegrityIssue>,
) {
    let mut cycles = BTreeMap::<(i64, i64), bool>::new();
    for record in table
        .records
        .iter()
        .filter(|record| record.relationship == "mother" && record.is_history_known)
    {
        let key = (record.cell_id, record.generation_num);
        let has_g1 = record.cell_cycle_stage == "G1";
        cycles
            .entry(key)
            .and_modify(|existing_has_g1| *existing_has_g1 |= has_g1)
            .or_insert(has_g1);
    }
    let missing_g1 = cycles
        .into_iter()
        .filter_map(|((cell_id, generation_num), has_g1)| {
            (!has_g1).then_some(CellCycleCycleId {
                cell_id,
                generation_num,
            })
        })
        .collect::<Vec<_>>();
    if !missing_g1.is_empty() {
        issues.push(CellCycleIntegrityIssue {
            frame_i: -1,
            category: "cell cycles without G1".to_string(),
            cell_ids: Vec::new(),
            cycles: missing_g1,
            mother_ids: Vec::new(),
            relation_mismatches: Vec::new(),
            counts: BTreeMap::new(),
        });
    }

    if let Some(will_divide_idx) = table.source_table.maybe_header_index("will_divide") {
        let existing_cycles = table
            .records
            .iter()
            .map(|record| (record.cell_id, record.generation_num))
            .collect::<BTreeSet<_>>();
        let mut bad_cycles = BTreeSet::<(i64, i64)>::new();
        for (row_idx, row) in table.source_table.rows.iter().enumerate() {
            let will_divide = row
                .get(will_divide_idx)
                .and_then(TableValue::as_f64)
                .unwrap_or(0.0);
            if will_divide <= 0.0 {
                continue;
            }
            let cell_id = row_i64(&table.source_table, row_idx, "Cell_ID").unwrap_or_default();
            let generation_num =
                row_i64(&table.source_table, row_idx, "generation_num").unwrap_or_default();
            if !existing_cycles.contains(&(cell_id, generation_num + 1)) {
                bad_cycles.insert((cell_id, generation_num));
            }
        }
        let bad_cycles = bad_cycles
            .into_iter()
            .map(|(cell_id, generation_num)| CellCycleCycleId {
                cell_id,
                generation_num,
            })
            .collect::<Vec<_>>();
        if !bad_cycles.is_empty() {
            issues.push(CellCycleIntegrityIssue {
                frame_i: -1,
                category: "will_divide without next generation".to_string(),
                cell_ids: Vec::new(),
                cycles: bad_cycles,
                mother_ids: Vec::new(),
                relation_mismatches: Vec::new(),
                counts: BTreeMap::new(),
            });
        }
    }
}

fn cell_cycle_issue(
    frame_i: i64,
    category: impl Into<String>,
    cell_ids: Vec<i64>,
) -> CellCycleIntegrityIssue {
    CellCycleIntegrityIssue {
        frame_i,
        category: category.into(),
        cell_ids,
        cycles: Vec::new(),
        mother_ids: Vec::new(),
        relation_mismatches: Vec::new(),
        counts: BTreeMap::new(),
    }
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
    if let Some(emerg_frame_i) = edit.emerg_frame_i {
        if emerg_frame_i < -1 {
            bail!("emerg_frame_i must be >= -1");
        }
    }
    if let Some(division_frame_i) = edit.division_frame_i {
        if division_frame_i < -1 {
            bail!("division_frame_i must be >= -1");
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

fn apply_cell_cycle_edit_to_row(
    table: &mut Table,
    row_idx: usize,
    edit: &CellCycleEdit,
) -> Result<()> {
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
    let acdc_name = acdc_output_name(segm_endname);
    let canonical = images_dir.join(format!("{basename}{acdc_name}.csv"));
    if canonical.exists() {
        return canonical;
    }
    find_visible_acdc_output_by_endname(images_dir, &format!("{acdc_name}.csv"))
        .unwrap_or(canonical)
}

fn find_visible_acdc_output_by_endname(images_dir: &Path, endname: &str) -> Option<PathBuf> {
    let mut matches = fs::read_dir(images_dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if !path.is_file() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            if should_skip_python_listdir_entry(name)
                || name.contains("_lineage")
                || !name.ends_with(endname)
            {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        let left_name = left
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let right_name = right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        left_name
            .len()
            .cmp(&right_name.len())
            .then_with(|| left_name.cmp(right_name))
            .then_with(|| left.cmp(right))
    });
    matches.into_iter().next()
}

fn acdc_output_name(segm_endname: Option<&str>) -> String {
    match segm_endname {
        Some(value) if value.trim().trim_end_matches(".npz").starts_with("segm") => value
            .trim()
            .trim_end_matches(".npz")
            .replacen("segm", "acdc_output", 1),
        Some(value) if !value.trim().is_empty() => {
            format!("acdc_output_{}", value.trim().trim_end_matches(".npz"))
        }
        _ => "acdc_output".to_string(),
    }
}

fn lineage_output_path(acdc_output_path: &Path) -> PathBuf {
    let stem = acdc_output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("acdc_output");
    acdc_output_path.with_file_name(format!("{stem}_lineage.csv"))
}

pub fn builtin_acdc_measurement_columns() -> BTreeSet<String> {
    [
        "frame_i",
        "time_seconds",
        "time_minutes",
        "time_hours",
        "z_slice_used",
        "which_z_proj",
        "Cell_ID",
        "cell_cycle_stage",
        "generation_num",
        "relative_ID",
        "relationship",
        "emerg_frame_i",
        "division_frame_i",
        "is_history_known",
        "corrected_on_frame_i",
        "will_divide",
        "daughter_disappears_before_division",
        "disappears_before_division",
        "is_cell_dead",
        "is_cell_excluded",
        "was_manually_edited",
        "x_centroid",
        "y_centroid",
        "cell_area_pxl",
        "cell_area_um2",
        "cell_vol_vox",
        "cell_vol_fl",
        "cell_vol_vox_3D",
        "cell_vol_fl_3D",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub fn validate_custom_annotation_definition(
    definition: &CustomAnnotationDefinition,
) -> Result<()> {
    let name = definition.name.trim();
    if name.is_empty() {
        bail!("Custom annotation name cannot be empty");
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        bail!("Custom annotation name may only contain letters, numbers, '_' or '-'");
    }
    if builtin_acdc_measurement_columns().contains(name) {
        bail!("Custom annotation name {name:?} is reserved by Cell-ACDC measurements");
    }
    Ok(())
}

fn table_value_is_true(value: &TableValue) -> bool {
    match value {
        TableValue::Bool(value) => *value,
        TableValue::Number(value) => *value != 0.0,
        TableValue::Text(value) => matches!(value.to_ascii_lowercase().as_str(), "1" | "true"),
        TableValue::Empty => false,
    }
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
    use tempfile::tempdir;

    fn sample_table() -> CellCycleAnnotationTable {
        let headers = REQUIRED_CCA_COLUMNS
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
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

    #[test]
    fn reports_cell_cycle_integrity_issues() {
        let source_table = Table {
            headers: REQUIRED_CCA_COLUMNS
                .iter()
                .map(|column| (*column).to_string())
                .collect(),
            rows: vec![
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Text("S".into()),
                    TableValue::Number(0.0),
                    TableValue::Number(2.0),
                    TableValue::Text("mother".into()),
                    TableValue::Number(-1.0),
                    TableValue::Number(-1.0),
                    TableValue::Bool(true),
                ],
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(2.0),
                    TableValue::Text("S".into()),
                    TableValue::Number(1.0),
                    TableValue::Number(1.0),
                    TableValue::Text("bud".into()),
                    TableValue::Number(-1.0),
                    TableValue::Number(-1.0),
                    TableValue::Bool(false),
                ],
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(3.0),
                    TableValue::Text("S".into()),
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Text("bud".into()),
                    TableValue::Number(-1.0),
                    TableValue::Number(-1.0),
                    TableValue::Bool(false),
                ],
            ],
        };
        let table = cell_cycle_table_from_table(PathBuf::from("demo.csv"), source_table).unwrap();

        let report = check_cell_cycle_integrity(&table, Some(0));

        assert!(!report.is_valid);
        assert_eq!(report.frames_checked, vec![0]);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.category == "mother cells with multiple buds"
                && issue.mother_ids == vec![1]));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.category == "ID-relative_ID mismatches"));

        let global_report = check_cell_cycle_integrity(&table, None);
        assert!(global_report.issues.iter().any(|issue| issue.category
            == "cell cycles without G1"
            && issue.cycles
                == vec![CellCycleCycleId {
                    cell_id: 1,
                    generation_num: 0
                }]));
    }

    #[test]
    fn validates_custom_annotation_definitions() {
        let valid = CustomAnnotationDefinition {
            name: "mitotic_entry".into(),
            kind: CustomAnnotationKind::SingleTimePoint,
            symbol: "o".into(),
            shortcut: Some("M".into()),
            description: String::new(),
            keep_active: true,
            hide_when_inactive: false,
            symbol_color_rgba: [255, 0, 0, 255],
        };
        validate_custom_annotation_definition(&valid).unwrap();

        let reserved = CustomAnnotationDefinition {
            name: "Cell_ID".into(),
            ..valid.clone()
        };
        assert!(validate_custom_annotation_definition(&reserved).is_err());
    }

    #[test]
    fn toggles_custom_annotation_membership() {
        let mut store = CustomAnnotationStore::default();
        store.definitions.insert(
            "mitotic_entry".into(),
            CustomAnnotationDefinition {
                name: "mitotic_entry".into(),
                kind: CustomAnnotationKind::SingleTimePoint,
                symbol: "o".into(),
                shortcut: Some("M".into()),
                description: String::new(),
                keep_active: true,
                hide_when_inactive: false,
                symbol_color_rgba: [255, 0, 0, 255],
            },
        );
        let toggled = apply_custom_annotation_mutation(
            &store,
            CustomAnnotationMutation::ToggleObject {
                position_key: "Position_1".into(),
                annotation_name: "mitotic_entry".into(),
                frame_index: 3,
                object_id: 7,
            },
        )
        .unwrap();
        assert!(toggled.annotated_ids_by_position["Position_1"]["mitotic_entry"][&3].contains(&7));

        let untoggled = apply_custom_annotation_mutation(
            &toggled,
            CustomAnnotationMutation::ToggleObject {
                position_key: "Position_1".into(),
                annotation_name: "mitotic_entry".into(),
                frame_index: 3,
                object_id: 7,
            },
        )
        .unwrap();
        assert!(
            !untoggled.annotated_ids_by_position["Position_1"]["mitotic_entry"][&3].contains(&7)
        );
    }

    #[test]
    fn roundtrips_custom_annotation_definition_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("custom_annot_params.json");
        let mut definitions = BTreeMap::new();
        definitions.insert(
            "mitotic_entry".into(),
            CustomAnnotationDefinition {
                name: "mitotic_entry".into(),
                kind: CustomAnnotationKind::SingleTimePoint,
                symbol: "o".into(),
                shortcut: Some("M".into()),
                description: "Marks a mitotic event".into(),
                keep_active: true,
                hide_when_inactive: false,
                symbol_color_rgba: [255, 0, 0, 255],
            },
        );
        save_custom_annotation_definitions(&path, &definitions).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"type\": \"Single time-point\""));
        assert!(content.contains("\"isHideChecked\": false"));
        assert!(content.contains("\"symbolColor\""));
        assert!(!content.contains("\"kind\""));
        let restored = load_custom_annotation_definitions(&path).unwrap();
        assert_eq!(restored, definitions);
    }

    #[test]
    fn loads_python_custom_annotation_definition_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("custom_annot_params.json");
        fs::write(
            &path,
            r#"{
  "division": {
    "type": "Single time-point",
    "name": "division",
    "symbol": "'t'",
    "shortcut": "D",
    "description": "Division event",
    "keepActive": true,
    "isHideChecked": false,
    "symbolColor": [20, 40, 60, 255]
  }
}"#,
        )
        .unwrap();

        let restored = load_custom_annotation_definitions(&path).unwrap();
        let definition = restored.get("division").unwrap();
        assert_eq!(definition.name, "division");
        assert_eq!(definition.kind, CustomAnnotationKind::SingleTimePoint);
        assert_eq!(definition.symbol, "t");
        assert_eq!(definition.shortcut.as_deref(), Some("D"));
        assert_eq!(definition.description, "Division event");
        assert!(definition.keep_active);
        assert!(!definition.hide_when_inactive);
        assert_eq!(definition.symbol_color_rgba, [20, 40, 60, 255]);
    }

    #[test]
    fn lineage_basename_ignores_python_listdir_excluded_metadata() {
        let dir = tempdir().unwrap();
        let position = dir.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images).unwrap();
        fs::write(
            images.join(".hidden_metadata.csv"),
            "Description,values\nbasename,hidden_\n",
        )
        .unwrap();
        fs::write(
            images.join("visible_acdc_output.csv"),
            "frame_i,Cell_ID\n0,1\n",
        )
        .unwrap();

        let (_, basename) = resolve_lineage_position_basename(&position).unwrap();

        assert_eq!(basename, "visible_");
    }

    #[test]
    fn lineage_basename_prefers_shortest_visible_acdc_output_match() {
        let dir = tempdir().unwrap();
        let images = dir.path().join("Position_1").join("Images");
        fs::create_dir_all(&images).unwrap();
        fs::write(
            images.join("longer_prefix_acdc_output.csv"),
            "frame_i,Cell_ID\n0,1\n",
        )
        .unwrap();
        fs::write(images.join("b_acdc_output.csv"), "frame_i,Cell_ID\n0,1\n").unwrap();
        fs::write(images.join(".a_acdc_output.csv"), "frame_i,Cell_ID\n0,1\n").unwrap();
        fs::write(
            images.join("a_lineage_acdc_output.csv"),
            "frame_i,Cell_ID\n0,1\n",
        )
        .unwrap();

        let (_, basename) = resolve_lineage_position_basename(&images).unwrap();

        assert_eq!(basename, "b_");
    }

    #[test]
    fn lineage_paths_normalize_python_segmentation_endnames() {
        let dir = tempdir().unwrap();
        let position = dir.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images).unwrap();
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\n",
        )
        .unwrap();

        let (acdc_path, lineage_path) =
            lineage_table_paths_for_position(&position, Some("segm_rust.npz")).unwrap();

        assert_eq!(acdc_path, images.join("demo_acdc_output_rust.csv"));
        assert_eq!(
            lineage_path,
            images.join("demo_acdc_output_rust_lineage.csv")
        );
    }

    #[test]
    fn lineage_paths_use_visible_legacy_position_token_acdc_output() {
        let dir = tempdir().unwrap();
        let position = dir.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images).unwrap();
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\n",
        )
        .unwrap();
        fs::write(
            images.join("demo_s01_acdc_output.csv"),
            "frame_i,Cell_ID\n0,1\n",
        )
        .unwrap();

        let (acdc_path, lineage_path) = lineage_table_paths_for_position(&position, None).unwrap();

        assert_eq!(acdc_path, images.join("demo_s01_acdc_output.csv"));
        assert_eq!(
            lineage_path,
            images.join("demo_s01_acdc_output_lineage.csv")
        );
    }

    #[test]
    fn builds_snapshot_profiles_with_plane_gating() {
        let profile = build_snapshot_profile(1, 12, ViewPlane::XZ);
        assert!(profile.is_snapshot);
        assert!(profile.is_3d_snapshot);
        assert!(!profile.editing_allowed_on_current_plane);

        let xy_profile = build_snapshot_profile(1, 12, ViewPlane::XY);
        assert!(xy_profile.editing_allowed_on_current_plane);
    }
}
