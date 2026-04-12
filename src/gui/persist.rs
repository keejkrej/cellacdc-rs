use super::state::{
    AnnotationTool, AppRoute, DisplaySettingsState, GuiDockLayoutState, GuiMode, LineageTool,
    ProjectionMode, ShortcutOverrides, UtilityState,
};
use cellacdc_rs::{
    ImportConflictMode, ImportLayoutKind, ImportOutputFormat, ImportReaderBackend,
    MetadataReusePolicy, ViewPlane,
};
use eframe::Storage;
use serde::{Deserialize, Serialize};

pub(crate) const APP_KEY: &str = "cellacdc_rs_gui";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct PersistedState {
    pub(crate) route: AppRoute,
    pub(crate) recent_paths: Vec<String>,
    pub(crate) last_opened_path: Option<String>,
    pub(crate) phase_channel: String,
    pub(crate) fluo_channel: String,
    pub(crate) model_path: String,
    pub(crate) run_output_suffix: String,
    pub(crate) cpu: bool,
    pub(crate) track: bool,
    pub(crate) track_ioa_threshold: f32,
    pub(crate) tile: usize,
    pub(crate) batch_size: usize,
    pub(crate) cellprob_threshold: f32,
    pub(crate) niter: usize,
    pub(crate) min_size: usize,
    pub(crate) overwrite_outputs: bool,
    pub(crate) overlay_alpha: f32,
    pub(crate) show_segmentation_overlay: bool,
    pub(crate) selected_channel: String,
    pub(crate) selected_segmentation_endname: Option<String>,
    pub(crate) data_structure_backend: ImportReaderBackend,
    pub(crate) data_structure_layout_kind: ImportLayoutKind,
    pub(crate) data_structure_conflict_mode: ImportConflictMode,
    pub(crate) data_structure_metadata_policy: MetadataReusePolicy,
    pub(crate) data_structure_output_format: ImportOutputFormat,
    pub(crate) data_structure_destination_path: String,
    pub(crate) data_prep_active_channel: String,
    pub(crate) data_prep_projection_mode: ProjectionMode,
    pub(crate) data_prep_z_index: usize,
    pub(crate) projection_mode: ProjectionMode,
    pub(crate) z_index: usize,
    pub(crate) annotation_tool: AnnotationTool,
    pub(crate) annotation_brush_radius: usize,
    pub(crate) gui_mode: GuiMode,
    pub(crate) lineage_tool: LineageTool,
    pub(crate) view_plane: ViewPlane,
    pub(crate) show_all_custom_annotations: bool,
    pub(crate) dock_layout: GuiDockLayoutState,
    pub(crate) shortcut_overrides: ShortcutOverrides,
    pub(crate) display: DisplaySettingsState,
    pub(crate) utility: UtilityState,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            route: AppRoute::Launcher,
            recent_paths: Vec::new(),
            last_opened_path: None,
            phase_channel: String::new(),
            fluo_channel: String::new(),
            model_path: String::new(),
            run_output_suffix: String::new(),
            cpu: false,
            track: false,
            track_ioa_threshold: 0.4,
            tile: 256,
            batch_size: 1,
            cellprob_threshold: 0.0,
            niter: 200,
            min_size: 15,
            overwrite_outputs: false,
            overlay_alpha: 0.45,
            show_segmentation_overlay: true,
            selected_channel: String::new(),
            selected_segmentation_endname: None,
            data_structure_backend: ImportReaderBackend::Auto,
            data_structure_layout_kind: ImportLayoutKind::FilePerPosition,
            data_structure_conflict_mode: ImportConflictMode::CreateNewPositions,
            data_structure_metadata_policy: MetadataReusePolicy::ConfirmEverySource,
            data_structure_output_format: ImportOutputFormat::Tiff,
            data_structure_destination_path: String::new(),
            data_prep_active_channel: String::new(),
            data_prep_projection_mode: ProjectionMode::Max,
            data_prep_z_index: 0,
            projection_mode: ProjectionMode::Max,
            z_index: 0,
            annotation_tool: AnnotationTool::Select,
            annotation_brush_radius: 5,
            gui_mode: GuiMode::Viewer,
            lineage_tool: LineageTool::NoTool,
            view_plane: ViewPlane::XY,
            show_all_custom_annotations: false,
            dock_layout: GuiDockLayoutState::default(),
            shortcut_overrides: ShortcutOverrides::default(),
            display: DisplaySettingsState::default(),
            utility: UtilityState::default(),
        }
    }
}

pub(crate) fn load(storage: Option<&dyn Storage>) -> PersistedState {
    storage
        .and_then(|storage| storage.get_string(APP_KEY))
        .and_then(|json| serde_json::from_str::<PersistedState>(&json).ok())
        .unwrap_or_default()
}

pub(crate) fn save(storage: &mut dyn Storage, state: &PersistedState) {
    if let Ok(json) = serde_json::to_string(state) {
        storage.set_string(APP_KEY, json);
    }
}
