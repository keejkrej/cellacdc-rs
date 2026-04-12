use cellacdc_rs::{
    BackgroundRoiSet, CropPreview, CropRoiRect, CustomAnnotationStore, FrameProjection,
    FreehandRoiMask, MaskEditSession, ViewPlane,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ProjectionMode {
    Max,
    ZSlice,
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::Max
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AppRoute {
    Launcher,
    DataStructure,
    DataPrep,
    #[serde(alias = "Viewer")]
    Segmentation,
    Annotation,
    Utilities,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AnnotationTool {
    Select,
    Brush,
    Eraser,
    Relabel,
    Merge,
    Delete,
}

impl Default for AnnotationTool {
    fn default() -> Self {
        Self::Select
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum GuiMode {
    Viewer,
    SegmentationAndTracking,
    CellCycleAnalysis,
    NormalDivisionLineageTree,
    CustomAnnotations,
    Snapshot,
}

impl Default for GuiMode {
    fn default() -> Self {
        Self::Viewer
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LineageTool {
    NoTool,
    FindNextMother,
    UnknownLineage,
}

impl Default for LineageTool {
    fn default() -> Self {
        Self::NoTool
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum GuiActionId {
    OpenSession,
    LoadDifferentPosition,
    RevealCurrentPosition,
    Save,
    QuickSave,
    SaveAsVersion,
    LoadOlderVersions,
    ExportImage,
    ExportVideo,
    Undo,
    Redo,
    CustomizeKeyboardShortcuts,
    OverlayLabelsAppearance,
    ToggleObjectsDock,
    ToggleLogDock,
    FindId,
    HighlightSelectedId,
    ToggleOverlayLabels,
    ToggleSegmentationOverlay,
    ToggleSingleChannelOverlay,
    ToggleTrueTransparency,
    ToggleScaleBar,
    ToggleTimestamp,
    RunSegmentationCurrentPosition,
    OpenSegmentationWorkspace,
    MeasureCurrentPosition,
    AutosaveInterval,
    OpenLogs,
    AboutRustPort,
    CurrentLimitations,
    ToolSelect,
    ToolBrush,
    ToolEraser,
    ToolRelabel,
    ToolMerge,
    ToolDelete,
    ModeViewer,
    ModeSegmentationAndTracking,
    ModeCellCycleAnalysis,
    ModeNormalDivisionLineageTree,
    ModeCustomAnnotations,
    RepeatTracking,
    TrackCurrentFrame,
    ManualTracking,
    EditRealTimeTrackerParameters,
    AssignMotherToBud,
    EditCellCycleAnnotations,
    ViewCellCycleAnnotations,
    FindNextPotentialMother,
    UnknownLineage,
    NoLineageTool,
    PropagateLineage,
    ViewLineageChanges,
    LoadSavedCustomAnnotations,
    AddCustomAnnotation,
    ShowAllCustomAnnotations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GuiActionState {
    pub(crate) enabled: bool,
    pub(crate) checked: bool,
    pub(crate) visible: bool,
}

impl Default for GuiActionState {
    fn default() -> Self {
        Self {
            enabled: true,
            checked: false,
            visible: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShortcutBinding {
    pub(crate) key: String,
    pub(crate) command: bool,
    pub(crate) shift: bool,
    pub(crate) alt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShortcutOverride {
    pub(crate) action: GuiActionId,
    pub(crate) binding: ShortcutBinding,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShortcutOverrides {
    pub(crate) bindings: Vec<ShortcutOverride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum GuiDockId {
    Objects,
    Log,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct GuiDockLayoutState {
    pub(crate) object_dock_width: f32,
    pub(crate) log_dock_height: f32,
}

impl Default for GuiDockLayoutState {
    fn default() -> Self {
        Self {
            object_dock_width: 300.0,
            log_dock_height: 180.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AutosaveUnit {
    Seconds,
    Minutes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AutosaveSettings {
    pub(crate) value: u32,
    pub(crate) unit: AutosaveUnit,
}

impl AutosaveSettings {
    pub(crate) fn as_seconds(&self) -> u64 {
        match self.unit {
            AutosaveUnit::Seconds => self.value.max(1) as u64,
            AutosaveUnit::Minutes => (self.value.max(1) as u64) * 60,
        }
    }
}

impl Default for AutosaveSettings {
    fn default() -> Self {
        Self {
            value: 5,
            unit: AutosaveUnit::Seconds,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct DisplaySettingsState {
    pub(crate) show_overlay_labels: bool,
    pub(crate) overlay_label_color: [u8; 4],
    pub(crate) overlay_label_scale: u32,
    pub(crate) highlight_on_hover: bool,
    pub(crate) highlight_searched_object: bool,
    pub(crate) overlay_single_channel_mode: bool,
    pub(crate) true_transparency: bool,
    pub(crate) add_scale_bar: bool,
    pub(crate) add_timestamp: bool,
    pub(crate) show_file_toolbar: bool,
    pub(crate) show_navigation_toolbar: bool,
    pub(crate) show_edit_toolbar: bool,
    pub(crate) show_overlay_toolbar: bool,
    pub(crate) show_highlight_toolbar: bool,
    pub(crate) show_objects_dock: bool,
    pub(crate) show_log_dock: bool,
    pub(crate) autosave: AutosaveSettings,
}

impl Default for DisplaySettingsState {
    fn default() -> Self {
        Self {
            show_overlay_labels: false,
            overlay_label_color: [255, 255, 255, 255],
            overlay_label_scale: 2,
            highlight_on_hover: true,
            highlight_searched_object: true,
            overlay_single_channel_mode: false,
            true_transparency: false,
            add_scale_bar: false,
            add_timestamp: false,
            show_file_toolbar: true,
            show_navigation_toolbar: true,
            show_edit_toolbar: true,
            show_overlay_toolbar: true,
            show_highlight_toolbar: true,
            show_objects_dock: true,
            show_log_dock: true,
            autosave: AutosaveSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GuiDialogState {
    pub(crate) shortcut_editor_open: bool,
    pub(crate) version_browser_open: bool,
    pub(crate) find_id_open: bool,
    pub(crate) export_image_open: bool,
    pub(crate) export_video_open: bool,
    pub(crate) autosave_interval_open: bool,
    pub(crate) overlay_labels_open: bool,
    pub(crate) tracking_params_open: bool,
    pub(crate) cell_cycle_editor_open: bool,
    pub(crate) cell_cycle_viewer_open: bool,
    pub(crate) lineage_review_open: bool,
    pub(crate) custom_annotation_editor_open: bool,
    pub(crate) load_saved_custom_annotations_open: bool,
    pub(crate) snapshot_save_scope_open: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ObjectDockState {
    pub(crate) selected_tab: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HighlightState {
    pub(crate) hovered_label: Option<u32>,
    pub(crate) searched_label: Option<u32>,
    pub(crate) highlighted_input: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ExportImageDialogState {
    pub(crate) path: String,
    pub(crate) include_overlay: bool,
    pub(crate) include_labels: bool,
    pub(crate) include_scale_bar: bool,
    pub(crate) include_timestamp: bool,
}

impl Default for ExportImageDialogState {
    fn default() -> Self {
        Self {
            path: String::new(),
            include_overlay: true,
            include_labels: false,
            include_scale_bar: false,
            include_timestamp: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExportVideoDialogState {
    pub(crate) output_path: String,
    pub(crate) start_frame: usize,
    pub(crate) end_frame: usize,
    pub(crate) include_overlay: bool,
    pub(crate) include_labels: bool,
    pub(crate) include_scale_bar: bool,
    pub(crate) include_timestamp: bool,
}

impl Default for ExportVideoDialogState {
    fn default() -> Self {
        Self {
            output_path: String::new(),
            start_frame: 0,
            end_frame: 0,
            include_overlay: true,
            include_labels: false,
            include_scale_bar: false,
            include_timestamp: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct VersionBrowserState {
    pub(crate) selected_endname: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ShortcutEditorState {
    pub(crate) capturing: Option<GuiActionId>,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ManualTrackingState {
    pub(crate) active: bool,
    pub(crate) target_label: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CellCycleDialogState {
    pub(crate) selected_row: Option<usize>,
    pub(crate) apply_to_future: bool,
    pub(crate) propagate_end_frame: String,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CellCycleTableState {
    pub(crate) records: Vec<cellacdc_rs::CellCycleAnnotationRecord>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LineageReviewDialogState {
    pub(crate) review: Option<cellacdc_rs::LineageReview>,
}

#[derive(Debug, Clone)]
pub(crate) struct TrackingParamsDialogState {
    pub(crate) ioa_threshold: f32,
}

impl Default for TrackingParamsDialogState {
    fn default() -> Self {
        Self { ioa_threshold: 0.4 }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ModeToolbarState {
    pub(crate) show: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CustomAnnotationToolbarState {
    pub(crate) show_all: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ActiveCustomAnnotationState {
    pub(crate) active_name: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CustomAnnotationDialogState {
    pub(crate) editing_name: Option<String>,
    pub(crate) name: String,
    pub(crate) kind_index: usize,
    pub(crate) symbol: String,
    pub(crate) shortcut: String,
    pub(crate) description: String,
    pub(crate) keep_active: bool,
    pub(crate) hide_when_inactive: bool,
    pub(crate) color: [u8; 4],
    pub(crate) error: Option<String>,
    pub(crate) reuse_existing_column: bool,
}

impl Default for CustomAnnotationDialogState {
    fn default() -> Self {
        Self {
            editing_name: None,
            name: String::new(),
            kind_index: 0,
            symbol: "o".to_string(),
            shortcut: String::new(),
            description: String::new(),
            keep_active: true,
            hide_when_inactive: true,
            color: [255, 0, 0, 255],
            error: None,
            reuse_existing_column: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SavedCustomAnnotationsDialogState {
    pub(crate) selected_names: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SnapshotModeState {
    pub(crate) last_profile_message: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SnapshotSaveDialogState {
    pub(crate) selected_positions: Vec<String>,
    pub(crate) quick_save: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CustomAnnotationCommandSnapshot {
    pub(crate) before: CustomAnnotationStore,
    pub(crate) after: CustomAnnotationStore,
    pub(crate) selected_label_before: Option<u32>,
    pub(crate) selected_label_after: Option<u32>,
}

#[derive(Debug, Clone)]
pub(crate) enum EditorHistoryKind {
    MaskEdit,
    CustomAnnotation(CustomAnnotationCommandSnapshot),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnnotationPendingAction {
    ChangePosition(usize),
    ChangeSegmentation(Option<String>),
}

#[derive(Debug, Clone)]
pub(crate) struct AnnotationWorkspaceState {
    pub(crate) tool: AnnotationTool,
    pub(crate) mode: GuiMode,
    pub(crate) lineage_tool: LineageTool,
    pub(crate) brush_radius: usize,
    pub(crate) relabel_target: String,
    pub(crate) merge_target: String,
    pub(crate) mother_target: String,
    pub(crate) save_as_endname: String,
    pub(crate) pending_action: Option<AnnotationPendingAction>,
    pub(crate) dialogs: GuiDialogState,
    pub(crate) object_dock: ObjectDockState,
    pub(crate) highlight: HighlightState,
    pub(crate) export_image: ExportImageDialogState,
    pub(crate) export_video: ExportVideoDialogState,
    pub(crate) version_browser: VersionBrowserState,
    pub(crate) shortcut_editor: ShortcutEditorState,
    pub(crate) manual_tracking: ManualTrackingState,
    pub(crate) cell_cycle_dialog: CellCycleDialogState,
    pub(crate) cell_cycle_table: CellCycleTableState,
    pub(crate) lineage_review: LineageReviewDialogState,
    pub(crate) tracking_params: TrackingParamsDialogState,
    pub(crate) mode_toolbar: ModeToolbarState,
    pub(crate) lineage_candidate_index: usize,
    pub(crate) pending_manual_tracking_edits: Vec<cellacdc_rs::ManualTrackingEdit>,
    pub(crate) custom_annotation_toolbar: CustomAnnotationToolbarState,
    pub(crate) active_custom_annotation: ActiveCustomAnnotationState,
    pub(crate) custom_annotation_dialog: CustomAnnotationDialogState,
    pub(crate) saved_custom_annotations_dialog: SavedCustomAnnotationsDialogState,
    pub(crate) snapshot_mode: SnapshotModeState,
    pub(crate) snapshot_save_dialog: SnapshotSaveDialogState,
    pub(crate) custom_annotations: CustomAnnotationStore,
    pub(crate) custom_annotations_dirty: bool,
    pub(crate) view_plane: ViewPlane,
    pub(crate) editor_undo: Vec<EditorHistoryKind>,
    pub(crate) editor_redo: Vec<EditorHistoryKind>,
}

impl Default for AnnotationWorkspaceState {
    fn default() -> Self {
        Self {
            tool: AnnotationTool::Select,
            mode: GuiMode::Viewer,
            lineage_tool: LineageTool::NoTool,
            brush_radius: 5,
            relabel_target: String::new(),
            merge_target: String::new(),
            mother_target: String::new(),
            save_as_endname: "edited".to_string(),
            pending_action: None,
            dialogs: GuiDialogState::default(),
            object_dock: ObjectDockState::default(),
            highlight: HighlightState::default(),
            export_image: ExportImageDialogState::default(),
            export_video: ExportVideoDialogState::default(),
            version_browser: VersionBrowserState::default(),
            shortcut_editor: ShortcutEditorState::default(),
            manual_tracking: ManualTrackingState::default(),
            cell_cycle_dialog: CellCycleDialogState::default(),
            cell_cycle_table: CellCycleTableState::default(),
            lineage_review: LineageReviewDialogState::default(),
            tracking_params: TrackingParamsDialogState::default(),
            mode_toolbar: ModeToolbarState { show: true },
            lineage_candidate_index: 0,
            pending_manual_tracking_edits: Vec::new(),
            custom_annotation_toolbar: CustomAnnotationToolbarState::default(),
            active_custom_annotation: ActiveCustomAnnotationState::default(),
            custom_annotation_dialog: CustomAnnotationDialogState::default(),
            saved_custom_annotations_dialog: SavedCustomAnnotationsDialogState::default(),
            snapshot_mode: SnapshotModeState::default(),
            snapshot_save_dialog: SnapshotSaveDialogState::default(),
            custom_annotations: CustomAnnotationStore::default(),
            custom_annotations_dirty: false,
            view_plane: ViewPlane::XY,
            editor_undo: Vec::new(),
            editor_redo: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedMaskDocument {
    pub(crate) position_dir: PathBuf,
    pub(crate) segmentation_endname: Option<String>,
    pub(crate) session: MaskEditSession,
    pub(crate) revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataPrepInteractionMode {
    None,
    AddCropRoi,
    AddBackgroundRoi,
    DrawFreeRoi,
}

impl Default for DataPrepInteractionMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DataPrepWorkspaceState {
    pub(crate) active_channel: String,
    pub(crate) segm_info: cellacdc_rs::SegmInfoTable,
    pub(crate) crop_rois: Vec<CropRoiRect>,
    pub(crate) background_rois: BackgroundRoiSet,
    pub(crate) free_roi_points: Vec<(usize, usize)>,
    pub(crate) free_roi: Option<FreehandRoiMask>,
    pub(crate) frame_range: Option<(usize, usize)>,
    pub(crate) z_range: Option<(usize, usize)>,
    pub(crate) pending_crop_preview: Option<CropPreview>,
    pub(crate) crop_dirty: bool,
    pub(crate) interaction_mode: DataPrepInteractionMode,
    pub(crate) projection_mode: ProjectionMode,
    pub(crate) z_index: usize,
    pub(crate) drag_start: Option<(usize, usize)>,
    pub(crate) drag_current: Option<(usize, usize)>,
    pub(crate) last_loaded_position: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InspectionKey {
    pub(crate) position_dir: PathBuf,
    pub(crate) segmentation_endname: Option<String>,
    pub(crate) frame_index: usize,
    pub(crate) projection: FrameProjection,
    pub(crate) selected_label: Option<u32>,
    pub(crate) revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum UtilityTool {
    CountObjects,
    FillHoles,
    Connect3d,
    Stack2dTo3d,
    CombineChannels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum UtilityScopeMode {
    Auto,
    Position,
    Experiment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ResolutionLayoutChoice {
    Auto,
    Yx,
    Tyx,
    Zyx,
    Tzyx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct UtilityState {
    pub(crate) selected_tool: UtilityTool,
    pub(crate) segmentation_path: String,
    pub(crate) output_path: String,
    pub(crate) scope_path: String,
    pub(crate) recipe_path: String,
    pub(crate) append_name: String,
    pub(crate) resolution_size_t: String,
    pub(crate) resolution_size_z: String,
    pub(crate) resolution_layout: ResolutionLayoutChoice,
    pub(crate) stack_target_size_z: usize,
    pub(crate) scope_mode: UtilityScopeMode,
}

impl Default for UtilityState {
    fn default() -> Self {
        Self {
            selected_tool: UtilityTool::CountObjects,
            segmentation_path: String::new(),
            output_path: String::new(),
            scope_path: String::new(),
            recipe_path: String::new(),
            append_name: "combined".to_string(),
            resolution_size_t: String::new(),
            resolution_size_z: String::new(),
            resolution_layout: ResolutionLayoutChoice::Auto,
            stack_target_size_z: 3,
            scope_mode: UtilityScopeMode::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ViewKey {
    pub(crate) position_dir: PathBuf,
    pub(crate) channel: String,
    pub(crate) frame_index: usize,
    pub(crate) view_plane: ViewPlane,
    pub(crate) projection: FrameProjection,
    pub(crate) segmentation_endname: Option<String>,
    pub(crate) overlay_alpha_bits: u32,
    pub(crate) show_overlay: bool,
    pub(crate) show_overlay_labels: bool,
    pub(crate) overlay_single_channel_mode: bool,
    pub(crate) true_transparency: bool,
    pub(crate) add_scale_bar: bool,
    pub(crate) add_timestamp: bool,
    pub(crate) highlighted_label: Option<u32>,
    pub(crate) selected_label: Option<u32>,
}
