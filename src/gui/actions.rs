use super::app::CellAcdcGui;
use super::state::{
    AnnotationTool, GuiActionId, GuiActionState, GuiMode, LineageTool, ShortcutBinding,
};

pub(crate) const FILE_ACTIONS: &[GuiActionId] = &[
    GuiActionId::OpenSession,
    GuiActionId::LoadDifferentPosition,
    GuiActionId::RevealCurrentPosition,
    GuiActionId::Save,
    GuiActionId::QuickSave,
    GuiActionId::SaveAsVersion,
    GuiActionId::LoadOlderVersions,
    GuiActionId::ExportImage,
    GuiActionId::ExportVideo,
];
pub(crate) const EDIT_ACTIONS: &[GuiActionId] = &[
    GuiActionId::Undo,
    GuiActionId::Redo,
    GuiActionId::ToolSelect,
    GuiActionId::ToolBrush,
    GuiActionId::ToolEraser,
    GuiActionId::ToolRelabel,
    GuiActionId::ToolMerge,
    GuiActionId::ToolDelete,
    GuiActionId::CustomizeKeyboardShortcuts,
    GuiActionId::OverlayLabelsAppearance,
];
pub(crate) const VIEW_ACTIONS: &[GuiActionId] = &[
    GuiActionId::ToggleObjectsDock,
    GuiActionId::ToggleLogDock,
    GuiActionId::FindId,
    GuiActionId::HighlightSelectedId,
    GuiActionId::ToggleOverlayLabels,
    GuiActionId::ToggleSegmentationOverlay,
];
pub(crate) const IMAGE_ACTIONS: &[GuiActionId] = &[
    GuiActionId::ToggleSingleChannelOverlay,
    GuiActionId::ToggleTrueTransparency,
    GuiActionId::ToggleScaleBar,
    GuiActionId::ToggleTimestamp,
];
pub(crate) const SEGMENT_ACTIONS: &[GuiActionId] = &[
    GuiActionId::RunSegmentationCurrentPosition,
    GuiActionId::OpenSegmentationWorkspace,
];
pub(crate) const MEASUREMENT_ACTIONS: &[GuiActionId] = &[GuiActionId::MeasureCurrentPosition];
pub(crate) const TRACKING_ACTIONS: &[GuiActionId] = &[
    GuiActionId::RepeatTracking,
    GuiActionId::TrackCurrentFrame,
    GuiActionId::ManualTracking,
    GuiActionId::EditRealTimeTrackerParameters,
];
pub(crate) const CELL_CYCLE_ACTIONS: &[GuiActionId] = &[
    GuiActionId::AssignMotherToBud,
    GuiActionId::EditCellCycleAnnotations,
    GuiActionId::ViewCellCycleAnnotations,
];
pub(crate) const CUSTOM_ANNOTATION_ACTIONS: &[GuiActionId] = &[
    GuiActionId::LoadSavedCustomAnnotations,
    GuiActionId::AddCustomAnnotation,
    GuiActionId::ShowAllCustomAnnotations,
];
pub(crate) const LINEAGE_ACTIONS: &[GuiActionId] = &[
    GuiActionId::FindNextPotentialMother,
    GuiActionId::UnknownLineage,
    GuiActionId::NoLineageTool,
    GuiActionId::PropagateLineage,
    GuiActionId::ViewLineageChanges,
];
pub(crate) const SETTINGS_ACTIONS: &[GuiActionId] = &[
    GuiActionId::AutosaveInterval,
    GuiActionId::CustomizeKeyboardShortcuts,
];
pub(crate) const HELP_ACTIONS: &[GuiActionId] = &[
    GuiActionId::OpenLogs,
    GuiActionId::AboutRustPort,
    GuiActionId::CurrentLimitations,
];

pub(crate) fn action_label(action: GuiActionId) -> &'static str {
    match action {
        GuiActionId::OpenSession => "Open Session",
        GuiActionId::LoadDifferentPosition => "Load Different Position",
        GuiActionId::RevealCurrentPosition => "Reveal Current Position",
        GuiActionId::Save => "Save",
        GuiActionId::QuickSave => "Quick Save",
        GuiActionId::SaveAsVersion => "Save As Version",
        GuiActionId::LoadOlderVersions => "Load Older Versions",
        GuiActionId::ExportImage => "Export Image",
        GuiActionId::ExportVideo => "Export Video",
        GuiActionId::Undo => "Undo",
        GuiActionId::Redo => "Redo",
        GuiActionId::CustomizeKeyboardShortcuts => "Customize Keyboard Shortcuts",
        GuiActionId::OverlayLabelsAppearance => "Overlay Labels Appearance",
        GuiActionId::ToggleObjectsDock => "Toggle Objects Dock",
        GuiActionId::ToggleLogDock => "Toggle Log Dock",
        GuiActionId::FindId => "Find ID",
        GuiActionId::HighlightSelectedId => "Highlight Selected ID",
        GuiActionId::ToggleOverlayLabels => "Toggle Overlay Labels",
        GuiActionId::ToggleSegmentationOverlay => "Toggle Segmentation Overlay",
        GuiActionId::ToggleSingleChannelOverlay => "Single Channel Overlay",
        GuiActionId::ToggleTrueTransparency => "True Transparency",
        GuiActionId::ToggleScaleBar => "Add Scale Bar",
        GuiActionId::ToggleTimestamp => "Add Timestamp",
        GuiActionId::RunSegmentationCurrentPosition => "Run Segmentation On Current Position",
        GuiActionId::OpenSegmentationWorkspace => "Open Segmentation Workspace",
        GuiActionId::MeasureCurrentPosition => "Measure Current Position",
        GuiActionId::AutosaveInterval => "Autosave Interval",
        GuiActionId::OpenLogs => "Open Logs",
        GuiActionId::AboutRustPort => "About Rust Port",
        GuiActionId::CurrentLimitations => "Current Limitations",
        GuiActionId::ToolSelect => "Select",
        GuiActionId::ToolBrush => "Brush",
        GuiActionId::ToolEraser => "Eraser",
        GuiActionId::ToolRelabel => "Relabel",
        GuiActionId::ToolMerge => "Merge",
        GuiActionId::ToolDelete => "Delete",
        GuiActionId::ModeViewer => "Viewer",
        GuiActionId::ModeSegmentationAndTracking => "Segmentation and Tracking",
        GuiActionId::ModeCellCycleAnalysis => "Cell cycle analysis",
        GuiActionId::ModeNormalDivisionLineageTree => "Normal division: Lineage tree",
        GuiActionId::ModeCustomAnnotations => "Custom annotations",
        GuiActionId::RepeatTracking => "Repeat Tracking",
        GuiActionId::TrackCurrentFrame => "Track Current Frame Forward",
        GuiActionId::ManualTracking => "Manual Tracking",
        GuiActionId::EditRealTimeTrackerParameters => "Edit Real-time Tracker Parameters",
        GuiActionId::AssignMotherToBud => "Assign Mother To Bud",
        GuiActionId::EditCellCycleAnnotations => "Edit Cell Cycle Annotations",
        GuiActionId::ViewCellCycleAnnotations => "View Cell Cycle Annotations",
        GuiActionId::FindNextPotentialMother => "Find Next Potential Mother",
        GuiActionId::UnknownLineage => "Unknown Lineage",
        GuiActionId::NoLineageTool => "No Lineage Tool",
        GuiActionId::PropagateLineage => "Propagate Lineage",
        GuiActionId::ViewLineageChanges => "View Lineage Changes",
        GuiActionId::LoadSavedCustomAnnotations => "Load previously used custom annotations",
        GuiActionId::AddCustomAnnotation => "Add custom annotation",
        GuiActionId::ShowAllCustomAnnotations => "Show all custom annotations",
    }
}

pub(crate) fn default_shortcut_binding(action: GuiActionId) -> Option<ShortcutBinding> {
    let binding = match action {
        GuiActionId::Save | GuiActionId::QuickSave => ShortcutBinding {
            key: "S".to_string(),
            command: true,
            shift: false,
            alt: false,
        },
        GuiActionId::SaveAsVersion => ShortcutBinding {
            key: "S".to_string(),
            command: true,
            shift: true,
            alt: false,
        },
        GuiActionId::Undo => ShortcutBinding {
            key: "Z".to_string(),
            command: true,
            shift: false,
            alt: false,
        },
        GuiActionId::Redo => ShortcutBinding {
            key: "Z".to_string(),
            command: true,
            shift: true,
            alt: false,
        },
        GuiActionId::FindId => ShortcutBinding {
            key: "F".to_string(),
            command: true,
            shift: false,
            alt: false,
        },
        GuiActionId::ToolBrush => ShortcutBinding {
            key: "B".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::ToolEraser => ShortcutBinding {
            key: "E".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::ToolSelect => ShortcutBinding {
            key: "V".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::HighlightSelectedId => ShortcutBinding {
            key: "H".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::ManualTracking => ShortcutBinding {
            key: "T".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::RepeatTracking => ShortcutBinding {
            key: "T".to_string(),
            command: false,
            shift: true,
            alt: false,
        },
        GuiActionId::AssignMotherToBud => ShortcutBinding {
            key: "A".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::ModeCustomAnnotations => ShortcutBinding {
            key: "C".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::UnknownLineage => ShortcutBinding {
            key: "U".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::NoLineageTool => ShortcutBinding {
            key: "N".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        GuiActionId::PropagateLineage => ShortcutBinding {
            key: "P".to_string(),
            command: false,
            shift: false,
            alt: false,
        },
        _ => return None,
    };
    Some(binding)
}

impl CellAcdcGui {
    pub(crate) fn gui_action_state(&self, action: GuiActionId) -> GuiActionState {
        let has_session = self.experiment.is_some();
        let has_document = self.current_annotation_document().is_some();
        let has_selection = self.current_annotation_label().is_some();
        let can_edit = self.annotation_edits_allowed();
        let snapshot_mode = self.annotation.mode == GuiMode::Snapshot;
        let tracking_mode = self.annotation.mode == GuiMode::SegmentationAndTracking;
        let cell_cycle_mode = self.annotation.mode == GuiMode::CellCycleAnalysis;
        let lineage_mode = self.annotation.mode == GuiMode::NormalDivisionLineageTree;
        let custom_annotation_mode = self.annotation.mode == GuiMode::CustomAnnotations;
        let session_is_snapshot = self
            .current_snapshot_profile()
            .map(|profile| profile.is_snapshot)
            .unwrap_or(false);
        let mut state = GuiActionState::default();
        state.enabled = match action {
            GuiActionId::OpenSession
            | GuiActionId::LoadDifferentPosition
            | GuiActionId::OpenLogs
            | GuiActionId::AboutRustPort
            | GuiActionId::CurrentLimitations
            | GuiActionId::ModeViewer
            | GuiActionId::ModeSegmentationAndTracking
            | GuiActionId::ModeCellCycleAnalysis
            | GuiActionId::ModeNormalDivisionLineageTree
            | GuiActionId::ModeCustomAnnotations => !session_is_snapshot,
            GuiActionId::RevealCurrentPosition
            | GuiActionId::Save
            | GuiActionId::QuickSave
            | GuiActionId::SaveAsVersion
            | GuiActionId::LoadOlderVersions
            | GuiActionId::ExportImage
            | GuiActionId::ExportVideo
            | GuiActionId::FindId
            | GuiActionId::HighlightSelectedId
            | GuiActionId::ToggleObjectsDock
            | GuiActionId::ToggleLogDock
            | GuiActionId::ToggleOverlayLabels
            | GuiActionId::ToggleSegmentationOverlay
            | GuiActionId::ToggleSingleChannelOverlay
            | GuiActionId::ToggleTrueTransparency
            | GuiActionId::ToggleScaleBar
            | GuiActionId::ToggleTimestamp
            | GuiActionId::AutosaveInterval
            | GuiActionId::CustomizeKeyboardShortcuts
            | GuiActionId::OverlayLabelsAppearance
            | GuiActionId::OpenSegmentationWorkspace
            | GuiActionId::ShowAllCustomAnnotations => has_session,
            GuiActionId::Undo | GuiActionId::Redo => has_document,
            GuiActionId::RunSegmentationCurrentPosition | GuiActionId::MeasureCurrentPosition => {
                has_session
            }
            GuiActionId::ToolSelect => has_document,
            GuiActionId::ToolBrush
            | GuiActionId::ToolEraser
            | GuiActionId::ToolRelabel
            | GuiActionId::ToolMerge
            | GuiActionId::ToolDelete => {
                has_document && (can_edit || has_selection) && !custom_annotation_mode
            }
            GuiActionId::RepeatTracking | GuiActionId::TrackCurrentFrame => {
                has_session && tracking_mode && !snapshot_mode && !self.annotation_document_dirty()
            }
            GuiActionId::ManualTracking => has_document && tracking_mode && !snapshot_mode,
            GuiActionId::EditRealTimeTrackerParameters => {
                has_session && tracking_mode && !snapshot_mode
            }
            GuiActionId::AssignMotherToBud => {
                has_session && (cell_cycle_mode || snapshot_mode) && has_selection
            }
            GuiActionId::EditCellCycleAnnotations | GuiActionId::ViewCellCycleAnnotations => {
                has_session && (cell_cycle_mode || snapshot_mode)
            }
            GuiActionId::FindNextPotentialMother
            | GuiActionId::UnknownLineage
            | GuiActionId::NoLineageTool
            | GuiActionId::PropagateLineage
            | GuiActionId::ViewLineageChanges => has_session && lineage_mode && has_selection,
            GuiActionId::LoadSavedCustomAnnotations | GuiActionId::AddCustomAnnotation => {
                has_session
            }
        };
        state.checked = match action {
            GuiActionId::ToggleObjectsDock => self.persisted.display.show_objects_dock,
            GuiActionId::ToggleLogDock => self.persisted.display.show_log_dock,
            GuiActionId::ToggleOverlayLabels => self.persisted.display.show_overlay_labels,
            GuiActionId::ToggleSegmentationOverlay => self.persisted.show_segmentation_overlay,
            GuiActionId::ToggleSingleChannelOverlay => {
                self.persisted.display.overlay_single_channel_mode
            }
            GuiActionId::ToggleTrueTransparency => self.persisted.display.true_transparency,
            GuiActionId::ToggleScaleBar => self.persisted.display.add_scale_bar,
            GuiActionId::ToggleTimestamp => self.persisted.display.add_timestamp,
            GuiActionId::ToolSelect => self.annotation.tool == AnnotationTool::Select,
            GuiActionId::ToolBrush => self.annotation.tool == AnnotationTool::Brush,
            GuiActionId::ToolEraser => self.annotation.tool == AnnotationTool::Eraser,
            GuiActionId::ToolRelabel => self.annotation.tool == AnnotationTool::Relabel,
            GuiActionId::ToolMerge => self.annotation.tool == AnnotationTool::Merge,
            GuiActionId::ToolDelete => self.annotation.tool == AnnotationTool::Delete,
            GuiActionId::ModeViewer => self.annotation.mode == GuiMode::Viewer,
            GuiActionId::ModeSegmentationAndTracking => {
                self.annotation.mode == GuiMode::SegmentationAndTracking
            }
            GuiActionId::ModeCellCycleAnalysis => {
                self.annotation.mode == GuiMode::CellCycleAnalysis
            }
            GuiActionId::ModeNormalDivisionLineageTree => {
                self.annotation.mode == GuiMode::NormalDivisionLineageTree
            }
            GuiActionId::ModeCustomAnnotations => {
                self.annotation.mode == GuiMode::CustomAnnotations
            }
            GuiActionId::ManualTracking => self.annotation.manual_tracking.active,
            GuiActionId::FindNextPotentialMother => {
                self.annotation.lineage_tool == LineageTool::FindNextMother
            }
            GuiActionId::UnknownLineage => {
                self.annotation.lineage_tool == LineageTool::UnknownLineage
            }
            GuiActionId::NoLineageTool => self.annotation.lineage_tool == LineageTool::NoTool,
            GuiActionId::ShowAllCustomAnnotations => {
                self.annotation.custom_annotation_toolbar.show_all
            }
            _ => false,
        };
        if session_is_snapshot {
            state.enabled = match action {
                GuiActionId::ModeViewer
                | GuiActionId::ModeSegmentationAndTracking
                | GuiActionId::ModeCellCycleAnalysis
                | GuiActionId::ModeNormalDivisionLineageTree
                | GuiActionId::ModeCustomAnnotations => false,
                _ => state.enabled,
            };
        }
        if matches!(
            action,
            GuiActionId::LoadSavedCustomAnnotations | GuiActionId::ShowAllCustomAnnotations
        ) {
            state.enabled = has_session;
        }
        if action == GuiActionId::AddCustomAnnotation {
            state.enabled = has_session && has_document;
        }
        if action == GuiActionId::ShowAllCustomAnnotations {
            state.visible =
                has_session && !self.annotation.custom_annotations.definitions.is_empty();
        }
        if matches!(action, GuiActionId::LoadSavedCustomAnnotations) {
            state.visible = has_session;
        }
        if custom_annotation_mode {
            if matches!(
                action,
                GuiActionId::ToolBrush
                    | GuiActionId::ToolEraser
                    | GuiActionId::ToolRelabel
                    | GuiActionId::ToolMerge
                    | GuiActionId::ToolDelete
            ) {
                state.enabled = false;
            }
        }
        if snapshot_mode
            && matches!(
                action,
                GuiActionId::RepeatTracking
                    | GuiActionId::TrackCurrentFrame
                    | GuiActionId::ManualTracking
                    | GuiActionId::EditRealTimeTrackerParameters
            )
        {
            state.enabled = false;
        }
        state
    }

    pub(crate) fn dispatch_gui_action(&mut self, action: GuiActionId) {
        match action {
            GuiActionId::OpenSession | GuiActionId::LoadDifferentPosition => {
                self.pick_and_open_session();
            }
            GuiActionId::RevealCurrentPosition => {
                if let Err(err) = self.reveal_current_position() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::Save => {
                if let Err(err) = self.request_save_current_annotation_overwrite(false) {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::QuickSave => {
                if let Err(err) = self.request_save_current_annotation_overwrite(true) {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::SaveAsVersion => {
                if let Err(err) = self.save_current_annotation_as_version() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::LoadOlderVersions => {
                self.annotation.dialogs.version_browser_open = true;
            }
            GuiActionId::ExportImage => {
                self.annotation.dialogs.export_image_open = true;
                self.prepare_export_defaults();
            }
            GuiActionId::ExportVideo => {
                self.annotation.dialogs.export_video_open = true;
                self.prepare_export_defaults();
            }
            GuiActionId::Undo => self.annotation_undo(),
            GuiActionId::Redo => self.annotation_redo(),
            GuiActionId::CustomizeKeyboardShortcuts => {
                self.annotation.dialogs.shortcut_editor_open = true;
            }
            GuiActionId::OverlayLabelsAppearance => {
                self.annotation.dialogs.overlay_labels_open = true;
            }
            GuiActionId::ToggleObjectsDock => {
                self.persisted.display.show_objects_dock =
                    !self.persisted.display.show_objects_dock;
            }
            GuiActionId::ToggleLogDock => {
                self.persisted.display.show_log_dock = !self.persisted.display.show_log_dock;
            }
            GuiActionId::FindId => {
                self.annotation.dialogs.find_id_open = true;
                self.annotation.highlight.highlighted_input = self
                    .current_annotation_label()
                    .map(|label| label.to_string())
                    .unwrap_or_default();
            }
            GuiActionId::HighlightSelectedId => {
                self.annotation.highlight.searched_label = self.current_annotation_label();
                self.invalidate_texture();
            }
            GuiActionId::ToggleOverlayLabels => {
                self.persisted.display.show_overlay_labels =
                    !self.persisted.display.show_overlay_labels;
                self.invalidate_texture();
            }
            GuiActionId::ToggleSegmentationOverlay => {
                self.persisted.show_segmentation_overlay =
                    !self.persisted.show_segmentation_overlay;
                self.invalidate_texture();
            }
            GuiActionId::ToggleSingleChannelOverlay => {
                self.persisted.display.overlay_single_channel_mode =
                    !self.persisted.display.overlay_single_channel_mode;
                self.invalidate_texture();
            }
            GuiActionId::ToggleTrueTransparency => {
                self.persisted.display.true_transparency =
                    !self.persisted.display.true_transparency;
                self.invalidate_texture();
            }
            GuiActionId::ToggleScaleBar => {
                self.persisted.display.add_scale_bar = !self.persisted.display.add_scale_bar;
                self.invalidate_texture();
            }
            GuiActionId::ToggleTimestamp => {
                self.persisted.display.add_timestamp = !self.persisted.display.add_timestamp;
                self.invalidate_texture();
            }
            GuiActionId::RunSegmentationCurrentPosition => self.start_run_position_job(),
            GuiActionId::OpenSegmentationWorkspace => {
                self.set_route(super::state::AppRoute::Segmentation);
            }
            GuiActionId::MeasureCurrentPosition => self.start_measure_position_job(),
            GuiActionId::AutosaveInterval => {
                self.annotation.dialogs.autosave_interval_open = true;
            }
            GuiActionId::OpenLogs => {
                self.persisted.display.show_log_dock = true;
            }
            GuiActionId::AboutRustPort => {
                self.append_log(
                    "Cell-ACDC Rust GUI route: desktop editor parity stage.".to_string(),
                );
            }
            GuiActionId::CurrentLimitations => {
                self.append_log(
                    "Current GUI limitations: Python-only trackers and wider non-editor module parity are still deferred."
                        .to_string(),
                );
            }
            GuiActionId::ToolSelect => self.annotation.tool = AnnotationTool::Select,
            GuiActionId::ToolBrush => self.annotation.tool = AnnotationTool::Brush,
            GuiActionId::ToolEraser => self.annotation.tool = AnnotationTool::Eraser,
            GuiActionId::ToolRelabel => self.annotation.tool = AnnotationTool::Relabel,
            GuiActionId::ToolMerge => self.annotation.tool = AnnotationTool::Merge,
            GuiActionId::ToolDelete => self.annotation.tool = AnnotationTool::Delete,
            GuiActionId::ModeViewer => self.annotation.mode = GuiMode::Viewer,
            GuiActionId::ModeSegmentationAndTracking => {
                self.annotation.mode = GuiMode::SegmentationAndTracking;
            }
            GuiActionId::ModeCellCycleAnalysis => {
                self.annotation.mode = GuiMode::CellCycleAnalysis;
            }
            GuiActionId::ModeNormalDivisionLineageTree => {
                self.annotation.mode = GuiMode::NormalDivisionLineageTree;
            }
            GuiActionId::ModeCustomAnnotations => {
                self.annotation.mode = GuiMode::CustomAnnotations;
            }
            GuiActionId::RepeatTracking => self.start_repeat_tracking_job(None),
            GuiActionId::TrackCurrentFrame => {
                self.start_repeat_tracking_job(Some(self.selected_frame_idx));
            }
            GuiActionId::ManualTracking => {
                self.annotation.manual_tracking.active = !self.annotation.manual_tracking.active;
            }
            GuiActionId::EditRealTimeTrackerParameters => {
                self.annotation.dialogs.tracking_params_open = true;
            }
            GuiActionId::AssignMotherToBud => {
                if let Err(err) = self.assign_selected_bud_to_mother() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::EditCellCycleAnnotations => {
                if let Err(err) = self.load_cell_cycle_dialog_state() {
                    self.last_error = Some(err.to_string());
                }
                self.annotation.dialogs.cell_cycle_editor_open = true;
            }
            GuiActionId::ViewCellCycleAnnotations => {
                if let Err(err) = self.load_cell_cycle_dialog_state() {
                    self.last_error = Some(err.to_string());
                }
                self.annotation.dialogs.cell_cycle_viewer_open = true;
            }
            GuiActionId::FindNextPotentialMother => {
                self.annotation.lineage_tool = LineageTool::FindNextMother;
                if let Err(err) = self.select_next_lineage_candidate() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::UnknownLineage => {
                self.annotation.lineage_tool = LineageTool::UnknownLineage;
                if let Err(err) = self.mark_selected_lineage_unknown() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::NoLineageTool => self.annotation.lineage_tool = LineageTool::NoTool,
            GuiActionId::PropagateLineage => {
                if let Err(err) = self.propagate_selected_lineage() {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::ViewLineageChanges => {
                if let Err(err) = self.refresh_lineage_review() {
                    self.last_error = Some(err.to_string());
                }
                self.annotation.dialogs.lineage_review_open = true;
            }
            GuiActionId::LoadSavedCustomAnnotations => {
                self.annotation.dialogs.load_saved_custom_annotations_open = true;
            }
            GuiActionId::AddCustomAnnotation => {
                if let Err(err) = self.open_custom_annotation_editor(None) {
                    self.last_error = Some(err.to_string());
                }
            }
            GuiActionId::ShowAllCustomAnnotations => {
                self.annotation.custom_annotation_toolbar.show_all =
                    !self.annotation.custom_annotation_toolbar.show_all;
                self.invalidate_texture();
            }
        }
    }
}
