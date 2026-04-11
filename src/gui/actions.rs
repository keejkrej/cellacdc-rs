use super::app::CellAcdcGui;
use super::state::{
    AnnotationTool, GuiActionId, GuiActionState, ShortcutBinding,
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
        let mut state = GuiActionState::default();
        state.enabled = match action {
            GuiActionId::OpenSession
            | GuiActionId::LoadDifferentPosition
            | GuiActionId::OpenLogs
            | GuiActionId::AboutRustPort
            | GuiActionId::CurrentLimitations => true,
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
            | GuiActionId::OpenSegmentationWorkspace => has_session,
            GuiActionId::Undo | GuiActionId::Redo => has_document,
            GuiActionId::RunSegmentationCurrentPosition | GuiActionId::MeasureCurrentPosition => {
                has_session
            }
            GuiActionId::ToolSelect => has_document,
            GuiActionId::ToolBrush
            | GuiActionId::ToolEraser
            | GuiActionId::ToolRelabel
            | GuiActionId::ToolMerge
            | GuiActionId::ToolDelete => has_document && (can_edit || has_selection),
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
            _ => false,
        };
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
            GuiActionId::Save | GuiActionId::QuickSave => {
                if let Err(err) = self.save_current_annotation_overwrite() {
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
                self.persisted.display.show_objects_dock = !self.persisted.display.show_objects_dock;
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
                self.persisted.show_segmentation_overlay = !self.persisted.show_segmentation_overlay;
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
                self.append_log("Cell-ACDC Rust GUI route: desktop editor parity stage.".to_string());
            }
            GuiActionId::CurrentLimitations => {
                self.append_log(
                    "Current GUI limitations: no lineage editor, no cell-cycle editor, no OS-native menu bar."
                        .to_string(),
                );
            }
            GuiActionId::ToolSelect => self.annotation.tool = AnnotationTool::Select,
            GuiActionId::ToolBrush => self.annotation.tool = AnnotationTool::Brush,
            GuiActionId::ToolEraser => self.annotation.tool = AnnotationTool::Eraser,
            GuiActionId::ToolRelabel => self.annotation.tool = AnnotationTool::Relabel,
            GuiActionId::ToolMerge => self.annotation.tool = AnnotationTool::Merge,
            GuiActionId::ToolDelete => self.annotation.tool = AnnotationTool::Delete,
        }
    }
}
