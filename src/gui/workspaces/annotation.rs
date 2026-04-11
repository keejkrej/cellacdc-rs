use crate::gui::app::CellAcdcGui;
use cellacdc_rs::{MaskEditCommand, SelectionState};

use super::render_planned_workspace;

impl CellAcdcGui {
    pub(crate) fn draw_annotation_panel(&mut self, ctx: &eframe::egui::Context) {
        let selection = SelectionState::default();
        let preview = match (MaskEditCommand::SelectLabel { label: Some(1) }) {
            MaskEditCommand::SelectLabel { label } => label.unwrap_or(0),
            _ => 0,
        };
        let body = format!(
            "Annotation foundations are in place: selection state, undo stack, autosave helpers, and native mask edit commands. Current default selected label preview: {preview}, frame {}.",
            selection.frame_index
        );
        render_planned_workspace(
            ctx,
            "Annotation",
            &body,
            &[
                "Embed a session-backed mask document model",
                "Wire brush, erase, relabel, merge, and delete commands to the viewer",
                "Add save, save-as-version, autosave, and crash recovery controls",
                "Layer tracking and lineage workflows on top of editable masks",
            ],
        );
    }
}
