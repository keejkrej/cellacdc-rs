use crate::gui::app::CellAcdcGui;

use super::render_planned_workspace;

impl CellAcdcGui {
    pub(crate) fn draw_help_panel(&mut self, ctx: &eframe::egui::Context) {
        render_planned_workspace(
            ctx,
            "Help",
            "This workspace tracks the Rust desktop port status and the current native limitations relative to the Python reference app.",
            &[
                "Keep the module names recognizable to existing Cell-ACDC users",
                "Prefer Rust-native behavior over Python parity hacks",
                "Treat Bio-Formats, Napari, and Python-only models as deferred",
                "Expand CLI-backed utilities into full desktop flows incrementally",
            ],
        );
    }
}
