use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;

use super::render_planned_workspace;

impl CellAcdcGui {
    pub(crate) fn draw_help_panel(&mut self, ctx: &eframe::egui::Context) {
        let (back_to_launcher, open_session) = render_planned_workspace(
            ctx,
            AppRoute::Help,
            "This workspace tracks the Rust desktop port status and the current native limitations relative to the Python reference app.",
            &[
                "Keep the module names recognizable to existing Cell-ACDC users",
                "Prefer Rust-native behavior over Python parity hacks",
                "Treat Bio-Formats, Napari, and Python-only models as deferred",
                "Expand CLI-backed utilities into full desktop flows incrementally",
            ],
            self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
            false,
        );
        if back_to_launcher {
            self.set_route(AppRoute::Launcher);
        }
        if open_session {
            self.pick_and_open_session();
        }
    }
}
