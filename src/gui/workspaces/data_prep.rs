use crate::gui::app::CellAcdcGui;
use cellacdc_rs::{BackgroundRoiArchive, BackgroundRoiRect, BackgroundRoiSet};

use super::render_planned_workspace;

impl CellAcdcGui {
    pub(crate) fn draw_data_prep_panel(&mut self, ctx: &eframe::egui::Context) {
        let roi_count = self
            .selected_position()
            .and_then(|position| position.spec.data_prep_background_rois_path.as_ref())
            .map(|_| BackgroundRoiSet {
                items: vec![BackgroundRoiRect {
                    pos: [0.0, 0.0],
                    size: [0.0, 0.0],
                }],
            })
            .map(|set| set.items.len())
            .unwrap_or(0);
        let body = format!(
            "The Rust core now defines data-prep-compatible ROI JSON/NPZ helpers. Current session ROI sidecars detected: {roi_count}. Interactive alignment and crop tools are the next UI layer."
        );
        render_planned_workspace(
            ctx,
            "Data Prep",
            &body,
            &[
                "Wire alignment and crop operations to native Rust prep APIs",
                "Author and persist background ROI JSON sidecars",
                "Export background intensity archives compatible with measurement",
                "Reuse the open session context for per-position prep",
            ],
        );
        let _archive: Option<BackgroundRoiArchive> = None;
    }
}
