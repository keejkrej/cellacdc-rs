use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_lineage_review_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.lineage_review_open {
            return;
        }
        let mut open = self.annotation.dialogs.lineage_review_open;
        egui::Window::new("Lineage Review")
            .open(&mut open)
            .default_size([520.0, 320.0])
            .show(ctx, |ui| {
                let Some(review) = self.annotation.lineage_review.review.as_ref() else {
                    ui.label("No lineage review is available for the current frame.");
                    return;
                };
                ui.label(format!("Frame {}", review.frame_i));
                ui.separator();
                ui.label("Cells with parent");
                for (cell_id, parent_id) in &review.cells_with_parent {
                    ui.label(format!("Cell {cell_id} -> parent {parent_id}"));
                }
                ui.separator();
                ui.label("Orphan cells");
                if review.orphan_cells.is_empty() {
                    ui.label("None");
                } else {
                    for cell_id in &review.orphan_cells {
                        ui.label(format!("Cell {cell_id}"));
                    }
                }
                ui.separator();
                ui.label("Lost cells");
                if review.lost_cells.is_empty() {
                    ui.label("None");
                } else {
                    for cell_id in &review.lost_cells {
                        ui.label(format!("Cell {cell_id}"));
                    }
                }
            });
        self.annotation.dialogs.lineage_review_open = open;
    }
}
