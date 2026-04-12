use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_cell_cycle_viewer_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.cell_cycle_viewer_open {
            return;
        }
        let mut open = self.annotation.dialogs.cell_cycle_viewer_open;
        egui::Window::new("View Cell Cycle Annotations")
            .open(&mut open)
            .default_size([860.0, 360.0])
            .show(ctx, |ui| {
                let frame_i = self.selected_frame_idx as i64;
                ui.label(format!("Current frame: {frame_i}"));
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("cell_cycle_viewer_grid")
                        .striped(true)
                        .show(ui, |ui| {
                            for heading in [
                                "Cell_ID",
                                "Stage",
                                "Generation",
                                "Relative ID",
                                "Relationship",
                                "Emerg",
                                "Division",
                                "History",
                            ] {
                                ui.strong(heading);
                            }
                            ui.end_row();
                            for record in self
                                .annotation
                                .cell_cycle_table
                                .records
                                .iter()
                                .filter(|record| record.frame_i == frame_i)
                            {
                                ui.monospace(record.cell_id.to_string());
                                ui.label(&record.cell_cycle_stage);
                                ui.label(record.generation_num.to_string());
                                ui.label(record.relative_id.to_string());
                                ui.label(&record.relationship);
                                ui.label(record.emerg_frame_i.to_string());
                                ui.label(record.division_frame_i.to_string());
                                ui.label(record.is_history_known.to_string());
                                ui.end_row();
                            }
                        });
                });
            });
        self.annotation.dialogs.cell_cycle_viewer_open = open;
    }
}
