use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_cell_cycle_editor_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.cell_cycle_editor_open {
            return;
        }
        let mut open = self.annotation.dialogs.cell_cycle_editor_open;
        egui::Window::new("Edit Cell Cycle Annotations")
            .open(&mut open)
            .default_size([980.0, 420.0])
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut self.annotation.cell_cycle_dialog.apply_to_future,
                        "Apply changes to future frames",
                    );
                    ui.label("End frame");
                    ui.text_edit_singleline(&mut self.annotation.cell_cycle_dialog.propagate_end_frame);
                    if ui.button("Reload").clicked() {
                        if let Err(err) = self.load_cell_cycle_dialog_state() {
                            self.annotation.cell_cycle_dialog.error = Some(err.to_string());
                        }
                    }
                });
                if let Some(error) = self.annotation.cell_cycle_dialog.error.clone() {
                    ui.colored_label(egui::Color32::from_rgb(200, 60, 60), error);
                }
                let frame_i = self.selected_frame_idx as i64;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("cell_cycle_editor_grid")
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
                                .iter_mut()
                                .filter(|record| record.frame_i == frame_i)
                            {
                                ui.monospace(record.cell_id.to_string());
                                ui.text_edit_singleline(&mut record.cell_cycle_stage);
                                ui.add(egui::DragValue::new(&mut record.generation_num));
                                ui.add(egui::DragValue::new(&mut record.relative_id));
                                ui.text_edit_singleline(&mut record.relationship);
                                ui.add(egui::DragValue::new(&mut record.emerg_frame_i));
                                ui.add(egui::DragValue::new(&mut record.division_frame_i));
                                ui.checkbox(&mut record.is_history_known, "");
                                ui.end_row();
                            }
                        });
                });
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Apply").clicked() {
                        match self.save_cell_cycle_dialog_state() {
                            Ok(()) => {
                                self.annotation.dialogs.cell_cycle_editor_open = false;
                                self.annotation.cell_cycle_dialog.error = None;
                            }
                            Err(err) => {
                                self.annotation.cell_cycle_dialog.error = Some(err.to_string());
                            }
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.annotation.dialogs.cell_cycle_editor_open = false;
                    }
                });
            });
        self.annotation.dialogs.cell_cycle_editor_open = open;
    }
}
