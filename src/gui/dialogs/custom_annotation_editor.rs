use crate::gui::app::CellAcdcGui;
use eframe::egui::{self, Color32};

impl CellAcdcGui {
    pub(crate) fn draw_custom_annotation_editor_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.custom_annotation_editor_open {
            return;
        }
        let mut open = self.annotation.dialogs.custom_annotation_editor_open;
        let mut apply_clicked = false;
        let mut cancel_clicked = false;
        egui::Window::new("Custom Annotation")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Define a custom annotation tool for Cell-ACDC objects.");
                if let Some(error) = self.annotation.custom_annotation_dialog.error.clone() {
                    ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                }

                ui.horizontal(|ui| {
                    ui.label("Type");
                    let selected = match self.annotation.custom_annotation_dialog.kind_index {
                        1 => "Multiple time-points",
                        2 => "Multiple values class",
                        _ => "Single time-point",
                    };
                    egui::ComboBox::from_id_salt("custom_annotation_kind")
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.annotation.custom_annotation_dialog.kind_index,
                                0,
                                "Single time-point",
                            );
                            ui.selectable_value(
                                &mut self.annotation.custom_annotation_dialog.kind_index,
                                1,
                                "Multiple time-points",
                            );
                            ui.selectable_value(
                                &mut self.annotation.custom_annotation_dialog.kind_index,
                                2,
                                "Multiple values class",
                            );
                        });
                });
                if self.annotation.custom_annotation_dialog.kind_index != 0 {
                    ui.colored_label(
                        Color32::from_rgb(220, 150, 60),
                        "Only Single time-point custom annotations are implemented in the Rust port.",
                    );
                }

                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.annotation.custom_annotation_dialog.name);
                });
                ui.horizontal(|ui| {
                    ui.label("Symbol");
                    ui.text_edit_singleline(&mut self.annotation.custom_annotation_dialog.symbol);
                });
                ui.horizontal(|ui| {
                    ui.label("Shortcut");
                    ui.text_edit_singleline(&mut self.annotation.custom_annotation_dialog.shortcut);
                });
                ui.horizontal(|ui| {
                    ui.label("Description");
                    ui.text_edit_singleline(&mut self.annotation.custom_annotation_dialog.description);
                });
                ui.checkbox(
                    &mut self.annotation.custom_annotation_dialog.keep_active,
                    "Keep tool active after toggle",
                );
                ui.checkbox(
                    &mut self.annotation.custom_annotation_dialog.hide_when_inactive,
                    "Hide annotation when inactive",
                );
                ui.checkbox(
                    &mut self.annotation.custom_annotation_dialog.reuse_existing_column,
                    "Reuse an existing CSV column with the same name",
                );
                ui.horizontal(|ui| {
                    ui.label("Symbol color");
                    let color = &mut self.annotation.custom_annotation_dialog.color;
                    ui.add(egui::DragValue::new(&mut color[0]).range(0..=255).prefix("R "));
                    ui.add(egui::DragValue::new(&mut color[1]).range(0..=255).prefix("G "));
                    ui.add(egui::DragValue::new(&mut color[2]).range(0..=255).prefix("B "));
                    ui.add(egui::DragValue::new(&mut color[3]).range(0..=255).prefix("A "));
                });

                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        apply_clicked = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                });
            });

        if apply_clicked {
            match self.apply_custom_annotation_dialog() {
                Ok(()) => {
                    self.annotation.custom_annotation_dialog.error = None;
                    open = false;
                }
                Err(err) => {
                    self.annotation.custom_annotation_dialog.error = Some(err.to_string());
                    open = true;
                }
            }
        }
        if cancel_clicked {
            self.annotation.custom_annotation_dialog.error = None;
            open = false;
        }
        self.annotation.dialogs.custom_annotation_editor_open = open;
    }
}
