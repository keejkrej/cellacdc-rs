use crate::gui::app::CellAcdcGui;
use eframe::egui;
use rfd::FileDialog;

impl CellAcdcGui {
    pub(crate) fn draw_export_image_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.export_image_open {
            return;
        }
        let mut open = self.annotation.dialogs.export_image_open;
        egui::Window::new("Export Image")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Output");
                    ui.text_edit_singleline(&mut self.annotation.export_image.path);
                    if ui.button("Browse").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Image", &["png", "tiff"])
                            .save_file()
                        {
                            self.annotation.export_image.path = path.display().to_string();
                        }
                    }
                });
                ui.checkbox(
                    &mut self.annotation.export_image.include_overlay,
                    "Include overlay",
                );
                ui.checkbox(
                    &mut self.annotation.export_image.include_labels,
                    "Include labels",
                );
                ui.checkbox(
                    &mut self.annotation.export_image.include_scale_bar,
                    "Include scale bar",
                );
                ui.checkbox(
                    &mut self.annotation.export_image.include_timestamp,
                    "Include timestamp",
                );
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() {
                        if let Err(err) = self.export_current_image() {
                            self.last_error = Some(err.to_string());
                        } else {
                            self.annotation.dialogs.export_image_open = false;
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.annotation.dialogs.export_image_open = false;
                    }
                });
            });
        self.annotation.dialogs.export_image_open = open;
    }
}
