use crate::gui::app::CellAcdcGui;
use eframe::egui;
use rfd::FileDialog;

impl CellAcdcGui {
    pub(crate) fn draw_export_video_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.export_video_open {
            return;
        }
        let mut open = self.annotation.dialogs.export_video_open;
        egui::Window::new("Export Video")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Output path");
                    ui.text_edit_singleline(&mut self.annotation.export_video.output_path);
                    if ui.button("Browse").clicked() {
                        if let Some(path) = FileDialog::new().save_file() {
                            self.annotation.export_video.output_path = path.display().to_string();
                        }
                    }
                });
                ui.add(
                    egui::DragValue::new(&mut self.annotation.export_video.start_frame)
                        .prefix("Start "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.annotation.export_video.end_frame)
                        .prefix("End "),
                );
                ui.checkbox(&mut self.annotation.export_video.include_overlay, "Include overlay");
                ui.checkbox(&mut self.annotation.export_video.include_labels, "Include labels");
                ui.checkbox(
                    &mut self.annotation.export_video.include_scale_bar,
                    "Include scale bar",
                );
                ui.checkbox(
                    &mut self.annotation.export_video.include_timestamp,
                    "Include timestamp",
                );
                ui.label("Use a `.mp4` output path to encode with ffmpeg when it is available. Otherwise a PNG sequence is exported.");
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() {
                        if let Err(err) = self.export_current_video_or_sequence() {
                            self.last_error = Some(err.to_string());
                        } else {
                            self.annotation.dialogs.export_video_open = false;
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.annotation.dialogs.export_video_open = false;
                    }
                });
            });
        self.annotation.dialogs.export_video_open = open;
    }
}
