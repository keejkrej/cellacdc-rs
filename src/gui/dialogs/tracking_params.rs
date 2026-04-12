use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_tracking_params_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.tracking_params_open {
            return;
        }
        let mut open = self.annotation.dialogs.tracking_params_open;
        egui::Window::new("Tracking Parameters")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Rust GUI currently exposes the IoA tracker used by repeat tracking.");
                ui.add(
                    egui::Slider::new(
                        &mut self.annotation.tracking_params.ioa_threshold,
                        0.0..=1.0,
                    )
                    .text("IoA threshold"),
                );
                if ui.button("Apply").clicked() {
                    self.persisted.track = true;
                    self.persisted.track_ioa_threshold =
                        self.annotation.tracking_params.ioa_threshold;
                    self.annotation.dialogs.tracking_params_open = false;
                }
            });
        self.annotation.dialogs.tracking_params_open = open;
    }
}
