use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_overlay_labels_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.overlay_labels_open {
            return;
        }
        let mut open = self.annotation.dialogs.overlay_labels_open;
        egui::Window::new("Overlay labels appearance")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.checkbox(
                    &mut self.persisted.display.show_overlay_labels,
                    "Show overlay labels",
                );
                ui.add(
                    egui::Slider::new(&mut self.persisted.display.overlay_label_scale, 1..=6)
                        .text("Label scale"),
                );
                ui.checkbox(
                    &mut self.persisted.display.highlight_on_hover,
                    "Highlight object on hover",
                );
                ui.checkbox(
                    &mut self.persisted.display.highlight_searched_object,
                    "Keep searched ID highlighted",
                );
                ui.horizontal(|ui| {
                    ui.label("Label color");
                    let mut color = egui::Color32::from_rgba_unmultiplied(
                        self.persisted.display.overlay_label_color[0],
                        self.persisted.display.overlay_label_color[1],
                        self.persisted.display.overlay_label_color[2],
                        self.persisted.display.overlay_label_color[3],
                    );
                    if ui.color_edit_button_srgba(&mut color).changed() {
                        self.persisted.display.overlay_label_color =
                            [color.r(), color.g(), color.b(), color.a()];
                        self.invalidate_texture();
                    }
                });
                if ui.button("Apply").clicked() {
                    self.invalidate_texture();
                    self.annotation.dialogs.overlay_labels_open = false;
                }
            });
        self.annotation.dialogs.overlay_labels_open = open;
    }
}
