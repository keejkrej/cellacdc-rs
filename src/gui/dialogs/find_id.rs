use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_find_id_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.find_id_open {
            return;
        }
        let mut open = self.annotation.dialogs.find_id_open;
        egui::Window::new("Find ID")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Enter an object ID to select and highlight.");
                ui.text_edit_singleline(&mut self.annotation.highlight.highlighted_input);
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        match self
                            .annotation
                            .highlight
                            .highlighted_input
                            .trim()
                            .parse::<u32>()
                        {
                            Ok(label) => {
                                if let Err(err) = self.annotation_select_label(Some(label)) {
                                    self.last_error = Some(err.to_string());
                                } else {
                                    self.annotation.highlight.searched_label = Some(label);
                                    self.invalidate_texture();
                                    self.annotation.dialogs.find_id_open = false;
                                }
                            }
                            Err(err) => self.last_error = Some(err.to_string()),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.annotation.dialogs.find_id_open = false;
                    }
                });
            });
        self.annotation.dialogs.find_id_open = open;
    }
}
