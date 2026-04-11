use crate::gui::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_version_browser_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.version_browser_open {
            return;
        }
        let mut open = self.annotation.dialogs.version_browser_open;
        egui::Window::new("Load Older Versions")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                let Some(position) = self.selected_position().cloned() else {
                    ui.label("No session is open.");
                    return;
                };
                ui.label("Select a segmentation version to load.");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for asset in &position.segmentations {
                        let selected =
                            self.annotation.version_browser.selected_endname == asset.endname;
                        if ui.selectable_label(selected, &asset.name).clicked() {
                            self.annotation.version_browser.selected_endname = asset.endname.clone();
                        }
                    }
                });

                let recovery_available = self
                    .current_annotation_document()
                    .map(|document| document.session.recovery_state())
                    .map(|state| state == cellacdc_rs::MaskRecoveryState::RecoveryAvailable)
                    .unwrap_or(false);
                if recovery_available {
                    ui.separator();
                    if ui.button("Restore Recovery Autosave").clicked() {
                        if let Err(err) = self.restore_annotation_recovery() {
                            self.last_error = Some(err.to_string());
                        } else {
                            self.annotation.dialogs.version_browser_open = false;
                        }
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Load Selected").clicked() {
                        self.request_segmentation_selection(
                            self.annotation.version_browser.selected_endname.clone(),
                        );
                        self.annotation.dialogs.version_browser_open = false;
                    }
                    if ui.button("Close").clicked() {
                        self.annotation.dialogs.version_browser_open = false;
                    }
                });
            });
        self.annotation.dialogs.version_browser_open = open;
    }
}
