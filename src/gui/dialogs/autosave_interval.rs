use crate::gui::app::CellAcdcGui;
use crate::gui::state::AutosaveUnit;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_autosave_interval_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.autosave_interval_open {
            return;
        }
        let mut open = self.annotation.dialogs.autosave_interval_open;
        egui::Window::new("Autosave interval")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Set the delay before dirty edits are written to the recovery autosave.");
                ui.add(
                    egui::DragValue::new(&mut self.persisted.display.autosave.value)
                        .range(1..=120),
                );
                egui::ComboBox::from_label("Unit")
                    .selected_text(match self.persisted.display.autosave.unit {
                        AutosaveUnit::Seconds => "Seconds",
                        AutosaveUnit::Minutes => "Minutes",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.persisted.display.autosave.unit,
                            AutosaveUnit::Seconds,
                            "Seconds",
                        );
                        ui.selectable_value(
                            &mut self.persisted.display.autosave.unit,
                            AutosaveUnit::Minutes,
                            "Minutes",
                        );
                    });
                if ui.button("Close").clicked() {
                    self.annotation.dialogs.autosave_interval_open = false;
                }
            });
        self.annotation.dialogs.autosave_interval_open = open;
    }
}
