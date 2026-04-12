use crate::gui::actions::{action_label, default_shortcut_binding};
use crate::gui::app::CellAcdcGui;
use crate::gui::shortcuts::{
    detect_shortcut_capture, display_binding, set_override, validate_shortcut_overrides,
};
use crate::gui::state::GuiActionId;
use eframe::egui;

const SHORTCUT_ACTIONS: &[GuiActionId] = &[
    GuiActionId::Save,
    GuiActionId::Undo,
    GuiActionId::Redo,
    GuiActionId::FindId,
    GuiActionId::ToolSelect,
    GuiActionId::ToolBrush,
    GuiActionId::ToolEraser,
    GuiActionId::ManualTracking,
    GuiActionId::RepeatTracking,
    GuiActionId::AssignMotherToBud,
    GuiActionId::UnknownLineage,
    GuiActionId::NoLineageTool,
    GuiActionId::PropagateLineage,
];

impl CellAcdcGui {
    pub(crate) fn draw_shortcut_editor_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.shortcut_editor_open {
            return;
        }
        if let Some(action) = self.annotation.shortcut_editor.capturing {
            if let Some(binding) = detect_shortcut_capture(ctx) {
                let mut overrides = self.persisted.shortcut_overrides.clone();
                set_override(&mut overrides, action, binding);
                match validate_shortcut_overrides(&overrides) {
                    Ok(()) => {
                        self.persisted.shortcut_overrides = overrides;
                        self.annotation.shortcut_editor.error = None;
                        self.annotation.shortcut_editor.capturing = None;
                    }
                    Err(err) => {
                        self.annotation.shortcut_editor.error = Some(err.to_string());
                        self.annotation.shortcut_editor.capturing = None;
                    }
                }
            }
        }

        let mut open = self.annotation.dialogs.shortcut_editor_open;
        egui::Window::new("Customize keyboard shortcuts")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                if let Some(error) = &self.annotation.shortcut_editor.error {
                    ui.colored_label(egui::Color32::from_rgb(200, 60, 60), error);
                }
                if let Some(action) = self.annotation.shortcut_editor.capturing {
                    ui.label(format!(
                        "Press a new shortcut for {}...",
                        action_label(action)
                    ));
                }
                for action in SHORTCUT_ACTIONS {
                    ui.horizontal(|ui| {
                        ui.label(action_label(*action));
                        let current = self
                            .persisted
                            .shortcut_overrides
                            .bindings
                            .iter()
                            .find(|entry| entry.action == *action)
                            .map(|entry| entry.binding.clone())
                            .or_else(|| default_shortcut_binding(*action));
                        ui.monospace(
                            current
                                .map(|binding| display_binding(&binding))
                                .unwrap_or_else(|| "<none>".to_string()),
                        );
                        if ui.button("Rebind").clicked() {
                            self.annotation.shortcut_editor.capturing = Some(*action);
                        }
                        if ui.button("Default").clicked() {
                            if let Some(binding) = default_shortcut_binding(*action) {
                                let mut overrides = self.persisted.shortcut_overrides.clone();
                                set_override(&mut overrides, *action, binding);
                                if let Err(err) = validate_shortcut_overrides(&overrides) {
                                    self.annotation.shortcut_editor.error = Some(err.to_string());
                                } else {
                                    self.persisted.shortcut_overrides = overrides;
                                    self.annotation.shortcut_editor.error = None;
                                }
                            }
                        }
                    });
                }
                ui.separator();
                if ui.button("Restore all defaults").clicked() {
                    self.persisted.shortcut_overrides.bindings.clear();
                    self.annotation.shortcut_editor.error = None;
                }
            });
        self.annotation.dialogs.shortcut_editor_open = open;
    }
}
