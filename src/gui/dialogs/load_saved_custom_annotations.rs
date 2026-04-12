use crate::gui::app::CellAcdcGui;
use cellacdc_rs::{
    global_custom_annotation_definitions_path, load_custom_annotation_definitions,
    save_custom_annotation_definitions,
};
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_load_saved_custom_annotations_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.load_saved_custom_annotations_open {
            return;
        }
        let mut open = self.annotation.dialogs.load_saved_custom_annotations_open;
        let global_path = global_custom_annotation_definitions_path();
        let mut saved = load_custom_annotation_definitions(&global_path).unwrap_or_default();
        let mut load_clicked = false;
        let mut delete_clicked = false;
        let mut close_clicked = false;

        egui::Window::new("Load Previously Used Custom Annotations")
            .collapsible(false)
            .resizable(true)
            .open(&mut open)
            .show(ctx, |ui| {
                if saved.is_empty() {
                    ui.label("No saved custom annotation definitions were found.");
                } else {
                    ui.label("Select one or more saved definitions to load into this position.");
                    egui::ScrollArea::vertical()
                        .max_height(260.0)
                        .show(ui, |ui| {
                            for (name, definition) in &saved {
                                let selected = self
                                    .annotation
                                    .saved_custom_annotations_dialog
                                    .selected_names
                                    .iter()
                                    .any(|item| item == name);
                                let mut checked = selected;
                                if ui
                                    .checkbox(
                                        &mut checked,
                                        format!(
                                            "{}  [{}]{}",
                                            name,
                                            definition.symbol,
                                            definition
                                                .shortcut
                                                .as_deref()
                                                .map(|value| format!("  shortcut: {value}"))
                                                .unwrap_or_default()
                                        ),
                                    )
                                    .changed()
                                {
                                    if checked {
                                        self.annotation
                                            .saved_custom_annotations_dialog
                                            .selected_names
                                            .push(name.clone());
                                        self.annotation
                                            .saved_custom_annotations_dialog
                                            .selected_names
                                            .sort();
                                        self.annotation
                                            .saved_custom_annotations_dialog
                                            .selected_names
                                            .dedup();
                                    } else {
                                        self.annotation
                                            .saved_custom_annotations_dialog
                                            .selected_names
                                            .retain(|item| item != name);
                                    }
                                }
                            }
                        });
                }

                ui.horizontal(|ui| {
                    if ui.button("Load Selected").clicked() {
                        load_clicked = true;
                    }
                    if ui.button("Delete Selected From Saved List").clicked() {
                        delete_clicked = true;
                    }
                    if ui.button("Close").clicked() {
                        close_clicked = true;
                    }
                });
            });

        if load_clicked {
            let names = self
                .annotation
                .saved_custom_annotations_dialog
                .selected_names
                .clone();
            for name in &names {
                if let Some(definition) = saved.get(name).cloned() {
                    self.annotation
                        .custom_annotations
                        .definitions
                        .insert(name.clone(), definition);
                }
            }
            if !names.is_empty() {
                if let Some(position_key) = self.current_position_key() {
                    if let Err(err) = self.persist_custom_annotation_definitions(&[position_key]) {
                        self.last_error = Some(err.to_string());
                    } else {
                        self.invalidate_texture();
                        close_clicked = true;
                    }
                }
            }
        }

        if delete_clicked {
            let names = self
                .annotation
                .saved_custom_annotations_dialog
                .selected_names
                .clone();
            for name in &names {
                saved.remove(name);
            }
            if let Err(err) = save_custom_annotation_definitions(&global_path, &saved) {
                self.last_error = Some(err.to_string());
            } else {
                self.annotation
                    .saved_custom_annotations_dialog
                    .selected_names
                    .clear();
            }
        }

        if close_clicked {
            open = false;
        }

        self.annotation.dialogs.load_saved_custom_annotations_open = open;
    }
}
