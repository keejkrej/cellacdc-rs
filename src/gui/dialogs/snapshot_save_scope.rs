use crate::gui::app::CellAcdcGui;
use cellacdc_rs::{resolve_snapshot_save_scope, SnapshotSaveScope};
use eframe::egui;
use std::collections::BTreeSet;

impl CellAcdcGui {
    pub(crate) fn draw_snapshot_save_scope_dialog(&mut self, ctx: &egui::Context) {
        if !self.annotation.dialogs.snapshot_save_scope_open {
            return;
        }
        let mut open = self.annotation.dialogs.snapshot_save_scope_open;
        let mut save_clicked = false;
        let mut cancel_clicked = false;
        let positions = self.experiment_position_keys();

        egui::Window::new("Snapshot Save Scope")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Choose which positions to save for this snapshot session.");
                for key in &positions {
                    let selected = self
                        .annotation
                        .snapshot_save_dialog
                        .selected_positions
                        .iter()
                        .any(|item| item == key);
                    let mut checked = selected;
                    if ui.checkbox(&mut checked, key).changed() {
                        if checked {
                            self.annotation
                                .snapshot_save_dialog
                                .selected_positions
                                .push(key.clone());
                            self.annotation
                                .snapshot_save_dialog
                                .selected_positions
                                .sort();
                            self.annotation
                                .snapshot_save_dialog
                                .selected_positions
                                .dedup();
                        } else {
                            self.annotation
                                .snapshot_save_dialog
                                .selected_positions
                                .retain(|item| item != key);
                        }
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("Save Selected").clicked() {
                        save_clicked = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                });
            });

        if save_clicked {
            let selected = self
                .annotation
                .snapshot_save_dialog
                .selected_positions
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            let current = self.current_position_key().unwrap_or_default();
            match resolve_snapshot_save_scope(&selected, &current) {
                Ok(SnapshotSaveScope::CurrentPosition) => {
                    if let Err(err) =
                        self.save_current_annotation_overwrite_for_positions(&[current])
                    {
                        self.last_error = Some(err.to_string());
                    } else {
                        cancel_clicked = true;
                    }
                }
                Ok(SnapshotSaveScope::SelectedPositions(selected)) => {
                    let keys = selected.into_iter().collect::<Vec<_>>();
                    if let Err(err) = self.save_current_annotation_overwrite_for_positions(&keys) {
                        self.last_error = Some(err.to_string());
                    } else {
                        cancel_clicked = true;
                    }
                }
                Err(err) => {
                    self.last_error = Some(err.to_string());
                }
            }
        }

        if cancel_clicked {
            open = false;
        }

        self.annotation.dialogs.snapshot_save_scope_open = open;
    }
}
