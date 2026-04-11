use crate::gui::actions::action_label;
use crate::gui::app::CellAcdcGui;
use crate::gui::state::{AnnotationTool, GuiActionId};
use cellacdc_rs::{MaskEditCommand, MaskRecoveryState};
use eframe::egui::{self, Color32, RichText};

use super::{draw_status_label, parse_label_input, validate_segm_endname};

impl CellAcdcGui {
    pub(crate) fn draw_annotation_panel(&mut self, ctx: &eframe::egui::Context) {
        self.ensure_annotation_document_loaded();
        self.draw_gui_chrome(ctx);
        self.draw_log_dock(ctx);
        self.draw_objects_dock(ctx);
        self.draw_gui_tools_panel(ctx);
        self.draw_canvas_panel_only(ctx);
        self.draw_annotation_pending_dialog(ctx);
        self.draw_gui_dialogs(ctx);
    }

    fn draw_gui_tools_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("gui_tools_panel")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("GUI Tools");

                if let Some(error) = self.last_error.clone() {
                    ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                    ui.add_space(4.0);
                }

                let Some(position) = self.selected_position().cloned() else {
                    ui.label("Open a Cell-ACDC session to use the GUI window.");
                    return;
                };
                if position.segmentations.is_empty() {
                    ui.label("No segmentation asset is available for this position.");
                    ui.small("Run segmentation first, then return to GUI for correction.");
                    return;
                }

                let selected_label = self.current_annotation_label();
                let recovery_state = self.annotation_recovery_state();
                let is_dirty = self.annotation_document_dirty();

                ui.label(RichText::new("Document").strong());
                if is_dirty {
                    draw_status_label(ui, "Unsaved edits", Color32::from_rgb(220, 160, 70));
                } else {
                    draw_status_label(ui, "Saved", Color32::from_rgb(80, 170, 90));
                }

                match recovery_state {
                    MaskRecoveryState::RecoveryAvailable => {
                        ui.add_space(8.0);
                        draw_status_label(
                            ui,
                            "Recovery data is available for this segmentation.",
                            Color32::from_rgb(220, 120, 60),
                        );
                        ui.label("Restore it to continue editing, or discard it and keep the saved mask.");
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Restore Recovery").clicked() {
                                if let Err(err) = self.restore_annotation_recovery() {
                                    self.last_error = Some(err.to_string());
                                }
                            }
                            if ui.button("Discard Recovery").clicked() {
                                if let Err(err) = self.discard_annotation_recovery() {
                                    self.last_error = Some(err.to_string());
                                }
                            }
                        });
                        return;
                    }
                    MaskRecoveryState::Recovered => {
                        draw_status_label(
                            ui,
                            "Recovered autosave loaded",
                            Color32::from_rgb(130, 180, 90),
                        );
                    }
                    MaskRecoveryState::None => {}
                }

                ui.separator();
                ui.label(RichText::new("Selected object").strong());
                ui.monospace(
                    selected_label
                        .map(|label| label.to_string())
                        .unwrap_or_else(|| "<none>".to_string()),
                );
                ui.horizontal_wrapped(|ui| {
                    for action in [
                        GuiActionId::ToolSelect,
                        GuiActionId::ToolBrush,
                        GuiActionId::ToolEraser,
                        GuiActionId::ToolRelabel,
                        GuiActionId::ToolMerge,
                        GuiActionId::ToolDelete,
                    ] {
                        let state = self.gui_action_state(action);
                        if ui
                            .add_enabled(
                                state.enabled,
                                egui::Button::new(action_label(action)).selected(state.checked),
                            )
                            .clicked()
                        {
                            self.dispatch_gui_action(action);
                        }
                    }
                });
                if matches!(self.annotation.tool, AnnotationTool::Brush | AnnotationTool::Eraser) {
                    ui.add(
                        egui::Slider::new(&mut self.annotation.brush_radius, 1..=20)
                            .text("Brush radius"),
                    );
                }

                ui.separator();
                ui.label(RichText::new("ID actions").strong());
                ui.horizontal(|ui| {
                    ui.label("Relabel target");
                    ui.text_edit_singleline(&mut self.annotation.relabel_target);
                });
                if ui
                    .add_enabled(selected_label.is_some(), egui::Button::new("Apply Relabel"))
                    .clicked()
                {
                    let action = (|| -> anyhow::Result<()> {
                        let from = selected_label.ok_or_else(|| anyhow::anyhow!("Select an ID to relabel"))?;
                        let to = parse_label_input(&self.annotation.relabel_target, "Relabel target")?;
                        self.run_annotation_command(MaskEditCommand::ReplaceLabel { from, to })?;
                        self.annotation_select_label(Some(to))?;
                        Ok(())
                    })();
                    if let Err(err) = action {
                        self.last_error = Some(err.to_string());
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Merge target");
                    ui.text_edit_singleline(&mut self.annotation.merge_target);
                });
                if ui
                    .add_enabled(selected_label.is_some(), egui::Button::new("Merge Into Target"))
                    .clicked()
                {
                    let action = (|| -> anyhow::Result<()> {
                        let from = selected_label.ok_or_else(|| anyhow::anyhow!("Select an ID to merge"))?;
                        let to = parse_label_input(&self.annotation.merge_target, "Merge target")?;
                        self.run_annotation_command(MaskEditCommand::ReplaceLabel { from, to })?;
                        self.annotation_select_label(Some(to))?;
                        Ok(())
                    })();
                    if let Err(err) = action {
                        self.last_error = Some(err.to_string());
                    }
                }
                if ui
                    .add_enabled(selected_label.is_some(), egui::Button::new("Delete Selected ID"))
                    .clicked()
                {
                    let action = (|| -> anyhow::Result<()> {
                        let label = selected_label.ok_or_else(|| anyhow::anyhow!("Select an ID to delete"))?;
                        self.run_annotation_command(MaskEditCommand::DeleteLabel { label })?;
                        self.annotation_select_label(None)?;
                        Ok(())
                    })();
                    if let Err(err) = action {
                        self.last_error = Some(err.to_string());
                    }
                }

                ui.separator();
                ui.label(RichText::new("Document actions").strong());
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Undo").clicked() {
                        self.annotation_undo();
                    }
                    if ui.button("Redo").clicked() {
                        self.annotation_redo();
                    }
                    if ui.button("Save").clicked() {
                        if let Err(err) = self.save_current_annotation_overwrite() {
                            self.last_error = Some(err.to_string());
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.text_edit_singleline(&mut self.annotation.save_as_endname);
                    if ui.button("Save As Version").clicked() {
                        let action = (|| -> anyhow::Result<()> {
                            self.annotation.save_as_endname =
                                validate_segm_endname(&self.annotation.save_as_endname)?;
                            self.save_current_annotation_as_version()
                        })();
                        if let Err(err) = action {
                            self.last_error = Some(err.to_string());
                        }
                    }
                });

                ui.separator();
                ui.label(RichText::new("Jobs").strong());
                if ui.button("Run segmentation on current position").clicked() {
                    self.start_run_position_job();
                }
                if ui.button("Measure current position").clicked() {
                    self.start_measure_position_job();
                }
                if ui.button("Open segmentation workspace").clicked() {
                    self.set_route(crate::gui::state::AppRoute::Segmentation);
                }
            });
    }

    fn draw_annotation_pending_dialog(&mut self, ctx: &egui::Context) {
        if self.annotation.pending_action.is_none() {
            return;
        }
        egui::Window::new("Unsaved GUI edits")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("The current GUI document has unsaved edits.");
                ui.label(
                    "Save the current segmentation before switching, discard the edits, or cancel.",
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Save and Continue").clicked() {
                        self.save_annotation_changes_and_continue();
                    }
                    if ui.button("Discard and Continue").clicked() {
                        self.discard_annotation_changes_and_continue();
                    }
                    if ui.button("Cancel").clicked() {
                        self.cancel_pending_annotation_action();
                    }
                });
            });
    }
}
