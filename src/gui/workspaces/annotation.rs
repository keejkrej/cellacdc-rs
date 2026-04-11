use crate::gui::app::CellAcdcGui;
use crate::gui::state::{AnnotationTool, AppRoute};
use cellacdc_rs::{MaskEditCommand, MaskRecoveryState};
use eframe::egui::{self, Color32, RichText};

use super::{
    draw_status_label, draw_workspace_header, parse_label_input, selected_segm_label,
    validate_segm_endname,
};

impl CellAcdcGui {
    pub(crate) fn draw_annotation_panel(&mut self, ctx: &eframe::egui::Context) {
        self.ensure_annotation_document_loaded();
        let (back_to_launcher, open_session) = egui::TopBottomPanel::top("annotation_header")
            .show(ctx, |ui| {
                draw_workspace_header(
                    ui,
                    AppRoute::Annotation,
                    Some("Review segmentation masks, make native corrections, and save safely."),
                    self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
                    self.experiment.is_none(),
                )
            })
            .inner;
        if back_to_launcher {
            self.set_route(AppRoute::Launcher);
        }
        if open_session {
            self.pick_and_open_session();
        }
        self.draw_left_panel(ctx);
        self.draw_annotation_tools_panel(ctx);
        self.draw_viewer_panel(ctx);
        self.draw_annotation_pending_dialog(ctx);
    }

    fn draw_annotation_tools_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("annotation_tools_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("GUI Tools");

                if let Some(error) = self.last_error.clone() {
                    ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                    ui.add_space(4.0);
                }

                let Some(position) = self.selected_position().cloned() else {
                    ui.label("Open a Cell-ACDC session to review and correct segmentation masks.");
                    return;
                };
                if position.segmentations.is_empty() {
                    ui.label("No segmentation asset is available for this position.");
                    ui.small("Run segmentation first, then come back to GUI for correction.");
                    return;
                }

                let selected_segm = selected_segm_label(
                    &position,
                    &self.persisted.selected_segmentation_endname,
                );
                let selected_label = self.current_annotation_label();
                let recovery_state = self.annotation_recovery_state();
                let is_dirty = self.annotation_document_dirty();
                let doc_path = self
                    .current_annotation_document()
                    .and_then(|document| document.session.path())
                    .map(|path| path.display().to_string());

                ui.label(RichText::new("Document").strong());
                ui.label(format!("Editing: {selected_segm}"));
                if let Some(path) = doc_path {
                    ui.monospace(path);
                }
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
                        ui.separator();
                        ui.label("Editing is locked until the recovery decision is resolved.");
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

                if self.current_annotation_document().is_none() {
                    ui.separator();
                    ui.label("Failed to load the current segmentation as an editable mask document.");
                    return;
                }

                if !self.annotation_edits_allowed() {
                    ui.separator();
                    draw_status_label(
                        ui,
                        "3D masks are editable only in Z slice mode.",
                        Color32::from_rgb(200, 120, 60),
                    );
                }

                ui.separator();
                ui.label(RichText::new("Selection").strong());
                match selected_label {
                    Some(label) => ui.monospace(format!("Selected ID: {label}")),
                    None => ui.monospace("Selected ID: <none>"),
                };

                ui.add_space(6.0);
                ui.label(RichText::new("Tools").strong());
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.annotation.tool, AnnotationTool::Select, "Select");
                    ui.selectable_value(&mut self.annotation.tool, AnnotationTool::Brush, "Brush");
                    ui.selectable_value(&mut self.annotation.tool, AnnotationTool::Eraser, "Eraser");
                    ui.selectable_value(
                        &mut self.annotation.tool,
                        AnnotationTool::Relabel,
                        "Relabel",
                    );
                    ui.selectable_value(&mut self.annotation.tool, AnnotationTool::Merge, "Merge");
                    ui.selectable_value(&mut self.annotation.tool, AnnotationTool::Delete, "Delete");
                });

                if matches!(self.annotation.tool, AnnotationTool::Brush | AnnotationTool::Eraser) {
                    ui.add(
                        egui::Slider::new(&mut self.annotation.brush_radius, 1..=20)
                            .text("Brush radius"),
                    );
                }

                ui.separator();
                ui.label(RichText::new("ID Actions").strong());
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
                ui.label(RichText::new("Document Actions").strong());
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Undo").clicked() {
                        self.annotation_undo();
                    }
                    if ui.button("Redo").clicked() {
                        self.annotation_redo();
                    }
                });

                ui.horizontal_wrapped(|ui| {
                    if ui.button("Save").clicked() {
                        if let Err(err) = self.save_current_annotation_overwrite() {
                            self.last_error = Some(err.to_string());
                        }
                    }
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
                ui.small("Save As Version writes a new `segm_<endname>.npz` file inside the current position.");
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
                ui.label("Save the current segmentation before switching, discard the edits, or cancel.");
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
