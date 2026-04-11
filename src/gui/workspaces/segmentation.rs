use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;
use eframe::egui::{self, Color32, RichText};
use std::time::Duration;

use super::{
    combo_for_channel, draw_logs, path_edit_row, route_label, selected_segm_label, PathEditKind,
};

impl CellAcdcGui {
    pub(crate) fn draw_navigation(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                for route in [
                    AppRoute::Launcher,
                    AppRoute::DataStructure,
                    AppRoute::DataPrep,
                    AppRoute::Segmentation,
                    AppRoute::Annotation,
                    AppRoute::Utilities,
                    AppRoute::Help,
                ] {
                    ui.selectable_value(&mut self.persisted.route, route, route_label(route));
                }
                ui.separator();

                if ui.button("Open Experiment or Position").clicked() {
                    self.pick_and_open_session();
                }
                if ui
                    .add_enabled(self.experiment.is_some(), egui::Button::new("Reload"))
                    .clicked()
                {
                    self.reload_experiment();
                }
                let cancel_request = self
                    .active_job
                    .as_ref()
                    .map(|job| (ui.button("Cancel Job").clicked(), job.label.clone()));
                if let Some((true, label)) = cancel_request {
                    if let Some(job) = self.active_job.as_ref() {
                        job.cancel();
                    }
                    self.append_log(format!("Cancellation requested: {label}"));
                }

                if !self.persisted.recent_paths.is_empty() {
                    egui::ComboBox::from_label("Recent")
                        .selected_text("Select path")
                        .show_ui(ui, |ui| {
                            for recent in self.persisted.recent_paths.clone() {
                                if ui.selectable_label(false, &recent).clicked() {
                                    if let Err(err) =
                                        self.open_path(std::path::PathBuf::from(recent))
                                    {
                                        self.last_error = Some(err.to_string());
                                    }
                                }
                            }
                        });
                }

                if let Some(job) = self.active_job.as_ref() {
                    ui.separator();
                    ui.label(RichText::new(format!("Running: {}", job.label)).strong());
                }
            });
        });
    }

    pub(crate) fn render_segmentation_workspace(&mut self, ctx: &egui::Context) {
        self.draw_left_panel(ctx);
        self.draw_jobs_panel(ctx);
        self.draw_viewer_panel(ctx);
    }

    pub(crate) fn draw_jobs_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("jobs_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("Segmentation");
                let Some(position) = self.selected_position().cloned() else {
                    ui.label("Open a session to configure jobs.");
                    return;
                };
                let channel_names = position.channel_names();
                let has_position_segmentations = !position.segmentations.is_empty();
                let experiment_supports_batch = self
                    .experiment
                    .as_ref()
                    .map(|experiment| !experiment.is_single_position)
                    .unwrap_or(false);

                if let Some(error) = self.last_error.clone() {
                    ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                    ui.add_space(4.0);
                }

                ui.label(RichText::new("Segmentation Jobs").strong());
                combo_for_channel(
                    ui,
                    "Phase channel",
                    &channel_names,
                    &mut self.persisted.phase_channel,
                );
                combo_for_channel(
                    ui,
                    "Fluorescence channel",
                    &channel_names,
                    &mut self.persisted.fluo_channel,
                );
                path_edit_row(
                    ui,
                    "Model",
                    &mut self.persisted.model_path,
                    PathEditKind::PickFile,
                );
                ui.horizontal(|ui| {
                    ui.label("Output suffix");
                    ui.text_edit_singleline(&mut self.persisted.run_output_suffix);
                });
                ui.checkbox(&mut self.persisted.cpu, "Run on CPU");
                ui.checkbox(&mut self.persisted.track, "Track after segmentation");
                if self.persisted.track {
                    ui.add(
                        egui::Slider::new(&mut self.persisted.track_ioa_threshold, 0.0..=1.0)
                            .text("Tracking IoA"),
                    );
                }
                ui.checkbox(&mut self.persisted.overwrite_outputs, "Overwrite outputs");
                ui.collapsing("Segmentation parameters", |ui| {
                    ui.add(egui::DragValue::new(&mut self.persisted.tile).prefix("tile "));
                    ui.add(egui::DragValue::new(&mut self.persisted.batch_size).prefix("batch "));
                    ui.add(
                        egui::DragValue::new(&mut self.persisted.cellprob_threshold)
                            .prefix("cellprob "),
                    );
                    ui.add(egui::DragValue::new(&mut self.persisted.niter).prefix("niter "));
                    ui.add(egui::DragValue::new(&mut self.persisted.min_size).prefix("min size "));
                });

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(self.active_job.is_none(), egui::Button::new("Run Position"))
                        .clicked()
                    {
                        self.start_run_position_job();
                    }
                    if ui
                        .add_enabled(
                            self.active_job.is_none() && experiment_supports_batch,
                            egui::Button::new("Run Experiment"),
                        )
                        .clicked()
                    {
                        self.start_run_experiment_job();
                    }
                });

                ui.separator();
                ui.label(RichText::new("Measurements").strong());
                ui.label(format!(
                    "Selected segmentation: {}",
                    selected_segm_label(&position, &self.persisted.selected_segmentation_endname)
                ));

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            self.active_job.is_none() && has_position_segmentations,
                            egui::Button::new("Measure Position"),
                        )
                        .clicked()
                    {
                        self.start_measure_position_job();
                    }
                    if ui
                        .add_enabled(
                            self.active_job.is_none()
                                && has_position_segmentations
                                && experiment_supports_batch,
                            egui::Button::new("Measure Experiment"),
                        )
                        .clicked()
                    {
                        self.start_measure_experiment_job();
                    }
                });

                ui.separator();
                ui.label(RichText::new("Workspace").strong());
                ui.label(
                    "This module is the Rust-native replacement for the Qt segmentation launcher and image review shell.",
                );
                draw_logs(ui, &self.logs, 280.0);
            });
    }

    pub(crate) fn request_repaint_for_active_job(&self, ctx: &egui::Context) {
        if self.active_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}
