use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;
use eframe::egui::{self, Color32, RichText};

use super::{draw_launcher_module_button, draw_logs, LauncherModuleSpec};

impl CellAcdcGui {
    pub(crate) fn draw_launcher_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cell-ACDC");
            ui.label(
                "Press any of the following buttons to launch the respective module.",
            );
            ui.add_space(8.0);

            if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                ui.add_space(4.0);
            }

            if let Some(experiment) = self.experiment.as_ref() {
                ui.group(|ui| {
                    ui.label(RichText::new("Current Session").strong());
                    ui.monospace(experiment.root_path.display().to_string());
                    ui.label(format!("Positions: {}", experiment.positions.len()));
                });
            } else {
                ui.group(|ui| {
                    ui.label(RichText::new("Current Session").strong());
                    ui.label("No structured experiment is open.");
                });
            }

            ui.add_space(10.0);
            ui.group(|ui| {
                ui.label(RichText::new("Modules").strong());
                ui.add_space(6.0);
                for spec in [
                    LauncherModuleSpec {
                        title: "0. Create data structure from microscopy/image file(s)...",
                        subtitle: "Scan supported native image stacks before the full import workflow lands.",
                        route: AppRoute::DataStructure,
                        enabled_when_session_required: false,
                        is_primary_module: true,
                    },
                    LauncherModuleSpec {
                        title: "1. Launch data prep module...",
                        subtitle: "Reserved for the Rust-native data prep workflow and background ROI tools.",
                        route: AppRoute::DataPrep,
                        enabled_when_session_required: false,
                        is_primary_module: true,
                    },
                    LauncherModuleSpec {
                        title: "2. Launch segmentation module...",
                        subtitle: "Browse sessions, inspect channels and overlays, then run segmentation and measurement jobs.",
                        route: AppRoute::Segmentation,
                        enabled_when_session_required: true,
                        is_primary_module: true,
                    },
                    LauncherModuleSpec {
                        title: "3. Launch GUI...",
                        subtitle: "Open the future correction and annotation workspace.",
                        route: AppRoute::Annotation,
                        enabled_when_session_required: true,
                        is_primary_module: true,
                    },
                ] {
                    let enabled = !spec.enabled_when_session_required || self.experiment.is_some();
                    if draw_launcher_module_button(ui, spec) {
                        match spec.route {
                            AppRoute::Segmentation | AppRoute::Annotation if !enabled => {
                                self.pick_and_open_session();
                                if self.experiment.is_some() {
                                    self.set_route(spec.route);
                                }
                            }
                            route => self.set_route(route),
                        }
                    }
                    ui.add_space(4.0);
                }
            });

            ui.add_space(10.0);
            ui.group(|ui| {
                ui.label(RichText::new("Controls").strong());
                ui.add_space(6.0);
                if ui
                    .add_enabled(self.experiment.is_some(), egui::Button::new("Restore current session"))
                    .clicked()
                {
                    self.restore_current_session_route();
                }
                if ui.button("Open Experiment or Position").clicked() {
                    self.pick_and_open_session();
                }
            });

            ui.add_space(10.0);
            ui.group(|ui| {
                ui.label(RichText::new("Additional Tools").strong());
                ui.add_space(6.0);
                for spec in [
                    LauncherModuleSpec {
                        title: "Utilities",
                        subtitle: "Run file-based Rust utilities for masks and experiment outputs.",
                        route: AppRoute::Utilities,
                        enabled_when_session_required: false,
                        is_primary_module: false,
                    },
                    LauncherModuleSpec {
                        title: "Help",
                        subtitle: "See current port status, limitations, and next milestones.",
                        route: AppRoute::Help,
                        enabled_when_session_required: false,
                        is_primary_module: false,
                    },
                ] {
                    if draw_launcher_module_button(ui, spec) {
                        self.set_route(spec.route);
                    }
                    ui.add_space(4.0);
                }
            });

            ui.add_space(10.0);
            draw_logs(ui, &self.logs, 220.0);
        });
    }
}
