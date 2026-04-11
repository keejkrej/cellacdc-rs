use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;
use eframe::egui::{self, Color32, RichText};

use super::{draw_logs, launcher_placeholder_card};

impl CellAcdcGui {
    pub(crate) fn draw_launcher_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cell-ACDC Rust");
            ui.label(
                "Rust-native launcher preserving the Python app structure: data ingestion, data prep, segmentation, annotation, utilities, and help.",
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
                    ui.horizontal(|ui| {
                        if ui.button("Open Segmentation").clicked() {
                            self.persisted.route = AppRoute::Segmentation;
                        }
                        if ui.button("Open Utilities").clicked() {
                            self.persisted.route = AppRoute::Utilities;
                        }
                    });
                });
            } else {
                ui.group(|ui| {
                    ui.label(RichText::new("Current Session").strong());
                    ui.label("No structured experiment is open.");
                    if ui.button("Open Experiment or Position").clicked() {
                        self.pick_and_open_session();
                    }
                });
            }

            ui.add_space(10.0);
            ui.columns(3, |columns| {
                columns[0].group(|ui| {
                    ui.label(RichText::new("Data Structure").strong());
                    ui.label(
                        "Scan folders for supported native image stacks before the full Rust import workflow lands.",
                    );
                    if ui.button("Open Data Structure").clicked() {
                        self.persisted.route = AppRoute::DataStructure;
                    }
                });

                columns[1].group(|ui| {
                    ui.label(RichText::new("Segmentation").strong());
                    ui.label(
                        "Browse sessions, inspect channels and overlays, then run segmentation and measurement jobs.",
                    );
                    if ui
                        .add_enabled(self.experiment.is_some(), egui::Button::new("Open Segmentation"))
                        .clicked()
                    {
                        self.persisted.route = AppRoute::Segmentation;
                    }
                });

                columns[2].group(|ui| {
                    ui.label(RichText::new("Utilities").strong());
                    ui.label(
                        "Run file-based Rust utilities for masks and experiment outputs on native Cell-ACDC data.",
                    );
                    if ui.button("Open Utilities").clicked() {
                        self.persisted.route = AppRoute::Utilities;
                    }
                });
            });

            ui.add_space(10.0);
            ui.columns(3, |columns| {
                columns[0].group(|ui| {
                    ui.label(RichText::new("Data Prep").strong());
                    ui.label(
                        "Planned Rust-native alignment, cropping, z selection, and background ROI management.",
                    );
                    if ui.button("Open Data Prep").clicked() {
                        self.persisted.route = AppRoute::DataPrep;
                    }
                });

                columns[1].group(|ui| {
                    ui.label(RichText::new("Annotation").strong());
                    ui.label(
                        "Planned mask correction workspace with undo/redo, autosave, and tracking-aware edits.",
                    );
                    if ui.button("Open Annotation").clicked() {
                        self.persisted.route = AppRoute::Annotation;
                    }
                });

                columns[2].group(|ui| {
                    ui.label(RichText::new("Help").strong());
                    ui.label(
                        "Project status, native limitations, and port roadmap surfaced inside the app shell.",
                    );
                    if ui.button("Open Help").clicked() {
                        self.persisted.route = AppRoute::Help;
                    }
                });
            });

            ui.add_space(10.0);
            ui.columns(3, |columns| {
                launcher_placeholder_card(
                    &mut columns[0],
                    "Data Prep MVP",
                    "Background ROI JSON/NPZ compatibility is now defined in Rust; interactive authoring is next.",
                    Some("In Progress"),
                );
                launcher_placeholder_card(
                    &mut columns[1],
                    "Annotation MVP",
                    "Foundational mask-edit session, undo stack, and autosave helpers are available for future UI wiring.",
                    Some("In Progress"),
                );
                launcher_placeholder_card(
                    &mut columns[2],
                    "Data Import",
                    "Native source discovery exists for .npz, .h5, and .tif/.tiff files; full structure creation remains planned.",
                    Some("In Progress"),
                );
            });

            ui.add_space(10.0);
            draw_logs(ui, &self.logs, 220.0);
        });
    }
}
