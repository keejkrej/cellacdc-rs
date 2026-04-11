use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;
use cellacdc_rs::{discover_import_sources, ImportSourceKind};
use eframe::egui::{self, RichText};

use super::{draw_workspace_header, path_edit_row, PathEditKind};

impl CellAcdcGui {
    pub(crate) fn draw_data_structure_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (back_to_launcher, _) = draw_workspace_header(
                ui,
                AppRoute::DataStructure,
                Some(
                    "Native source discovery for supported image stack formats. This is the first Rust-native step toward replacing the Python data-structure builder.",
                ),
                self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
                false,
            );
            if back_to_launcher {
                self.set_route(AppRoute::Launcher);
            }
            ui.add_space(8.0);
            path_edit_row(
                ui,
                "Source folder",
                &mut self.data_structure_scan_path,
                PathEditKind::PickFolder,
            );
            ui.horizontal(|ui| {
                if ui.button("Scan Supported Sources").clicked() {
                    let path = std::path::PathBuf::from(self.data_structure_scan_path.trim());
                    if path.as_os_str().is_empty() {
                        self.data_structure_scan_error =
                            Some("Choose a folder to scan for importable files.".to_string());
                    } else {
                        match discover_import_sources(&path) {
                            Ok(results) => {
                                self.data_structure_scan_results = results;
                                self.data_structure_scan_error = None;
                            }
                            Err(err) => {
                                self.data_structure_scan_results.clear();
                                self.data_structure_scan_error = Some(err.to_string());
                            }
                        }
                    }
                }
                if ui.button("Use current session folder").clicked() {
                    if let Some(experiment) = self.experiment.as_ref() {
                        self.data_structure_scan_path = experiment.root_path.display().to_string();
                    }
                }
            });

            if let Some(error) = self.data_structure_scan_error.clone() {
                ui.colored_label(egui::Color32::from_rgb(200, 60, 60), error);
            }

            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label(RichText::new("Discovered Sources").strong());
                if self.data_structure_scan_results.is_empty() {
                    ui.label("No supported sources scanned yet.");
                } else {
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for source in &self.data_structure_scan_results {
                            let label = match source.kind {
                                ImportSourceKind::Npz => "NPZ",
                                ImportSourceKind::H5 => "H5",
                                ImportSourceKind::Tiff => "TIFF",
                            };
                            ui.horizontal(|ui| {
                                ui.label(label);
                                ui.monospace(source.path.display().to_string());
                            });
                        }
                    });
                }
            });

            ui.add_space(8.0);
            ui.label("Full structure creation is deferred until the Rust-native import backend grows beyond discovery and validation.");
        });
    }
}
