use crate::gui::app::CellAcdcGui;
use crate::gui::state::AppRoute;
use cellacdc_rs::{
    discover_import_sources, probe_import_source, read_import_sample_planes, ImportConflictMode,
    ImportLayoutKind, ImportOutputFormat, ImportReaderBackend, MetadataReusePolicy,
};
use eframe::egui::{self, ColorImage, RichText, TextureOptions};
use rfd::FileDialog;

use super::{draw_logs, draw_workspace_header, path_edit_row, PathEditKind};

impl CellAcdcGui {
    pub(crate) fn draw_data_structure_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (back_to_launcher, _) = draw_workspace_header(
                ui,
                AppRoute::DataStructure,
                Some(
                    "Create a Cell-ACDC experiment from raw microscopy or image-stack sources. Native TIFF/NPZ/H5 import works now. Vendor microscopy formats use the Bio-Formats bridge when available.",
                ),
                self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
                false,
            );
            if back_to_launcher {
                self.set_route(AppRoute::Launcher);
            }

            ui.add_space(8.0);
            ui.columns(2, |columns| {
                self.draw_data_structure_controls(&mut columns[0]);
                self.draw_data_structure_preview(ctx, &mut columns[1]);
            });

            ui.add_space(8.0);
            draw_logs(ui, &self.logs, 140.0);
        });
    }

    fn draw_data_structure_controls(&mut self, ui: &mut egui::Ui) {
        path_edit_row(
            ui,
            "Source folder or file",
            &mut self.data_structure.source_path,
            PathEditKind::PickFolder,
        );
        ui.horizontal(|ui| {
            if ui.button("Pick File").clicked() {
                if let Some(path) = FileDialog::new().pick_file() {
                    self.data_structure.source_path = path.display().to_string();
                }
            }
            if ui.button("Use current session folder").clicked() {
                if let Some(experiment) = self.experiment.as_ref() {
                    self.data_structure.source_path = experiment.root_path.display().to_string();
                }
            }
        });
        path_edit_row(
            ui,
            "Destination experiment",
            &mut self.data_structure.destination_path,
            PathEditKind::PickFolder,
        );

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Layout")
                .selected_text(import_layout_label(self.data_structure.layout_kind))
                .show_ui(ui, |ui| {
                    for layout in [
                        ImportLayoutKind::SingleFileMultiPosition,
                        ImportLayoutKind::FilePerPosition,
                        ImportLayoutKind::FilePerChannel,
                        ImportLayoutKind::CustomMapping,
                    ] {
                        ui.selectable_value(
                            &mut self.data_structure.layout_kind,
                            layout,
                            import_layout_label(layout),
                        );
                    }
                });
            egui::ComboBox::from_label("Backend")
                .selected_text(import_backend_label(self.data_structure.backend))
                .show_ui(ui, |ui| {
                    for backend in [
                        ImportReaderBackend::Auto,
                        ImportReaderBackend::Native,
                        ImportReaderBackend::BioFormatsJvmBridge,
                    ] {
                        ui.selectable_value(
                            &mut self.data_structure.backend,
                            backend,
                            import_backend_label(backend),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Conflict")
                .selected_text(import_conflict_label(self.data_structure.conflict_mode))
                .show_ui(ui, |ui| {
                    for mode in [
                        ImportConflictMode::OverwritePositionFiles,
                        ImportConflictMode::AddFilesToExistingExperiment,
                        ImportConflictMode::CreateNewPositions,
                    ] {
                        ui.selectable_value(
                            &mut self.data_structure.conflict_mode,
                            mode,
                            import_conflict_label(mode),
                        );
                    }
                });
            egui::ComboBox::from_label("Metadata policy")
                .selected_text(metadata_policy_label(self.data_structure.metadata_policy))
                .show_ui(ui, |ui| {
                    for policy in [
                        MetadataReusePolicy::ConfirmEverySource,
                        MetadataReusePolicy::UseForRemainingSources,
                        MetadataReusePolicy::TrustReaderForRemainingSources,
                    ] {
                        ui.selectable_value(
                            &mut self.data_structure.metadata_policy,
                            policy,
                            metadata_policy_label(policy),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Output")
                .selected_text(import_output_label(self.data_structure.output_format))
                .show_ui(ui, |ui| {
                    for format in [ImportOutputFormat::Tiff, ImportOutputFormat::H5] {
                        ui.selectable_value(
                            &mut self.data_structure.output_format,
                            format,
                            import_output_label(format),
                        );
                    }
                });
            ui.checkbox(
                &mut self.data_structure.add_image_name,
                "Add image name to basename",
            );
        });

        self.persisted.data_structure_backend = self.data_structure.backend;
        self.persisted.data_structure_layout_kind = self.data_structure.layout_kind;
        self.persisted.data_structure_conflict_mode = self.data_structure.conflict_mode;
        self.persisted.data_structure_metadata_policy = self.data_structure.metadata_policy;
        self.persisted.data_structure_output_format = self.data_structure.output_format;
        self.persisted.data_structure_destination_path =
            self.data_structure.destination_path.clone();

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("Scan Sources").clicked() {
                let source_path = std::path::PathBuf::from(self.data_structure.source_path.trim());
                match discover_import_sources(&source_path) {
                    Ok(sources) => {
                        self.data_structure.discovered_sources = sources;
                        self.data_structure.metadata_drafts.clear();
                        self.data_structure.sample_planes = None;
                        self.data_structure.selected_source_idx = 0;
                        self.data_structure.error = None;
                    }
                    Err(err) => self.data_structure.error = Some(err.to_string()),
                }
            }
            if ui
                .add_enabled(
                    !self.data_structure.discovered_sources.is_empty(),
                    egui::Button::new("Probe Metadata"),
                )
                .clicked()
            {
                let mut drafts = Vec::new();
                let mut error = None;
                for source in &self.data_structure.discovered_sources {
                    match probe_import_source(&source.path, self.data_structure.backend) {
                        Ok(draft) => drafts.push(draft),
                        Err(err) => {
                            error = Some(err.to_string());
                            break;
                        }
                    }
                }
                if let Some(error) = error {
                    self.data_structure.error = Some(error);
                } else {
                    self.data_structure.metadata_drafts = drafts;
                    self.sync_data_structure_selection_state();
                    self.data_structure.error = None;
                }
            }
            if ui
                .add_enabled(
                    !self.data_structure.discovered_sources.is_empty(),
                    egui::Button::new("Load Sample Preview"),
                )
                .clicked()
            {
                if let Some(source) = self
                    .data_structure
                    .discovered_sources
                    .get(self.data_structure.selected_source_idx)
                {
                    match read_import_sample_planes(&source.path, self.data_structure.backend) {
                        Ok(sample) => {
                            self.data_structure.sample_planes = Some(sample);
                            self.data_structure.preview_frame_index = 0;
                            self.data_structure.preview_z_index = 0;
                            self.data_structure.error = None;
                        }
                        Err(err) => self.data_structure.error = Some(err.to_string()),
                    }
                }
            }
        });

        if let Some(error) = self.data_structure.error.clone() {
            ui.colored_label(egui::Color32::from_rgb(200, 60, 60), error);
        }

        ui.add_space(8.0);
        ui.group(|ui| {
            ui.label(RichText::new("Discovered Sources").strong());
            if self.data_structure.discovered_sources.is_empty() {
                ui.label("No sources scanned yet.");
            } else {
                egui::ScrollArea::vertical()
                    .max_height(150.0)
                    .show(ui, |ui| {
                        for (index, source) in
                            self.data_structure.discovered_sources.iter().enumerate()
                        {
                            let selected = self.data_structure.selected_source_idx == index;
                            let label = format!(
                                "{} [{}]",
                                source.path.display(),
                                source_kind_label(source.detected_kind)
                            );
                            if ui.selectable_label(selected, label).clicked() {
                                self.data_structure.selected_source_idx = index;
                            }
                        }
                    });
            }
        });

        ui.add_space(8.0);
        self.draw_data_structure_metadata_editor(ui);

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.data_structure.discovered_sources.is_empty()
                        && !self.data_structure.metadata_drafts.is_empty(),
                    egui::Button::new("Start Import"),
                )
                .clicked()
            {
                self.start_data_structure_import_job();
            }
            if ui
                .add_enabled(
                    self.data_structure.imported_experiment_path.is_some(),
                    egui::Button::new("Open Imported Experiment"),
                )
                .clicked()
            {
                if let Some(path) = self.data_structure.imported_experiment_path.clone() {
                    if let Err(err) = self.open_path(path) {
                        self.last_error = Some(err.to_string());
                    }
                }
            }
        });
    }

    fn draw_data_structure_metadata_editor(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Metadata").strong());
            let Some(metadata) = self
                .data_structure
                .metadata_drafts
                .get_mut(self.data_structure.selected_source_idx)
            else {
                ui.label("Probe metadata to edit import settings.");
                return;
            };

            ui.horizontal(|ui| {
                ui.label("Image name");
                ui.text_edit_singleline(&mut metadata.image_name);
            });
            ui.horizontal(|ui| {
                ui.label("Lens NA");
                ui.add(egui::DragValue::new(&mut metadata.lens_na).speed(0.05));
                ui.label("SizeT");
                ui.add(egui::DragValue::new(&mut metadata.size_t).range(1..=100000));
                ui.label("SizeZ");
                ui.add(egui::DragValue::new(&mut metadata.size_z).range(1..=100000));
                ui.label("SizeC");
                ui.add(egui::DragValue::new(&mut metadata.size_c).range(1..=256));
                ui.label("SizeS");
                ui.add(egui::DragValue::new(&mut metadata.size_s).range(1..=100000));
            });
            ui.horizontal(|ui| {
                ui.label("Time increment");
                ui.add(egui::DragValue::new(&mut metadata.time_increment).speed(0.1));
                ui.text_edit_singleline(&mut metadata.time_increment_unit);
            });
            ui.horizontal(|ui| {
                ui.label("Physical sizes");
                ui.add(egui::DragValue::new(&mut metadata.physical_size_x).speed(0.1));
                ui.add(egui::DragValue::new(&mut metadata.physical_size_y).speed(0.1));
                ui.add(egui::DragValue::new(&mut metadata.physical_size_z).speed(0.1));
                ui.text_edit_singleline(&mut metadata.physical_size_unit);
            });
            ui.separator();
            ui.label("Channels to save");
            ensure_bool_len(
                &mut self.data_structure.save_channels,
                metadata.channel_names.len(),
                true,
            );
            for (index, channel_name) in metadata.channel_names.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    if let Some(save) = self.data_structure.save_channels.get_mut(index) {
                        ui.checkbox(save, "");
                    }
                    ui.text_edit_singleline(channel_name);
                    if metadata.emission_wavelengths.len() <= index {
                        metadata.emission_wavelengths.resize(index + 1, 0.0);
                    }
                    ui.label("Emission");
                    ui.add(
                        egui::DragValue::new(&mut metadata.emission_wavelengths[index]).speed(1.0),
                    );
                });
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Time range");
                ui.text_edit_singleline(&mut self.data_structure.time_range_start);
                ui.label("to");
                ui.text_edit_singleline(&mut self.data_structure.time_range_end);
            });
            ui.label("Leave the time range empty to save every frame.");

            let positions = (0..metadata.size_s.max(1))
                .map(|index| format!("Position_{}", index + 1))
                .collect::<Vec<_>>();
            if positions.len() > 1 {
                ui.separator();
                ui.label("Positions to save");
                let mut select_all = self
                    .data_structure
                    .selected_positions
                    .iter()
                    .any(|value| value == "All Positions");
                if ui.checkbox(&mut select_all, "All Positions").changed() {
                    if select_all {
                        self.data_structure.selected_positions = vec!["All Positions".to_string()];
                    } else {
                        self.data_structure.selected_positions.clear();
                    }
                }
                if !self
                    .data_structure
                    .selected_positions
                    .iter()
                    .any(|value| value == "All Positions")
                {
                    for position in positions {
                        let mut selected = self
                            .data_structure
                            .selected_positions
                            .iter()
                            .any(|value| value == &position);
                        if ui.checkbox(&mut selected, &position).changed() {
                            if selected {
                                self.data_structure
                                    .selected_positions
                                    .push(position.clone());
                            } else {
                                self.data_structure
                                    .selected_positions
                                    .retain(|value| value != &position);
                            }
                        }
                    }
                } else {
                    ui.small("All positions are selected.");
                }
            }
        });
    }

    fn draw_data_structure_preview(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Sample Preview").strong());
            let Some(sample) = self.data_structure.sample_planes.as_ref() else {
                ui.label("Load sample preview to inspect a source channel.");
                return;
            };
            self.data_structure.preview_frame_index = self
                .data_structure
                .preview_frame_index
                .min(sample.frames.saturating_sub(1));
            self.data_structure.preview_z_index = self
                .data_structure
                .preview_z_index
                .min(sample.size_z.saturating_sub(1));
            if sample.frames > 1 {
                ui.add(
                    egui::Slider::new(
                        &mut self.data_structure.preview_frame_index,
                        0..=sample.frames.saturating_sub(1),
                    )
                    .text("Frame"),
                );
            }
            if sample.size_z > 1 {
                ui.add(
                    egui::Slider::new(
                        &mut self.data_structure.preview_z_index,
                        0..=sample.size_z.saturating_sub(1),
                    )
                    .text("Z"),
                );
            }
            let image = render_sample_preview(
                sample,
                self.data_structure.preview_frame_index,
                self.data_structure.preview_z_index,
            );
            let texture = ctx.load_texture("data_structure_preview", image, TextureOptions::LINEAR);
            let available = ui.available_size();
            let image_size = texture.size_vec2();
            let scale = (available.x / image_size.x)
                .min((available.y.max(1.0)) / image_size.y)
                .max(1.0);
            let desired = image_size * scale.min(3.0);
            ui.image((texture.id(), desired));
            ui.small(format!(
                "{} x {} px | T={} | Z={}",
                sample.width, sample.height, sample.frames, sample.size_z
            ));
        });
    }

    fn sync_data_structure_selection_state(&mut self) {
        let Some(metadata) = self
            .data_structure
            .metadata_drafts
            .get(self.data_structure.selected_source_idx)
        else {
            return;
        };
        ensure_bool_len(
            &mut self.data_structure.save_channels,
            metadata.channel_names.len(),
            true,
        );
        self.data_structure.selected_positions = if metadata.size_s > 1 {
            vec!["All Positions".to_string()]
        } else {
            vec!["Position_1".to_string()]
        };
    }
}

fn source_kind_label(kind: cellacdc_rs::ImportSourceKind) -> &'static str {
    match kind {
        cellacdc_rs::ImportSourceKind::Npz => "NPZ",
        cellacdc_rs::ImportSourceKind::H5 => "H5",
        cellacdc_rs::ImportSourceKind::Tiff => "TIFF",
        cellacdc_rs::ImportSourceKind::VendorMicroscopy => "Vendor",
    }
}

fn import_layout_label(layout: ImportLayoutKind) -> &'static str {
    match layout {
        ImportLayoutKind::SingleFileMultiPosition => {
            "Single microscopy file with one or more positions"
        }
        ImportLayoutKind::FilePerPosition => "One or more microscopy files, one file per position",
        ImportLayoutKind::FilePerChannel => "One or more microscopy files, one file per channel",
        ImportLayoutKind::CustomMapping => "None of the above",
    }
}

fn import_backend_label(backend: ImportReaderBackend) -> &'static str {
    match backend {
        ImportReaderBackend::Auto => "Auto",
        ImportReaderBackend::Native => "Native",
        ImportReaderBackend::BioFormatsJvmBridge => "Bio-Formats",
    }
}

fn import_conflict_label(mode: ImportConflictMode) -> &'static str {
    match mode {
        ImportConflictMode::OverwritePositionFiles => "Overwrite position files",
        ImportConflictMode::AddFilesToExistingExperiment => "Add files to existing experiment",
        ImportConflictMode::CreateNewPositions => "Create new positions",
    }
}

fn metadata_policy_label(policy: MetadataReusePolicy) -> &'static str {
    match policy {
        MetadataReusePolicy::ConfirmEverySource => "Confirm every source",
        MetadataReusePolicy::UseForRemainingSources => "Use for remaining sources",
        MetadataReusePolicy::TrustReaderForRemainingSources => "Trust reader for remaining sources",
    }
}

fn import_output_label(format: ImportOutputFormat) -> &'static str {
    match format {
        ImportOutputFormat::Tiff => "TIFF",
        ImportOutputFormat::H5 => "H5",
    }
}

fn ensure_bool_len(values: &mut Vec<bool>, len: usize, fill: bool) {
    if values.len() < len {
        values.resize(len, fill);
    } else if values.len() > len {
        values.truncate(len);
    }
}

fn render_sample_preview(
    sample: &cellacdc_rs::ImportSamplePlaneSet,
    frame_index: usize,
    z_index: usize,
) -> ColorImage {
    let plane_len = sample.width * sample.height;
    let planes_per_frame = sample.size_z.max(1);
    let plane_index = frame_index * planes_per_frame + z_index.min(planes_per_frame - 1);
    let offset = plane_index * plane_len;
    let plane = &sample.pixels[offset..offset + plane_len];
    let min_value = plane
        .iter()
        .fold(f32::INFINITY, |left, right| left.min(*right));
    let max_value = plane
        .iter()
        .fold(f32::NEG_INFINITY, |left, right| left.max(*right));
    let denom = (max_value - min_value).max(1e-6);
    let mut rgba = Vec::with_capacity(plane_len * 4);
    for value in plane {
        let normalized = (((*value - min_value) / denom) * 255.0).clamp(0.0, 255.0) as u8;
        rgba.extend_from_slice(&[normalized, normalized, normalized, 255]);
    }
    ColorImage::from_rgba_unmultiplied([sample.width, sample.height], &rgba)
}
