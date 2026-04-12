use crate::gui::app::CellAcdcGui;
use crate::gui::jobs::{JobRequest, JobSummary, JobUpdate};
use crate::gui::state::{AppRoute, DataPrepInteractionMode, ProjectionMode};
use cellacdc_rs::{
    apply_alignment, apply_segm_info_edit, compute_alignment_shifts, preview_crop,
    remove_freehand_roi_npz, save_crop_roi_coords_csv, save_cropped_data, save_segm_info,
    write_background_roi_json, write_freehand_roi_npz, AlignmentRunConfig, BackgroundRoiRect,
    CropRoiCoordsTable, CropRoiRect, CropSaveConfig, FrameProjection, FreehandRoiMask,
    SegmInfoEdit, SegmInfoInterpolationMode, ViewPlane, ZProjectionMode,
};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, StrokeKind, TextureOptions};

use super::{draw_logs, draw_workspace_header, viewer::image_pixel_from_pointer};

impl CellAcdcGui {
    pub(crate) fn draw_data_prep_panel(&mut self, ctx: &eframe::egui::Context) {
        if self
            .selected_position()
            .map(|position| {
                self.data_prep
                    .last_loaded_position
                    .as_ref()
                    != Some(&position.spec.position_dir)
            })
            .unwrap_or(false)
        {
            self.reload_data_prep_state();
        }

        let mut back_to_launcher = false;
        let mut open_session = false;
        egui::TopBottomPanel::top("data_prep_header").show(ctx, |ui| {
            let (back_requested, open_requested) = draw_workspace_header(
                ui,
                AppRoute::DataPrep,
                Some("Alignment, segmInfo editing, crop/background ROI authoring, and Python-compatible Data Prep sidecars."),
                self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
                self.experiment.is_none(),
            );
            back_to_launcher = back_requested;
            open_session = open_requested;
            if let Some(error) = self.last_error.as_deref() {
                ui.colored_label(Color32::from_rgb(220, 90, 90), error);
            }
        });
        if back_to_launcher {
            self.set_route(AppRoute::Launcher);
            return;
        }
        if open_session {
            self.pick_and_open_session();
            return;
        }

        if self.experiment.is_none() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a structured Cell-ACDC position or experiment to use Data Prep.");
                });
            });
            return;
        }

        self.draw_data_prep_controls(ctx);
        self.draw_data_prep_canvas(ctx);
    }

    fn draw_data_prep_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("data_prep_controls")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                let Some(position) = self.selected_position().cloned() else {
                    return;
                };
                ui.heading("Data Prep");
                ui.monospace(position.spec.position_dir.display().to_string());
                ui.separator();

                egui::ComboBox::from_label("Channel")
                    .selected_text(self.data_prep.active_channel.clone())
                    .show_ui(ui, |ui| {
                        for channel in position.channel_names() {
                            if ui
                                .selectable_label(self.data_prep.active_channel == channel, &channel)
                                .clicked()
                            {
                                self.data_prep.active_channel = channel;
                            }
                        }
                    });

                if position.spec.size_t > 1 {
                    ui.add(
                        egui::Slider::new(
                            &mut self.selected_frame_idx,
                            0..=position.spec.size_t.saturating_sub(1),
                        )
                        .text("Frame"),
                    );
                }

                if position.spec.size_z > 1 {
                    let mut current_proj = self.current_data_prep_projection_mode();
                    let changed = egui::ComboBox::from_label("Z / Projection")
                        .selected_text(match current_proj {
                            ZProjectionMode::SingleZSlice => "Single z-slice",
                            ZProjectionMode::MaxZProjection => "Max z-projection",
                            ZProjectionMode::MeanZProjection => "Mean z-projection",
                            ZProjectionMode::MedianZProjection => "Median z-projection",
                        })
                        .show_ui(ui, |ui| {
                            let mut changed = false;
                            changed |= ui
                                .selectable_value(
                                    &mut current_proj,
                                    ZProjectionMode::SingleZSlice,
                                    "Single z-slice",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut current_proj,
                                    ZProjectionMode::MaxZProjection,
                                    "Max z-projection",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut current_proj,
                                    ZProjectionMode::MeanZProjection,
                                    "Mean z-projection",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut current_proj,
                                    ZProjectionMode::MedianZProjection,
                                    "Median z-projection",
                                )
                                .changed();
                            changed
                        })
                        .inner
                        .unwrap_or(false);
                    if current_proj == ZProjectionMode::SingleZSlice {
                        ui.add(
                            egui::Slider::new(
                                &mut self.data_prep.z_index,
                                0..=position.spec.size_z.saturating_sub(1),
                            )
                            .text("Z slice"),
                        );
                    }
                    if changed || ui.button("Apply current frame selection").clicked() {
                        if let Err(err) = self.update_current_data_prep_segm_info(current_proj) {
                            self.last_error = Some(err.to_string());
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Apply to future").clicked() {
                            if let Err(err) = self.propagate_data_prep_segm_info(
                                SegmInfoInterpolationMode::ForwardFill,
                            ) {
                                self.last_error = Some(err.to_string());
                            }
                        }
                        if ui.button("Apply to past").clicked() {
                            if let Err(err) = self.propagate_data_prep_segm_info(
                                SegmInfoInterpolationMode::BackwardFill,
                            ) {
                                self.last_error = Some(err.to_string());
                            }
                        }
                    });
                    if ui.button("Interpolate between frames").clicked() {
                        if let Err(err) = self.propagate_data_prep_segm_info(
                            SegmInfoInterpolationMode::LinearFrames,
                        ) {
                            self.last_error = Some(err.to_string());
                        }
                    }
                }

                ui.separator();
                ui.label(RichText::new("Edit Mode").strong());
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(
                        &mut self.data_prep.interaction_mode,
                        DataPrepInteractionMode::AddCropRoi,
                        "Add crop ROI",
                    );
                    ui.selectable_value(
                        &mut self.data_prep.interaction_mode,
                        DataPrepInteractionMode::AddBackgroundRoi,
                        "Add background ROI",
                    );
                    ui.selectable_value(
                        &mut self.data_prep.interaction_mode,
                        DataPrepInteractionMode::DrawFreeRoi,
                        "Draw free-hand ROI",
                    );
                    if ui.button("Stop").clicked() {
                        self.data_prep.interaction_mode = DataPrepInteractionMode::None;
                        self.data_prep.drag_start = None;
                        self.data_prep.drag_current = None;
                    }
                });

                if self.data_prep.interaction_mode == DataPrepInteractionMode::DrawFreeRoi {
                    ui.horizontal(|ui| {
                        if ui.button("Close ROI").clicked() {
                            if let Err(err) = self.finalize_free_roi() {
                                self.last_error = Some(err.to_string());
                            }
                        }
                        if ui.button("Clear points").clicked() {
                            self.data_prep.free_roi_points.clear();
                            self.data_prep.free_roi = None;
                            if let Err(err) = self.save_data_prep_free_roi() {
                                self.last_error = Some(err.to_string());
                            }
                        }
                    });
                    ui.small("Left-click to add polygon points. Use Close ROI to persist the mask.");
                }

                ui.separator();
                ui.label(RichText::new("Crop Range").strong());
                self.draw_data_prep_range_controls(ui, position.spec.size_t, position.spec.size_z);
                if ui.button("Preview crop").clicked() {
                    match self.current_crop_save_config() {
                        Ok(config) => match preview_crop(&config) {
                            Ok(preview) => self.data_prep.pending_crop_preview = Some(preview),
                            Err(err) => self.last_error = Some(err.to_string()),
                        },
                        Err(err) => self.last_error = Some(err.to_string()),
                    }
                }
                if ui.button("Save cropped data").clicked() {
                    match self.current_crop_save_config() {
                        Ok(config) => {
                            self.start_job(
                                JobRequest {
                                    label: "Save cropped data".to_string(),
                                },
                                move |sender, token| {
                                    if token.is_cancelled() {
                                        anyhow::bail!("Crop save cancelled before start");
                                    }
                                    let _ = sender.send(JobUpdate::Log(
                                        "Saving cropped data and updating Data Prep sidecars"
                                            .to_string(),
                                    ));
                                    let result = save_cropped_data(config)?;
                                    Ok(JobSummary {
                                        summary: format!(
                                            "Saved cropped data ({} output paths)",
                                            result.written_files.len()
                                        ),
                                        reload_session: true,
                                        select_segmentation_endname: None,
                                    })
                                },
                            );
                        }
                        Err(err) => self.last_error = Some(err.to_string()),
                    }
                }
                if ui.button("Reset crop preview").clicked() {
                    self.data_prep.pending_crop_preview = None;
                }

                ui.separator();
                if ui.button("Start process").clicked() {
                    let config = AlignmentRunConfig {
                        position_dir: position.spec.position_dir.clone(),
                        reference_channel: self.data_prep.active_channel.clone(),
                        channels_to_align: position.channel_names(),
                        frame_range: self.data_prep.frame_range,
                        overwrite: true,
                    };
                    self.start_job(
                        JobRequest {
                            label: format!("Align {}", position.spec.position_dir.display()),
                        },
                        move |sender, token| {
                            if token.is_cancelled() {
                                anyhow::bail!("Alignment cancelled before start");
                            }
                            let _ = sender.send(JobUpdate::Log(
                                "Computing alignment shifts and writing aligned channels"
                                    .to_string(),
                            ));
                            let shifts = compute_alignment_shifts(&config)?;
                            let result = apply_alignment(config, &shifts)?;
                            Ok(JobSummary {
                                summary: format!(
                                    "Alignment complete -> {} output(s)",
                                    result.aligned_outputs.len()
                                ),
                                reload_session: true,
                                select_segmentation_endname: None,
                            })
                        },
                    );
                }

                ui.separator();
                ui.collapsing("Crop ROIs", |ui| {
                    let mut remove = None;
                    for (idx, roi) in self.data_prep.crop_rois.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "#{} x={} y={} w={} h={}",
                                roi.roi_id, roi.x, roi.y, roi.width, roi.height
                            ));
                            if ui.button("Remove").clicked() {
                                remove = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove {
                        self.data_prep.crop_rois.remove(idx);
                        if let Err(err) = self.save_data_prep_crop_rois() {
                            self.last_error = Some(err.to_string());
                        }
                    }
                });
                ui.collapsing("Background ROIs", |ui| {
                    let mut remove = None;
                    for (idx, roi) in self.data_prep.background_rois.items.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "#{} x={} y={} w={} h={}",
                                idx,
                                roi.pos[0] as usize,
                                roi.pos[1] as usize,
                                roi.size[0] as usize,
                                roi.size[1] as usize
                            ));
                            if ui.button("Remove").clicked() {
                                remove = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove {
                        self.data_prep.background_rois.items.remove(idx);
                        if let Err(err) = self.save_data_prep_background_rois() {
                            self.last_error = Some(err.to_string());
                        }
                    }
                });
                if let Some(preview) = self.data_prep.pending_crop_preview.as_ref() {
                    ui.separator();
                    ui.label(RichText::new("Crop Preview").strong());
                    for (idx, shape) in preview.output_shapes.iter().enumerate() {
                        ui.monospace(format!("ROI {idx}: {shape:?}"));
                    }
                }

                ui.separator();
                draw_logs(ui, &self.logs, 160.0);
            });
    }

    fn draw_data_prep_range_controls(&mut self, ui: &mut egui::Ui, size_t: usize, size_z: usize) {
        let mut enable_frame_range = self.data_prep.frame_range.is_some();
        ui.checkbox(&mut enable_frame_range, "Crop time range");
        if enable_frame_range {
            let mut start = self.data_prep.frame_range.map(|range| range.0).unwrap_or(0);
            let mut end = self
                .data_prep
                .frame_range
                .map(|range| range.1)
                .unwrap_or(size_t.max(1));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut start).range(0..=size_t.saturating_sub(1)));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut end).range(1..=size_t.max(1)));
            });
            if end <= start {
                end = (start + 1).min(size_t.max(1));
            }
            self.data_prep.frame_range = Some((start, end));
        } else {
            self.data_prep.frame_range = None;
        }

        if size_z > 1 {
            let mut enable_z_range = self.data_prep.z_range.is_some();
            ui.checkbox(&mut enable_z_range, "Crop z-slices");
            if enable_z_range {
                let mut start = self.data_prep.z_range.map(|range| range.0).unwrap_or(0);
                let mut end = self
                    .data_prep
                    .z_range
                    .map(|range| range.1)
                    .unwrap_or(size_z.max(1));
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut start).range(0..=size_z.saturating_sub(1)));
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut end).range(1..=size_z.max(1)));
                });
                if end <= start {
                    end = (start + 1).min(size_z.max(1));
                }
                self.data_prep.z_range = Some((start, end));
            } else {
                self.data_prep.z_range = None;
            }
        } else {
            self.data_prep.z_range = None;
        }
    }

    fn draw_data_prep_canvas(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(position) = self.selected_position().cloned() else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a position to start Data Prep.");
                });
                return;
            };

            match self.render_data_prep_image() {
                Ok(image) => {
                    let texture = ctx.load_texture("data_prep_view", image, TextureOptions::LINEAR);
                    let available = ui.available_size();
                    let image_size = texture.size_vec2();
                    let scale = (available.x / image_size.x)
                        .min(available.y / image_size.y)
                        .max(0.1);
                    let desired = image_size * scale;
                    let response = ui.add(egui::Image::new((texture.id(), desired)).sense(Sense::click_and_drag()));
                    self.handle_data_prep_canvas_interaction(
                        &response,
                        [texture.size()[0], texture.size()[1]],
                    );
                    self.paint_data_prep_overlays(ui, response.rect, texture.size());
                    ui.separator();
                    ui.label(format!(
                        "Frame {}  Channel {}  SizeT={} SizeZ={}",
                        self.selected_frame_idx,
                        self.data_prep.active_channel,
                        position.spec.size_t,
                        position.spec.size_z
                    ));
                }
                Err(err) => {
                    ui.colored_label(Color32::from_rgb(220, 90, 90), err.to_string());
                }
            }
        });
    }

    fn render_data_prep_image(&self) -> anyhow::Result<ColorImage> {
        let position = self
            .selected_position()
            .ok_or_else(|| anyhow::anyhow!("No position selected"))?;
        let projection = self.data_prep_projection();
        let frame = position.load_channel_frame_for_view(
            &self.data_prep.active_channel,
            self.selected_frame_idx,
            ViewPlane::XY,
            projection,
        )?;
        let (min_value, max_value) = frame
            .pixels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min_v, max_v), value| {
                (min_v.min(*value), max_v.max(*value))
            });
        let denom = if (max_value - min_value).abs() < f32::EPSILON {
            1.0
        } else {
            max_value - min_value
        };
        let mut rgba = Vec::with_capacity(frame.pixels.len() * 4);
        for value in &frame.pixels {
            let normalized = ((*value - min_value) / denom * 255.0).clamp(0.0, 255.0) as u8;
            rgba.extend_from_slice(&[normalized, normalized, normalized, 255]);
        }
        Ok(ColorImage::from_rgba_unmultiplied(
            [frame.width, frame.height],
            &rgba,
        ))
    }

    fn data_prep_projection(&self) -> FrameProjection {
        match self.current_data_prep_projection_mode() {
            ZProjectionMode::SingleZSlice => FrameProjection::ZSlice(self.data_prep.z_index),
            ZProjectionMode::MaxZProjection
            | ZProjectionMode::MeanZProjection
            | ZProjectionMode::MedianZProjection => FrameProjection::Max,
        }
    }

    fn current_data_prep_projection_mode(&self) -> ZProjectionMode {
        let Some(position) = self.selected_position() else {
            return ZProjectionMode::SingleZSlice;
        };
        let Some(channel) = position
            .spec
            .channels
            .iter()
            .find(|channel| channel.name == self.data_prep.active_channel)
        else {
            return if self.data_prep.projection_mode == ProjectionMode::ZSlice {
                ZProjectionMode::SingleZSlice
            } else {
                ZProjectionMode::MaxZProjection
            };
        };
        let Some(filename) = channel.image_path.file_name().and_then(|name| name.to_str()) else {
            return ZProjectionMode::SingleZSlice;
        };
        self.data_prep
            .segm_info
            .get(filename, self.selected_frame_idx)
            .map(|record| record.which_z_proj)
            .unwrap_or_else(|| {
                if self.data_prep.projection_mode == ProjectionMode::ZSlice {
                    ZProjectionMode::SingleZSlice
                } else {
                    ZProjectionMode::MaxZProjection
                }
            })
    }

    fn update_current_data_prep_segm_info(
        &mut self,
        projection_mode: ZProjectionMode,
    ) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        if position.spec.size_z <= 1 {
            return Ok(());
        }
        let channel = position
            .spec
            .channels
            .iter()
            .find(|channel| channel.name == self.data_prep.active_channel)
            .ok_or_else(|| anyhow::anyhow!("Unknown channel {:?}", self.data_prep.active_channel))?;
        let filename = channel
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid channel path {}", channel.image_path.display()))?
            .to_string();
        self.data_prep.segm_info = apply_segm_info_edit(
            &self.data_prep.segm_info,
            SegmInfoEdit {
                filename,
                frame_i: self.selected_frame_idx,
                z_slice_used_data_prep: Some(self.data_prep.z_index),
                which_z_proj: Some(projection_mode),
                crop_lower_z_slice: self
                    .data_prep
                    .z_range
                    .map(|range| range.0),
                crop_upper_z_slice: self
                    .data_prep
                    .z_range
                    .map(|range| range.1.saturating_sub(1)),
            },
        )?;
        self.save_current_data_prep_segm_info()
    }

    fn propagate_data_prep_segm_info(
        &mut self,
        mode: SegmInfoInterpolationMode,
    ) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        let channel = position
            .spec
            .channels
            .iter()
            .find(|channel| channel.name == self.data_prep.active_channel)
            .ok_or_else(|| anyhow::anyhow!("Unknown channel {:?}", self.data_prep.active_channel))?;
        let filename = channel
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid channel path {}", channel.image_path.display()))?;
        self.data_prep.segm_info = cellacdc_rs::propagate_segm_info_selection(
            &self.data_prep.segm_info,
            filename,
            self.selected_frame_idx,
            mode,
        )?;
        self.save_current_data_prep_segm_info()
    }

    fn save_current_data_prep_segm_info(&mut self) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        let path = position
            .spec
            .images_dir
            .join(format!("{}segmInfo.csv", position.spec.basename));
        save_segm_info(&path, &self.data_prep.segm_info)?;
        self.append_log(format!("Saved segmInfo -> {}", path.display()));
        Ok(())
    }

    fn current_crop_save_config(&self) -> anyhow::Result<CropSaveConfig> {
        let position = self
            .selected_position()
            .ok_or_else(|| anyhow::anyhow!("No position selected"))?;
        Ok(CropSaveConfig {
            position_dir: position.spec.position_dir.clone(),
            channels: position.channel_names(),
            frame_range: self.data_prep.frame_range,
            z_range: self.data_prep.z_range,
            crop_rois: self.data_prep.crop_rois.clone(),
            background_rois: self.data_prep.background_rois.clone(),
            free_roi: self.data_prep.free_roi.clone(),
            overwrite: true,
        })
    }

    fn save_data_prep_crop_rois(&mut self) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        let path = position
            .spec
            .images_dir
            .join(format!("{}dataPrepROIs_coords.csv", position.spec.basename));
        let table = CropRoiCoordsTable {
            rois: self.data_prep.crop_rois.clone(),
            cropped_roi_ids: Vec::new(),
        };
        save_crop_roi_coords_csv(&path, &table)?;
        self.append_log(format!("Saved crop ROIs -> {}", path.display()));
        Ok(())
    }

    fn save_data_prep_background_rois(&mut self) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        let path = position
            .spec
            .images_dir
            .join(format!("{}dataPrep_bkgrROIs.json", position.spec.basename));
        write_background_roi_json(&path, &self.data_prep.background_rois)?;
        self.append_log(format!("Saved background ROIs -> {}", path.display()));
        Ok(())
    }

    fn save_data_prep_free_roi(&mut self) -> anyhow::Result<()> {
        let Some(position) = self.selected_position() else {
            anyhow::bail!("No position selected");
        };
        let path = position
            .spec
            .images_dir
            .join(format!("{}dataPrepFreeRoi.npz", position.spec.basename));
        if let Some(free_roi) = self.data_prep.free_roi.as_ref() {
            write_freehand_roi_npz(&path, free_roi)?;
            self.append_log(format!("Saved free ROI -> {}", path.display()));
        } else {
            remove_freehand_roi_npz(&path)?;
            self.append_log(format!("Removed free ROI -> {}", path.display()));
        }
        Ok(())
    }

    fn finalize_free_roi(&mut self) -> anyhow::Result<()> {
        if self.data_prep.free_roi_points.len() < 3 {
            anyhow::bail!("Free-hand ROI requires at least three points");
        }
        let roi = polygon_points_to_free_roi(&self.data_prep.free_roi_points)?;
        self.data_prep.free_roi = Some(roi);
        self.save_data_prep_free_roi()?;
        self.data_prep.interaction_mode = DataPrepInteractionMode::None;
        Ok(())
    }

    fn handle_data_prep_canvas_interaction(
        &mut self,
        response: &egui::Response,
        image_size: [usize; 2],
    ) {
        let Some(pointer) = response.interact_pointer_pos() else {
            return;
        };
        let Some((x, y)) = image_pixel_from_pointer(response.rect, pointer, image_size) else {
            return;
        };
        self.status_text = format!(
            "Data Prep  frame={} x={} y={} channel={}",
            self.selected_frame_idx, x, y, self.data_prep.active_channel
        );

        if response.secondary_clicked() {
            if self.remove_data_prep_roi_at(x, y).is_ok() {
                return;
            }
        }

        match self.data_prep.interaction_mode {
            DataPrepInteractionMode::AddCropRoi | DataPrepInteractionMode::AddBackgroundRoi => {
                if response.drag_started() {
                    self.data_prep.drag_start = Some((x, y));
                    self.data_prep.drag_current = Some((x, y));
                } else if response.dragged() {
                    self.data_prep.drag_current = Some((x, y));
                } else if response.drag_stopped() {
                    self.data_prep.drag_current = Some((x, y));
                    if let Err(err) = self.finalize_data_prep_drag_roi() {
                        self.last_error = Some(err.to_string());
                    }
                }
            }
            DataPrepInteractionMode::DrawFreeRoi => {
                if response.clicked() {
                    self.data_prep.free_roi_points.push((x, y));
                }
            }
            DataPrepInteractionMode::None => {}
        }
    }

    fn finalize_data_prep_drag_roi(&mut self) -> anyhow::Result<()> {
        let Some((x0, y0)) = self.data_prep.drag_start.take() else {
            return Ok(());
        };
        let Some((x1, y1)) = self.data_prep.drag_current.take() else {
            return Ok(());
        };
        let x = x0.min(x1);
        let y = y0.min(y1);
        let width = x0.max(x1).saturating_sub(x) + 1;
        let height = y0.max(y1).saturating_sub(y) + 1;
        if width == 0 || height == 0 {
            self.data_prep.interaction_mode = DataPrepInteractionMode::None;
            return Ok(());
        }
        match self.data_prep.interaction_mode {
            DataPrepInteractionMode::AddCropRoi => {
                let next_id = self
                    .data_prep
                    .crop_rois
                    .iter()
                    .map(|roi| roi.roi_id)
                    .max()
                    .unwrap_or(0)
                    + usize::from(!self.data_prep.crop_rois.is_empty());
                self.data_prep.crop_rois.push(CropRoiRect {
                    roi_id: next_id,
                    x,
                    y,
                    width,
                    height,
                });
                self.save_data_prep_crop_rois()?;
            }
            DataPrepInteractionMode::AddBackgroundRoi => {
                self.data_prep.background_rois.items.push(BackgroundRoiRect {
                    pos: [x as f32, y as f32],
                    size: [width as f32, height as f32],
                });
                self.save_data_prep_background_rois()?;
            }
            DataPrepInteractionMode::DrawFreeRoi | DataPrepInteractionMode::None => {}
        }
        self.data_prep.interaction_mode = DataPrepInteractionMode::None;
        Ok(())
    }

    fn remove_data_prep_roi_at(&mut self, x: usize, y: usize) -> anyhow::Result<()> {
        if let Some(idx) = self
            .data_prep
            .crop_rois
            .iter()
            .position(|roi| point_in_crop_roi(x, y, roi))
        {
            self.data_prep.crop_rois.remove(idx);
            self.save_data_prep_crop_rois()?;
            return Ok(());
        }
        if let Some(idx) = self
            .data_prep
            .background_rois
            .items
            .iter()
            .position(|roi| point_in_background_roi(x, y, roi))
        {
            self.data_prep.background_rois.items.remove(idx);
            self.save_data_prep_background_rois()?;
        }
        Ok(())
    }

    fn paint_data_prep_overlays(
        &self,
        ui: &egui::Ui,
        rect: Rect,
        image_size: [usize; 2],
    ) {
        let painter = ui.painter_at(rect);
        for roi in &self.data_prep.crop_rois {
            paint_rect_roi(
                &painter,
                rect,
                image_size,
                roi.x as f32,
                roi.y as f32,
                roi.width as f32,
                roi.height as f32,
                Color32::from_rgb(240, 190, 40),
            );
        }
        for roi in &self.data_prep.background_rois.items {
            paint_rect_roi(
                &painter,
                rect,
                image_size,
                roi.pos[0],
                roi.pos[1],
                roi.size[0],
                roi.size[1],
                Color32::from_rgb(80, 200, 250),
            );
        }
        if let (Some((x0, y0)), Some((x1, y1))) =
            (self.data_prep.drag_start, self.data_prep.drag_current)
        {
            paint_rect_roi(
                &painter,
                rect,
                image_size,
                x0.min(x1) as f32,
                y0.min(y1) as f32,
                x0.max(x1).saturating_sub(x0.min(x1)) as f32 + 1.0,
                y0.max(y1).saturating_sub(y0.min(y1)) as f32 + 1.0,
                Color32::WHITE,
            );
        }
        if !self.data_prep.free_roi_points.is_empty() {
            paint_polyline(
                &painter,
                rect,
                image_size,
                &self.data_prep.free_roi_points,
                Color32::from_rgb(240, 100, 180),
            );
        } else if let Some(roi) = self.data_prep.free_roi.as_ref() {
            let (y0, x0, y1, x1) = roi.bbox_yxxy;
            paint_rect_roi(
                &painter,
                rect,
                image_size,
                x0 as f32,
                y0 as f32,
                (x1.saturating_sub(x0) + 1) as f32,
                (y1.saturating_sub(y0) + 1) as f32,
                Color32::from_rgb(240, 100, 180),
            );
        }
    }
}

fn point_in_crop_roi(x: usize, y: usize, roi: &CropRoiRect) -> bool {
    x >= roi.x && x < roi.x + roi.width && y >= roi.y && y < roi.y + roi.height
}

fn point_in_background_roi(x: usize, y: usize, roi: &BackgroundRoiRect) -> bool {
    let x0 = roi.pos[0].round().max(0.0) as usize;
    let y0 = roi.pos[1].round().max(0.0) as usize;
    let width = roi.size[0].round().max(0.0) as usize;
    let height = roi.size[1].round().max(0.0) as usize;
    x >= x0 && x < x0 + width && y >= y0 && y < y0 + height
}

fn paint_rect_roi(
    painter: &egui::Painter,
    rect: Rect,
    image_size: [usize; 2],
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: Color32,
) {
    if image_size[0] == 0 || image_size[1] == 0 {
        return;
    }
    let min = Pos2::new(
        rect.min.x + rect.width() * (x / image_size[0] as f32),
        rect.min.y + rect.height() * (y / image_size[1] as f32),
    );
    let max = Pos2::new(
        rect.min.x + rect.width() * ((x + width) / image_size[0] as f32),
        rect.min.y + rect.height() * ((y + height) / image_size[1] as f32),
    );
    painter.rect_stroke(
        Rect::from_min_max(min, max),
        0.0,
        egui::Stroke::new(2.0, color),
        StrokeKind::Outside,
    );
}

fn paint_polyline(
    painter: &egui::Painter,
    rect: Rect,
    image_size: [usize; 2],
    points: &[(usize, usize)],
    color: Color32,
) {
    if points.len() < 2 || image_size[0] == 0 || image_size[1] == 0 {
        return;
    }
    let screen_points = points
        .iter()
        .map(|(x, y)| {
            Pos2::new(
                rect.min.x + rect.width() * (*x as f32 / image_size[0] as f32),
                rect.min.y + rect.height() * (*y as f32 / image_size[1] as f32),
            )
        })
        .collect::<Vec<_>>();
    painter.add(egui::Shape::line(
        screen_points,
        egui::Stroke::new(2.0, color),
    ));
}

fn polygon_points_to_free_roi(points: &[(usize, usize)]) -> anyhow::Result<FreehandRoiMask> {
    let x0 = points.iter().map(|(x, _)| *x).min().unwrap_or(0);
    let x1 = points.iter().map(|(x, _)| *x).max().unwrap_or(0);
    let y0 = points.iter().map(|(_, y)| *y).min().unwrap_or(0);
    let y1 = points.iter().map(|(_, y)| *y).max().unwrap_or(0);
    let width = x1.saturating_sub(x0) + 1;
    let height = y1.saturating_sub(y0) + 1;
    if width == 0 || height == 0 {
        anyhow::bail!("Free-hand ROI has zero width or height");
    }
    let translated = points
        .iter()
        .map(|(x, y)| (*x as f32 - x0 as f32, *y as f32 - y0 as f32))
        .collect::<Vec<_>>();
    let mut mask = vec![false; width * height];
    for yy in 0..height {
        for xx in 0..width {
            let inside = point_in_polygon((xx as f32 + 0.5, yy as f32 + 0.5), &translated);
            mask[yy * width + xx] = inside;
        }
    }
    Ok(FreehandRoiMask {
        bbox_yxxy: (y0, x0, y1, x1),
        local_mask: ndarray::Array2::from_shape_vec((height, width), mask)?,
    })
}

fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        let intersect = ((yi > point.1) != (yj > point.1))
            && (point.0
                < (xj - xi) * (point.1 - yi) / ((yj - yi).abs().max(f32::EPSILON)) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}
