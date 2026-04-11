use crate::gui::app::CellAcdcGui;
use anyhow::{anyhow, bail, Result};
use cellacdc_rs::{FrameData, MaskEditCommand, SegmentationLayout};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, TextureOptions};

use super::selected_segm_label;

impl CellAcdcGui {
    pub(crate) fn refresh_texture_if_needed(&mut self, ctx: &egui::Context) {
        let Some(key) = self.current_view_key() else {
            self.texture = None;
            self.texture_key = None;
            return;
        };
        if self.texture_key.as_ref() == Some(&key) {
            return;
        }

        match self.render_current_view() {
            Ok(image) => {
                if let Some(texture) = self.texture.as_mut() {
                    texture.set(image, TextureOptions::LINEAR);
                } else {
                    self.texture =
                        Some(ctx.load_texture("cellacdc_viewer", image, TextureOptions::LINEAR));
                }
                self.texture_key = Some(key);
                self.last_error = None;
            }
            Err(err) => {
                self.texture = None;
                self.texture_key = Some(key);
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn render_current_view(&self) -> Result<ColorImage> {
        let position = self
            .selected_position()
            .ok_or_else(|| anyhow!("No position selected"))?;
        let frame = position.load_channel_frame(
            &self.persisted.selected_channel,
            self.selected_frame_idx,
            self.current_projection(),
        )?;
        let segm = if self.persisted.show_segmentation_overlay {
            self.current_segmentation_frame_data()?
        } else {
            None
        };
        compose_color_image(
            &frame,
            segm.as_ref(),
            self.persisted.overlay_alpha,
            self.current_annotation_label(),
        )
    }

    pub(crate) fn draw_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("positions_panel")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Session");
                if let Some(experiment) = self.experiment.as_ref() {
                    ui.label(format!("Root: {}", experiment.root_path.display()));
                    ui.label(format!("Positions: {}", experiment.positions.len()));
                } else {
                    ui.label("Open a Cell-ACDC experiment or Position_* folder.");
                    if ui.button("Open Session").clicked() {
                        self.pick_and_open_session();
                    }
                    return;
                }

                ui.separator();
                ui.label(egui::RichText::new("Positions").strong());
                let position_entries = self
                    .experiment
                    .as_ref()
                    .map(|experiment| {
                        experiment
                            .positions
                            .iter()
                            .enumerate()
                            .map(|(idx, position)| {
                                let name = position
                                    .spec
                                    .position_dir
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or("Position")
                                    .to_string();
                                (idx, name)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for (idx, name) in position_entries {
                            if ui
                                .selectable_label(self.selected_position_idx == idx, name)
                                .clicked()
                            {
                                self.request_position_selection(idx);
                            }
                        }
                    });

                ui.separator();
                if let Some(position) = self.selected_position().cloned() {
                    ui.label(egui::RichText::new("Position Details").strong());
                    ui.monospace(position.spec.position_dir.display().to_string());
                    ui.label(format!(
                        "SizeT={}  SizeZ={}  Pixel={:.3} x {:.3}",
                        position.spec.size_t,
                        position.spec.size_z,
                        position.spec.physical_size_x,
                        position.spec.physical_size_y
                    ));

                    let channel_names = position.channel_names();
                    egui::ComboBox::from_label("Display channel")
                        .selected_text(self.persisted.selected_channel.clone())
                        .show_ui(ui, |ui| {
                            for name in channel_names {
                                if ui
                                    .selectable_label(
                                        self.persisted.selected_channel == name,
                                        &name,
                                    )
                                    .clicked()
                                {
                                    self.persisted.selected_channel = name;
                                    self.invalidate_texture();
                                }
                            }
                        });

                    egui::ComboBox::from_label("Segmentation overlay")
                        .selected_text(selected_segm_label(
                            &position,
                            &self.persisted.selected_segmentation_endname,
                        ))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(position.segmentations.is_empty(), "<none>")
                                .clicked()
                            {
                                self.request_segmentation_selection(None);
                                self.persisted.show_segmentation_overlay = false;
                                self.invalidate_texture();
                            }
                            for asset in &position.segmentations {
                                let label = asset.name.clone();
                                if ui
                                    .selectable_label(
                                        self.persisted.selected_segmentation_endname
                                            == asset.endname,
                                        &label,
                                    )
                                    .clicked()
                                {
                                    self.request_segmentation_selection(asset.endname.clone());
                                    self.persisted.show_segmentation_overlay = true;
                                    self.invalidate_texture();
                                }
                            }
                        });

                    if ui
                        .checkbox(
                            &mut self.persisted.show_segmentation_overlay,
                            "Show segmentation overlay",
                        )
                        .changed()
                    {
                        self.invalidate_texture();
                    }

                    if ui
                        .add(
                            egui::Slider::new(&mut self.persisted.overlay_alpha, 0.0..=1.0)
                                .text("Overlay alpha"),
                        )
                        .changed()
                    {
                        self.invalidate_texture();
                    }
                }
            });
    }

    pub(crate) fn draw_viewer_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(position) = self.selected_position().cloned() else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an experiment or position to start.");
                });
                return;
            };
            let size_t = position.spec.size_t;
            let size_z = position.spec.size_z;

            ui.horizontal_wrapped(|ui| {
                let frame_max = size_t.saturating_sub(1);
                if size_t > 0 {
                    ui.add(
                        egui::Slider::new(&mut self.selected_frame_idx, 0..=frame_max)
                            .text("Frame"),
                    );
                }
                let projection_changed = egui::ComboBox::from_label("Projection")
                    .selected_text(match self.persisted.projection_mode {
                        crate::gui::state::ProjectionMode::Max => "Max projection",
                        crate::gui::state::ProjectionMode::ZSlice => "Z slice",
                    })
                    .show_ui(ui, |ui| {
                        let mut changed = false;
                        changed |= ui
                            .selectable_value(
                                &mut self.persisted.projection_mode,
                                crate::gui::state::ProjectionMode::Max,
                                "Max projection",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut self.persisted.projection_mode,
                                crate::gui::state::ProjectionMode::ZSlice,
                                "Z slice",
                            )
                            .changed();
                        changed
                    })
                    .inner
                    .unwrap_or(false);

                if self.persisted.projection_mode == crate::gui::state::ProjectionMode::ZSlice
                    && size_z > 0
                {
                    ui.add(
                        egui::Slider::new(
                            &mut self.persisted.z_index,
                            0..=size_z.saturating_sub(1),
                        )
                        .text("Z"),
                    );
                }

                if projection_changed || ui.button("Refresh").clicked() {
                    self.invalidate_texture();
                }
            });

            let frame_index = self.selected_frame_idx;
            let projection = self.current_projection();
            if let Some(document) = self.current_annotation_document_mut() {
                document.session.selection.frame_index = frame_index;
                document.session.selection.z_index = match projection {
                    cellacdc_rs::FrameProjection::Max => None,
                    cellacdc_rs::FrameProjection::ZSlice(z_index) => Some(z_index),
                };
            }

            self.refresh_texture_if_needed(ctx);
            ui.separator();

            if let Some(texture) = &self.texture {
                let available = ui.available_size();
                let image_size = texture.size_vec2();
                let scale = (available.x / image_size.x)
                    .min(available.y / image_size.y)
                    .max(0.1);
                let desired = image_size * scale;
                let sense = if self.persisted.route == crate::gui::state::AppRoute::Annotation {
                    egui::Sense::click_and_drag()
                } else {
                    egui::Sense::hover()
                };
                let response = ui.add(egui::Image::new((texture.id(), desired)).sense(sense));
                if self.persisted.route == crate::gui::state::AppRoute::Annotation {
                    self.handle_annotation_canvas_interaction(
                        &response,
                        [texture.size()[0], texture.size()[1]],
                    );
                }
            } else if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
            } else {
                ui.label("No image available for the current selection.");
            }
        });
    }

    fn handle_annotation_canvas_interaction(
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
        match self.annotation.tool {
            crate::gui::state::AnnotationTool::Select if response.clicked() => {
                if let Err(err) = self.select_annotation_label_at(x, y) {
                    self.last_error = Some(err.to_string());
                }
            }
            crate::gui::state::AnnotationTool::Brush
                if response.clicked() || response.dragged() =>
            {
                if let Err(err) = self.paint_annotation_at(x, y) {
                    self.last_error = Some(err.to_string());
                }
            }
            crate::gui::state::AnnotationTool::Eraser
                if response.clicked() || response.dragged() =>
            {
                if let Err(err) = self.erase_annotation_at(x, y) {
                    self.last_error = Some(err.to_string());
                }
            }
            _ => {}
        }
    }

    fn select_annotation_label_at(&mut self, x: usize, y: usize) -> Result<()> {
        let Some(segmentation) = self.current_segmentation_frame_data()? else {
            return Ok(());
        };
        if x >= segmentation.width || y >= segmentation.height {
            return Ok(());
        }
        let label = segmentation.pixels[y * segmentation.width + x];
        self.annotation_select_label((label != 0).then_some(label))
    }

    fn paint_annotation_at(&mut self, x: usize, y: usize) -> Result<()> {
        if !self.annotation_edits_allowed() {
            bail!("GUI edits require a writable 2D view. Switch to a single z-slice first.");
        }
        let label = self
            .current_annotation_label()
            .ok_or_else(|| anyhow!("Select an object ID before painting"))?;
        let flat_indices = self.annotation_disk_indices(x, y)?;
        self.run_annotation_command(MaskEditCommand::Paint {
            flat_indices,
            label,
        })
    }

    fn erase_annotation_at(&mut self, x: usize, y: usize) -> Result<()> {
        if !self.annotation_edits_allowed() {
            bail!("GUI edits require a writable 2D view. Switch to a single z-slice first.");
        }
        let flat_indices = self.annotation_disk_indices(x, y)?;
        self.run_annotation_command(MaskEditCommand::Erase { flat_indices })
    }

    pub(crate) fn annotation_disk_indices(&self, x: usize, y: usize) -> Result<Vec<usize>> {
        let document = self
            .current_annotation_document()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let shape = document.session.data.values.shape().to_vec();
        let (width, height, plane_offset) = match document.session.data.layout {
            SegmentationLayout::YX => (shape[1], shape[0], 0),
            SegmentationLayout::TYX => {
                let height = shape[1];
                let width = shape[2];
                (width, height, self.selected_frame_idx * width * height)
            }
            SegmentationLayout::ZYX => {
                let z_index = match self.current_projection() {
                    cellacdc_rs::FrameProjection::ZSlice(z_index) => z_index,
                    cellacdc_rs::FrameProjection::Max => {
                        bail!("GUI edits require a single z-slice for 3D segmentations")
                    }
                };
                let height = shape[1];
                let width = shape[2];
                (width, height, z_index * width * height)
            }
            SegmentationLayout::TZYX => {
                let z_index = match self.current_projection() {
                    cellacdc_rs::FrameProjection::ZSlice(z_index) => z_index,
                    cellacdc_rs::FrameProjection::Max => {
                        bail!("GUI edits require a single z-slice for 3D segmentations")
                    }
                };
                let height = shape[2];
                let width = shape[3];
                let plane_len = width * height;
                (
                    width,
                    height,
                    (self.selected_frame_idx * shape[1] + z_index) * plane_len,
                )
            }
        };

        let radius = self.annotation.brush_radius.max(1) as isize;
        let mut flat_indices = Vec::new();
        for y_offset in -radius..=radius {
            for x_offset in -radius..=radius {
                if x_offset * x_offset + y_offset * y_offset > radius * radius {
                    continue;
                }
                let pixel_x = x as isize + x_offset;
                let pixel_y = y as isize + y_offset;
                if pixel_x < 0
                    || pixel_y < 0
                    || pixel_x >= width as isize
                    || pixel_y >= height as isize
                {
                    continue;
                }
                flat_indices.push(plane_offset + pixel_y as usize * width + pixel_x as usize);
            }
        }
        Ok(flat_indices)
    }
}

fn compose_color_image(
    frame: &FrameData<f32>,
    segmentation: Option<&FrameData<u32>>,
    overlay_alpha: f32,
    selected_label: Option<u32>,
) -> Result<ColorImage> {
    if let Some(segm) = segmentation {
        if segm.width != frame.width || segm.height != frame.height {
            bail!(
                "Segmentation size {}x{} does not match image size {}x{}",
                segm.width,
                segm.height,
                frame.width,
                frame.height
            );
        }
    }
    let (min_value, max_value) = frame.pixels.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(*value), max_v.max(*value)),
    );
    let denom = (max_value - min_value).max(f32::EPSILON);
    let alpha = overlay_alpha.clamp(0.0, 1.0);
    let mut pixels = Vec::with_capacity(frame.pixels.len() * 4);
    for (index, value) in frame.pixels.iter().enumerate() {
        let normalized = (((*value - min_value) / denom).clamp(0.0, 1.0) * 255.0) as u8;
        let mut color = [normalized, normalized, normalized];
        if let Some(segm) = segmentation {
            let label = segm.pixels[index];
            if label != 0 {
                let is_selected = selected_label == Some(label);
                let overlay = if is_selected {
                    Color32::from_rgb(255, 240, 120)
                } else {
                    label_color(label)
                };
                let overlay_alpha = if is_selected { alpha.max(0.8) } else { alpha };
                color[0] = blend_channel(color[0], overlay.r(), overlay_alpha);
                color[1] = blend_channel(color[1], overlay.g(), overlay_alpha);
                color[2] = blend_channel(color[2], overlay.b(), overlay_alpha);
            }
        }
        pixels.extend_from_slice(&[color[0], color[1], color[2], 255]);
    }
    Ok(ColorImage::from_rgba_unmultiplied(
        [frame.width, frame.height],
        &pixels,
    ))
}

fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    ((base as f32) * (1.0 - alpha) + (overlay as f32) * alpha).round() as u8
}

fn label_color(label: u32) -> Color32 {
    let hash = label.wrapping_mul(0x9E37_79B9);
    let r = ((hash & 0xFF) as u8).max(60);
    let g = (((hash >> 8) & 0xFF) as u8).max(60);
    let b = (((hash >> 16) & 0xFF) as u8).max(60);
    Color32::from_rgb(r, g, b)
}

fn image_pixel_from_pointer(
    rect: Rect,
    pointer: Pos2,
    image_size: [usize; 2],
) -> Option<(usize, usize)> {
    if !rect.contains(pointer) || image_size[0] == 0 || image_size[1] == 0 {
        return None;
    }
    let uv_x = ((pointer.x - rect.min.x) / rect.width()).clamp(0.0, 0.999_999);
    let uv_y = ((pointer.y - rect.min.y) / rect.height()).clamp(0.0, 0.999_999);
    Some((
        (uv_x * image_size[0] as f32).floor() as usize,
        (uv_y * image_size[1] as f32).floor() as usize,
    ))
}
