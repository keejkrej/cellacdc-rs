use crate::gui::app::CellAcdcGui;
use anyhow::{anyhow, bail, Result};
use cellacdc_rs::FrameData;
use eframe::egui::{self, Color32, ColorImage, TextureOptions};

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
            position.load_segmentation_frame(
                self.persisted.selected_segmentation_endname.as_deref(),
                self.selected_frame_idx,
                self.current_projection(),
            )?
        } else {
            None
        };
        compose_color_image(&frame, segm.as_ref(), self.persisted.overlay_alpha)
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
                                self.selected_position_idx = idx;
                                self.selected_frame_idx = 0;
                                self.sync_selection_with_position();
                                self.invalidate_texture();
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
                                self.persisted.selected_segmentation_endname = None;
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
                                    self.persisted.selected_segmentation_endname =
                                        asset.endname.clone();
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

            self.refresh_texture_if_needed(ctx);
            ui.separator();

            if let Some(texture) = &self.texture {
                let available = ui.available_size();
                let image_size = texture.size_vec2();
                let scale = (available.x / image_size.x)
                    .min(available.y / image_size.y)
                    .max(0.1);
                let desired = image_size * scale;
                ui.image((texture.id(), desired));
            } else if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
            } else {
                ui.label("No image available for the current selection.");
            }
        });
    }
}

fn compose_color_image(
    frame: &FrameData<f32>,
    segmentation: Option<&FrameData<u32>>,
    overlay_alpha: f32,
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
                let overlay = label_color(label);
                color[0] = blend_channel(color[0], overlay.r(), alpha);
                color[1] = blend_channel(color[1], overlay.g(), alpha);
                color[2] = blend_channel(color[2], overlay.b(), alpha);
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
