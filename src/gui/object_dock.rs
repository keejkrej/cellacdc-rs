use super::app::CellAcdcGui;
use eframe::egui::{self, RichText};

impl CellAcdcGui {
    pub(crate) fn draw_objects_dock(&mut self, ctx: &egui::Context) {
        if !self.persisted.display.show_objects_dock {
            return;
        }
        egui::SidePanel::left("gui_objects_dock")
            .resizable(true)
            .default_width(self.persisted.dock_layout.object_dock_width)
            .show(ctx, |ui| {
                ui.heading("Cell-ACDC objects");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.annotation.object_dock.selected_tab, 0, "Measurements");
                });
                ui.separator();
                match self.current_inspection() {
                    Ok(Some(inspection)) => {
                        ui.label(format!("Frame: {}", inspection.frame_index));
                        ui.label(format!("Objects: {}", inspection.object_count));
                        ui.label(format!("Time: {:.2}s", inspection.time_seconds));
                        ui.separator();
                        if let Some(object) = &inspection.selected_object {
                            ui.label(RichText::new("Measurements").strong());
                            ui.monospace(format!("ID {}", object.label));
                            ui.label(format!("Area (px): {}", object.area_pixels));
                            ui.label(format!("Area (um^2): {:.3}", object.area_um2));
                            ui.label(format!(
                                "Centroid: x={:.1} y={:.1}",
                                object.centroid_x, object.centroid_y
                            ));
                            ui.label(format!(
                                "BBox: x {}..{}  y {}..{}",
                                object.bbox_min_x,
                                object.bbox_max_x,
                                object.bbox_min_y,
                                object.bbox_max_y
                            ));
                            ui.separator();
                            ui.label(RichText::new("Per-channel intensity").strong());
                            for (channel, mean) in &object.channel_mean {
                                let sum = object.channel_sum.get(channel).copied().unwrap_or_default();
                                ui.label(format!("{channel}: mean {mean:.3}  sum {sum:.3}"));
                            }
                        } else {
                            ui.label("Select an object to inspect its measurements.");
                        }
                    }
                    Ok(None) => {
                        ui.label("No segmentation is available for inspection.");
                    }
                    Err(err) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 60, 60),
                            err.to_string(),
                        );
                    }
                }
            });
    }

    pub(crate) fn draw_log_dock(&mut self, ctx: &egui::Context) {
        if !self.persisted.display.show_log_dock {
            return;
        }
        egui::TopBottomPanel::bottom("gui_log_dock")
            .resizable(true)
            .default_height(self.persisted.dock_layout.log_dock_height)
            .show(ctx, |ui| {
                super::workspaces::draw_logs(ui, &self.logs, 180.0);
            });
    }
}
