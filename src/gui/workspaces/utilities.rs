use crate::gui::app::CellAcdcGui;
use crate::gui::state::{AppRoute, ResolutionLayoutChoice, UtilityTool};
use eframe::egui::{self, RichText};

use super::{
    draw_workspace_header, path_edit_row, suggested_output_path, utility_tool_label, PathEditKind,
};

impl CellAcdcGui {
    pub(crate) fn draw_utility_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (back_to_launcher, _) = draw_workspace_header(
                ui,
                AppRoute::Utilities,
                Some(
                    "Rust-native utility center for file-based Cell-ACDC workflows. The current UI covers the utilities already hardened in this desktop shell.",
                ),
                self.experiment.as_ref().map(|experiment| experiment.root_path.as_path()),
                false,
            );
            if back_to_launcher {
                self.set_route(AppRoute::Launcher);
            }

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                for tool in [
                    UtilityTool::CountObjects,
                    UtilityTool::FillHoles,
                    UtilityTool::Connect3d,
                    UtilityTool::Stack2dTo3d,
                    UtilityTool::CombineChannels,
                ] {
                    ui.selectable_value(
                        &mut self.persisted.utility.selected_tool,
                        tool,
                        utility_tool_label(tool),
                    );
                }
            });

            if ui.button("Use selected segmentation from current session").clicked() {
                self.autofill_utility_from_selected_segmentation();
            }

            ui.add_space(8.0);
            ui.group(|ui| match self.persisted.utility.selected_tool {
                UtilityTool::CountObjects => self.draw_count_objects_form(ui),
                UtilityTool::FillHoles => self.draw_fill_holes_form(ui),
                UtilityTool::Connect3d => self.draw_connect_3d_form(ui),
                UtilityTool::Stack2dTo3d => self.draw_stack_2d_to_3d_form(ui),
                UtilityTool::CombineChannels => self.draw_combine_channels_form(ui),
            });

            ui.add_space(10.0);
            ui.group(|ui| {
                ui.label(RichText::new("Planned Next").strong());
                ui.label("CLI-backed tools still to wire into the desktop UI:");
                for tool in [
                    "Concat ACDC output tables",
                    "Combine metrics and compute multi-channel metrics",
                    "Prepare z-stack segm info",
                    "Filter segmentation from coordinate tables",
                    "Apply tracking from tables",
                    "Apply TrackMate XML tracking",
                    "Lineage normalization, propagation, and export",
                    "Mother-bud total generation",
                ] {
                    ui.label(format!("- {tool}"));
                }
            });
        });
    }

    pub(crate) fn draw_count_objects_form(&mut self, ui: &mut egui::Ui) {
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output CSV",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest output path").clicked() {
            let input = std::path::PathBuf::from(self.persisted.utility.segmentation_path.clone());
            self.persisted.utility.output_path =
                suggested_output_path(UtilityTool::CountObjects, &input)
                    .display()
                    .to_string();
        }
        self.draw_resolution_controls(ui);
        if ui
            .add_enabled(
                self.active_job.is_none(),
                egui::Button::new("Run Count Objects"),
            )
            .clicked()
        {
            self.start_count_objects_job();
        }
    }

    pub(crate) fn draw_fill_holes_form(&mut self, ui: &mut egui::Ui) {
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output segmentation",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest output path").clicked() {
            let input = std::path::PathBuf::from(self.persisted.utility.segmentation_path.clone());
            self.persisted.utility.output_path =
                suggested_output_path(UtilityTool::FillHoles, &input)
                    .display()
                    .to_string();
        }
        self.draw_resolution_controls(ui);
        if ui
            .add_enabled(
                self.active_job.is_none(),
                egui::Button::new("Run Fill Holes"),
            )
            .clicked()
        {
            self.start_fill_holes_job();
        }
    }

    pub(crate) fn draw_connect_3d_form(&mut self, ui: &mut egui::Ui) {
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output segmentation",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest output path").clicked() {
            let input = std::path::PathBuf::from(self.persisted.utility.segmentation_path.clone());
            self.persisted.utility.output_path =
                suggested_output_path(UtilityTool::Connect3d, &input)
                    .display()
                    .to_string();
        }
        self.draw_resolution_controls(ui);
        if ui
            .add_enabled(
                self.active_job.is_none(),
                egui::Button::new("Run Connect 3D"),
            )
            .clicked()
        {
            self.start_connect_3d_job();
        }
    }

    pub(crate) fn draw_stack_2d_to_3d_form(&mut self, ui: &mut egui::Ui) {
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output segmentation",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest output path").clicked() {
            let input = std::path::PathBuf::from(self.persisted.utility.segmentation_path.clone());
            self.persisted.utility.output_path =
                suggested_output_path(UtilityTool::Stack2dTo3d, &input)
                    .display()
                    .to_string();
        }
        ui.add(
            egui::DragValue::new(&mut self.persisted.utility.stack_target_size_z)
                .range(1..=1024)
                .prefix("size_z "),
        );
        self.draw_resolution_controls(ui);
        if ui
            .add_enabled(
                self.active_job.is_none(),
                egui::Button::new("Run Stack 2D to 3D"),
            )
            .clicked()
        {
            self.start_stack_2d_to_3d_job();
        }
    }

    pub(crate) fn draw_combine_channels_form(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Scope mode")
            .selected_text(match self.persisted.utility.scope_mode {
                crate::gui::state::UtilityScopeMode::Auto => "Auto detect",
                crate::gui::state::UtilityScopeMode::Position => "Position",
                crate::gui::state::UtilityScopeMode::Experiment => "Experiment",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    crate::gui::state::UtilityScopeMode::Auto,
                    "Auto detect",
                );
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    crate::gui::state::UtilityScopeMode::Position,
                    "Position",
                );
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    crate::gui::state::UtilityScopeMode::Experiment,
                    "Experiment",
                );
            });
        path_edit_row(
            ui,
            "Scope path",
            &mut self.persisted.utility.scope_path,
            PathEditKind::PickFolder,
        );
        path_edit_row(
            ui,
            "Recipe",
            &mut self.persisted.utility.recipe_path,
            PathEditKind::PickFile,
        );
        ui.horizontal(|ui| {
            ui.label("Append name");
            ui.text_edit_singleline(&mut self.persisted.utility.append_name);
        });
        if ui
            .add_enabled(
                self.active_job.is_none(),
                egui::Button::new("Run Combine Channels"),
            )
            .clicked()
        {
            self.start_combine_channels_job();
        }
    }

    pub(crate) fn draw_resolution_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Mask Resolution Override", |ui| {
            ui.label(
                "Leave these empty to use automatic detection. Override only when the mask layout is ambiguous.",
            );
            ui.horizontal(|ui| {
                ui.label("size_t");
                ui.text_edit_singleline(&mut self.persisted.utility.resolution_size_t);
                ui.label("size_z");
                ui.text_edit_singleline(&mut self.persisted.utility.resolution_size_z);
            });
            egui::ComboBox::from_label("Layout")
                .selected_text(match self.persisted.utility.resolution_layout {
                    ResolutionLayoutChoice::Auto => "Auto detect",
                    ResolutionLayoutChoice::Yx => "YX",
                    ResolutionLayoutChoice::Tyx => "TYX",
                    ResolutionLayoutChoice::Zyx => "ZYX",
                    ResolutionLayoutChoice::Tzyx => "TZYX",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.persisted.utility.resolution_layout,
                        ResolutionLayoutChoice::Auto,
                        "Auto detect",
                    );
                    ui.selectable_value(
                        &mut self.persisted.utility.resolution_layout,
                        ResolutionLayoutChoice::Yx,
                        "YX",
                    );
                    ui.selectable_value(
                        &mut self.persisted.utility.resolution_layout,
                        ResolutionLayoutChoice::Tyx,
                        "TYX",
                    );
                    ui.selectable_value(
                        &mut self.persisted.utility.resolution_layout,
                        ResolutionLayoutChoice::Zyx,
                        "ZYX",
                    );
                    ui.selectable_value(
                        &mut self.persisted.utility.resolution_layout,
                        ResolutionLayoutChoice::Tzyx,
                        "TZYX",
                    );
                });
        });
    }
}
