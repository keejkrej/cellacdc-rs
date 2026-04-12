use super::actions::{
    action_label, CELL_CYCLE_ACTIONS, CUSTOM_ANNOTATION_ACTIONS, EDIT_ACTIONS, FILE_ACTIONS,
    HELP_ACTIONS, IMAGE_ACTIONS, LINEAGE_ACTIONS, MEASUREMENT_ACTIONS, SEGMENT_ACTIONS,
    SETTINGS_ACTIONS, TRACKING_ACTIONS, VIEW_ACTIONS,
};
use super::app::CellAcdcGui;
use super::shortcuts::shortcut_label;
use super::state::{GuiActionId, GuiMode};
use eframe::egui::{self, Button, RichText};
use std::path::PathBuf;

impl CellAcdcGui {
    pub(crate) fn draw_gui_chrome(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("gui_menu_and_toolbars").show(ctx, |ui| {
            self.draw_gui_menu_bar(ui);
            if self.persisted.display.show_file_toolbar {
                self.draw_gui_toolbar(
                    ui,
                    "File",
                    &[
                        GuiActionId::OpenSession,
                        GuiActionId::Save,
                        GuiActionId::QuickSave,
                        GuiActionId::SaveAsVersion,
                        GuiActionId::LoadOlderVersions,
                        GuiActionId::ExportImage,
                        GuiActionId::ExportVideo,
                    ],
                );
            }
            if self.annotation.mode != GuiMode::Snapshot {
                self.draw_gui_toolbar(
                    ui,
                    "Mode",
                    &[
                        GuiActionId::ModeViewer,
                        GuiActionId::ModeSegmentationAndTracking,
                        GuiActionId::ModeCellCycleAnalysis,
                        GuiActionId::ModeNormalDivisionLineageTree,
                        GuiActionId::ModeCustomAnnotations,
                    ],
                );
            }
            if self.persisted.display.show_navigation_toolbar {
                self.draw_navigation_toolbar(ui);
            }
            if self.persisted.display.show_edit_toolbar {
                self.draw_gui_toolbar(
                    ui,
                    "Edit",
                    &[
                        GuiActionId::Undo,
                        GuiActionId::Redo,
                        GuiActionId::ToolSelect,
                        GuiActionId::ToolBrush,
                        GuiActionId::ToolEraser,
                        GuiActionId::ToolRelabel,
                        GuiActionId::ToolMerge,
                        GuiActionId::ToolDelete,
                    ],
                );
            }
            if self.persisted.display.show_overlay_toolbar {
                self.draw_gui_toolbar(
                    ui,
                    "Overlay",
                    &[
                        GuiActionId::ToggleSegmentationOverlay,
                        GuiActionId::ToggleOverlayLabels,
                        GuiActionId::ToggleSingleChannelOverlay,
                        GuiActionId::ToggleTrueTransparency,
                        GuiActionId::ToggleScaleBar,
                        GuiActionId::ToggleTimestamp,
                    ],
                );
            }
            if self.persisted.display.show_highlight_toolbar {
                self.draw_highlight_toolbar(ui);
            }
            match self.annotation.mode {
                GuiMode::SegmentationAndTracking => {
                    self.draw_gui_toolbar(ui, "Tracking", TRACKING_ACTIONS);
                }
                GuiMode::CellCycleAnalysis => {
                    self.draw_gui_toolbar(ui, "Cell Cycle", CELL_CYCLE_ACTIONS);
                }
                GuiMode::NormalDivisionLineageTree => {
                    self.draw_gui_toolbar(ui, "Lineage Tree", LINEAGE_ACTIONS);
                }
                GuiMode::CustomAnnotations | GuiMode::Snapshot => {
                    self.draw_custom_annotation_toolbar(ui);
                    if self.annotation.mode == GuiMode::Snapshot {
                        self.draw_gui_toolbar(ui, "Cell Cycle", CELL_CYCLE_ACTIONS);
                    }
                }
                GuiMode::Viewer => {}
            }
        });

        egui::TopBottomPanel::bottom("gui_status_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Status").strong());
                if self.status_text.is_empty() {
                    ui.label("Ready");
                } else {
                    ui.monospace(self.status_text.clone());
                }
                if self.annotation_document_dirty() {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(220, 160, 70), "Unsaved edits");
                }
                if self.pending_annotation_autosave {
                    ui.separator();
                    ui.label("Autosave pending");
                }
            });
        });
    }

    fn draw_gui_menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            self.draw_action_menu(ui, "File", FILE_ACTIONS, true);
            self.draw_action_menu(ui, "Edit", EDIT_ACTIONS, false);
            self.draw_action_menu(ui, "View", VIEW_ACTIONS, false);
            self.draw_action_menu(ui, "Image", IMAGE_ACTIONS, false);
            self.draw_action_menu(ui, "Segment", SEGMENT_ACTIONS, false);
            self.draw_action_menu(ui, "Tracking", TRACKING_ACTIONS, false);
            self.draw_action_menu(ui, "Measurements", MEASUREMENT_ACTIONS, false);
            self.draw_action_menu(ui, "Cell cycle", CELL_CYCLE_ACTIONS, false);
            self.draw_action_menu(ui, "Lineage", LINEAGE_ACTIONS, false);
            self.draw_action_menu(ui, "Custom annotations", CUSTOM_ANNOTATION_ACTIONS, false);
            self.draw_action_menu(ui, "Settings", SETTINGS_ACTIONS, false);
            self.draw_action_menu(ui, "Help", HELP_ACTIONS, false);
        });
    }

    fn draw_action_menu(
        &mut self,
        ui: &mut egui::Ui,
        title: &str,
        actions: &[GuiActionId],
        include_recent: bool,
    ) {
        ui.menu_button(title, |ui| {
            if include_recent {
                let open_state = self.gui_action_state(GuiActionId::OpenSession);
                if ui
                    .add_enabled(
                        open_state.enabled,
                        Button::new(action_label(GuiActionId::OpenSession)),
                    )
                    .clicked()
                {
                    self.dispatch_gui_action(GuiActionId::OpenSession);
                    ui.close_menu();
                }
                ui.separator();
                ui.menu_button("Open Recent", |ui| {
                    let recent_paths = self.persisted.recent_paths.clone();
                    if recent_paths.is_empty() {
                        ui.label("No recent sessions");
                    } else {
                        for path in recent_paths {
                            if ui.button(&path).clicked() {
                                if let Err(err) = self.open_path(PathBuf::from(&path)) {
                                    self.last_error = Some(err.to_string());
                                }
                                ui.close_menu();
                            }
                        }
                    }
                });
                ui.separator();
            }
            for action in actions {
                let state = self.gui_action_state(*action);
                let mut label = action_label(*action).to_string();
                if let Some(shortcut) = shortcut_label(&self.persisted.shortcut_overrides, *action)
                {
                    label.push_str(&format!("    {shortcut}"));
                }
                let button = if state.checked {
                    Button::new(RichText::new(label).strong())
                } else {
                    Button::new(label)
                };
                if ui.add_enabled(state.enabled, button).clicked() {
                    self.dispatch_gui_action(*action);
                    ui.close_menu();
                }
            }
        });
    }

    fn draw_gui_toolbar(&mut self, ui: &mut egui::Ui, title: &str, actions: &[GuiActionId]) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(title).strong());
            for action in actions {
                let state = self.gui_action_state(*action);
                let button = ui.add_enabled(
                    state.enabled,
                    Button::new(action_label(*action)).selected(state.checked),
                );
                if button.clicked() {
                    self.dispatch_gui_action(*action);
                }
            }
        });
    }

    fn draw_navigation_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Navigation").strong());
            if let Some(position) = self.selected_position().cloned() {
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
                egui::ComboBox::from_id_salt("gui_position_combo")
                    .selected_text(
                        position
                            .spec
                            .position_dir
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("Position"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, name) in position_entries {
                            if ui
                                .selectable_label(self.selected_position_idx == idx, name)
                                .clicked()
                            {
                                self.request_position_selection(idx);
                            }
                        }
                    });
                egui::ComboBox::from_id_salt("gui_channel_combo")
                    .selected_text(self.persisted.selected_channel.clone())
                    .show_ui(ui, |ui| {
                        for name in position.channel_names() {
                            if ui
                                .selectable_label(self.persisted.selected_channel == name, &name)
                                .clicked()
                            {
                                self.persisted.selected_channel = name;
                                self.invalidate_texture();
                            }
                        }
                    });
                egui::ComboBox::from_id_salt("gui_segm_combo")
                    .selected_text(super::workspaces::selected_segm_label(
                        &position,
                        &self.persisted.selected_segmentation_endname,
                    ))
                    .show_ui(ui, |ui| {
                        for asset in &position.segmentations {
                            if ui
                                .selectable_label(
                                    self.persisted.selected_segmentation_endname == asset.endname,
                                    &asset.name,
                                )
                                .clicked()
                            {
                                self.request_segmentation_selection(asset.endname.clone());
                            }
                        }
                    });
                if position.spec.size_t > 0 {
                    if self.annotation.mode == GuiMode::Snapshot {
                        let max_pos = self
                            .experiment
                            .as_ref()
                            .map(|experiment| experiment.positions.len().saturating_sub(1))
                            .unwrap_or(0);
                        ui.label(format!(
                            "Positions: {}/{}",
                            self.selected_position_idx + 1,
                            max_pos + 1
                        ));
                    } else {
                        ui.add(
                            egui::Slider::new(
                                &mut self.selected_frame_idx,
                                0..=position.spec.size_t.saturating_sub(1),
                            )
                            .text("Frame"),
                        );
                    }
                }
                if position.spec.size_z > 1 {
                    egui::ComboBox::from_id_salt("gui_view_plane")
                        .selected_text(match self.annotation.view_plane {
                            cellacdc_rs::ViewPlane::XY => "XY",
                            cellacdc_rs::ViewPlane::XZ => "XZ",
                            cellacdc_rs::ViewPlane::YZ => "YZ",
                        })
                        .show_ui(ui, |ui| {
                            for (plane, label) in [
                                (cellacdc_rs::ViewPlane::XY, "XY"),
                                (cellacdc_rs::ViewPlane::XZ, "XZ"),
                                (cellacdc_rs::ViewPlane::YZ, "YZ"),
                            ] {
                                if ui
                                    .selectable_label(self.annotation.view_plane == plane, label)
                                    .clicked()
                                {
                                    self.annotation.view_plane = plane;
                                    self.invalidate_texture();
                                }
                            }
                        });
                }
                egui::ComboBox::from_id_salt("gui_projection_mode")
                    .selected_text(match self.persisted.projection_mode {
                        super::state::ProjectionMode::Max => "Max projection",
                        super::state::ProjectionMode::ZSlice => "Z slice",
                    })
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                self.persisted.projection_mode == super::state::ProjectionMode::Max,
                                "Max projection",
                            )
                            .clicked()
                        {
                            self.persisted.projection_mode = super::state::ProjectionMode::Max;
                            self.invalidate_texture();
                        }
                        if ui
                            .selectable_label(
                                self.persisted.projection_mode
                                    == super::state::ProjectionMode::ZSlice,
                                "Z slice",
                            )
                            .clicked()
                        {
                            self.persisted.projection_mode = super::state::ProjectionMode::ZSlice;
                            self.invalidate_texture();
                        }
                    });
                let max_depth = self
                    .current_view_depth_limit()
                    .unwrap_or(position.spec.size_z);
                if self.persisted.projection_mode == super::state::ProjectionMode::ZSlice
                    && max_depth > 0
                {
                    if ui
                        .add(
                            egui::Slider::new(
                                &mut self.persisted.z_index,
                                0..=max_depth.saturating_sub(1),
                            )
                            .text(match self.annotation.view_plane {
                                cellacdc_rs::ViewPlane::XY => "Z",
                                cellacdc_rs::ViewPlane::XZ => "Y",
                                cellacdc_rs::ViewPlane::YZ => "X",
                            }),
                        )
                        .changed()
                    {
                        self.invalidate_texture();
                    }
                }
            } else {
                ui.label("Open a session to navigate positions and segmentations.");
            }
        });
    }

    fn draw_highlight_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Highlighted ID").strong());
            ui.text_edit_singleline(&mut self.annotation.highlight.highlighted_input);
            if ui.button("Select").clicked() {
                if let Ok(label) = self
                    .annotation
                    .highlight
                    .highlighted_input
                    .trim()
                    .parse::<u32>()
                {
                    if let Err(err) = self.annotation_select_label(Some(label)) {
                        self.last_error = Some(err.to_string());
                    } else {
                        self.annotation.highlight.searched_label = Some(label);
                        self.invalidate_texture();
                    }
                }
            }
            if let Some(selected) = self.current_annotation_label() {
                ui.label(format!("Selected: {selected}"));
            }
        });
    }

    fn draw_custom_annotation_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Custom annotations").strong());
            for action in CUSTOM_ANNOTATION_ACTIONS {
                let state = self.gui_action_state(*action);
                if ui
                    .add_enabled(
                        state.enabled,
                        Button::new(action_label(*action)).selected(state.checked),
                    )
                    .clicked()
                {
                    self.dispatch_gui_action(*action);
                }
            }
            ui.separator();
            let annotation_names = self
                .annotation
                .custom_annotations
                .definitions
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for name in annotation_names {
                let is_active = self
                    .annotation
                    .active_custom_annotation
                    .active_name
                    .as_deref()
                    == Some(name.as_str());
                let response = ui.add(Button::new(name.clone()).selected(is_active));
                if response.clicked() {
                    if is_active {
                        self.annotation.active_custom_annotation.active_name = None;
                    } else {
                        self.annotation.active_custom_annotation.active_name = Some(name.clone());
                    }
                    self.invalidate_texture();
                }
                response.context_menu(|ui| {
                    if ui.button("Modify").clicked() {
                        if let Err(err) = self.open_custom_annotation_editor(Some(&name)) {
                            self.last_error = Some(err.to_string());
                        }
                        ui.close_menu();
                    }
                    if ui.button("Remove only button").clicked() {
                        if let Err(err) = self.remove_custom_annotation(&name, false) {
                            self.last_error = Some(err.to_string());
                        }
                        ui.close_menu();
                    }
                    if ui.button("Remove also column with annotations").clicked() {
                        if let Err(err) = self.remove_custom_annotation(&name, true) {
                            self.last_error = Some(err.to_string());
                        }
                        ui.close_menu();
                    }
                    if let Some(definition) = self
                        .annotation
                        .custom_annotations
                        .definitions
                        .get(&name)
                        .cloned()
                    {
                        let keep_label = if definition.keep_active {
                            "Disable keep tool active"
                        } else {
                            "Keep tool active"
                        };
                        if ui.button(keep_label).clicked() {
                            if let Err(err) = self.update_custom_annotation_flags(
                                &name,
                                Some(!definition.keep_active),
                                None,
                            ) {
                                self.last_error = Some(err.to_string());
                            }
                            ui.close_menu();
                        }
                        let hide_label = if definition.hide_when_inactive {
                            "Show annotation when inactive"
                        } else {
                            "Hide annotation when inactive"
                        };
                        if ui.button(hide_label).clicked() {
                            if let Err(err) = self.update_custom_annotation_flags(
                                &name,
                                None,
                                Some(!definition.hide_when_inactive),
                            ) {
                                self.last_error = Some(err.to_string());
                            }
                            ui.close_menu();
                        }
                    }
                });
            }
        });
    }
}
