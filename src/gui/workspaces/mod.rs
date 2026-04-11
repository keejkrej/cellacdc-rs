mod annotation;
mod data_prep;
mod data_structure;
mod help;
mod launcher;
mod segmentation;
mod utilities;
mod viewer;

use crate::gui::state::UtilityTool;
use anyhow::{anyhow, Result};
use cellacdc_rs::{PositionSession, UtilityOutputPaths};
use eframe::egui::{self, Button, RichText};
use rfd::FileDialog;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub(crate) enum PathEditKind {
    PickFile,
    PickFolder,
    SaveFile,
}

#[derive(Clone, Copy)]
pub(crate) struct LauncherModuleSpec {
    pub(crate) title: &'static str,
    pub(crate) subtitle: &'static str,
    pub(crate) route: crate::gui::state::AppRoute,
    pub(crate) enabled_when_session_required: bool,
    pub(crate) is_primary_module: bool,
}

pub(crate) fn workspace_display_name(route: crate::gui::state::AppRoute) -> &'static str {
    match route {
        crate::gui::state::AppRoute::Launcher => "Launcher",
        crate::gui::state::AppRoute::DataStructure => "Data Structure",
        crate::gui::state::AppRoute::DataPrep => "Data Prep",
        crate::gui::state::AppRoute::Segmentation => "Segmentation",
        crate::gui::state::AppRoute::Annotation => "GUI",
        crate::gui::state::AppRoute::Utilities => "Utilities",
        crate::gui::state::AppRoute::Help => "Help",
    }
}

pub(crate) fn workspace_title(route: crate::gui::state::AppRoute) -> &'static str {
    match route {
        crate::gui::state::AppRoute::Launcher => "Launcher",
        crate::gui::state::AppRoute::DataStructure => "Create Data Structure",
        crate::gui::state::AppRoute::DataPrep => "Data Prep",
        crate::gui::state::AppRoute::Segmentation => "Segmentation",
        crate::gui::state::AppRoute::Annotation => "GUI",
        crate::gui::state::AppRoute::Utilities => "Utilities",
        crate::gui::state::AppRoute::Help => "Help",
    }
}

pub(crate) fn draw_workspace_header(
    ui: &mut egui::Ui,
    route: crate::gui::state::AppRoute,
    description: Option<&str>,
    session_path: Option<&Path>,
    show_open_session: bool,
) -> (bool, bool) {
    let mut back_to_launcher = false;
    let mut open_session = false;
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(workspace_title(route)).heading().strong());
                if let Some(description) = description {
                    ui.label(description);
                }
                if let Some(path) = session_path {
                    ui.monospace(path.display().to_string());
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                if ui.button("Back to Launcher").clicked() {
                    back_to_launcher = true;
                }
                if show_open_session && ui.button("Open Session").clicked() {
                    open_session = true;
                }
            });
        });
    });
    (back_to_launcher, open_session)
}

pub(crate) fn draw_launcher_module_button(ui: &mut egui::Ui, spec: LauncherModuleSpec) -> bool {
    let mut clicked = false;
    ui.group(|ui| {
        let width = ui.available_width();
        let title = if spec.is_primary_module {
            RichText::new(spec.title).strong().size(16.0)
        } else {
            RichText::new(spec.title).strong()
        };
        if ui.add_sized([width, 34.0], Button::new(title)).clicked() {
            clicked = true;
        }
        ui.label(spec.subtitle);
        if spec.enabled_when_session_required {
            ui.small("If no session is open, this will ask for one first.");
        }
    });
    clicked
}

pub(crate) fn render_planned_workspace(
    ctx: &egui::Context,
    route: crate::gui::state::AppRoute,
    body: &str,
    milestones: &[&str],
    session_path: Option<&Path>,
    show_open_session: bool,
) -> (bool, bool) {
    let mut back_to_launcher = false;
    let mut open_session = false;
    egui::CentralPanel::default().show(ctx, |ui| {
        let (back_requested, open_requested) =
            draw_workspace_header(ui, route, None, session_path, show_open_session);
        back_to_launcher = back_requested;
        open_session = open_requested;
        ui.add_space(8.0);
        ui.label(body);
        ui.add_space(8.0);
        ui.group(|ui| {
            ui.label(RichText::new("Next Milestones").strong());
            for item in milestones {
                ui.label(format!("- {item}"));
            }
        });
    });
    (back_to_launcher, open_session)
}

pub(crate) fn path_edit_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    kind: PathEditKind,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.text_edit_singleline(value);
        if ui.button("Browse").clicked() {
            let selected = match kind {
                PathEditKind::PickFile => FileDialog::new().pick_file(),
                PathEditKind::PickFolder => FileDialog::new().pick_folder(),
                PathEditKind::SaveFile => FileDialog::new().save_file(),
            };
            if let Some(path) = selected {
                *value = path.display().to_string();
            }
        }
    });
}

pub(crate) fn combo_for_channel(
    ui: &mut egui::Ui,
    label: &str,
    channel_names: &[String],
    selected: &mut String,
) {
    egui::ComboBox::from_label(label)
        .selected_text(selected.clone())
        .show_ui(ui, |ui| {
            for name in channel_names {
                ui.selectable_value(selected, name.clone(), name);
            }
        });
}

pub(crate) fn draw_logs(ui: &mut egui::Ui, logs: &[String], max_height: f32) {
    ui.label(RichText::new("Logs").strong());
    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .max_height(max_height)
        .show(ui, |ui| {
            for line in logs {
                ui.monospace(line);
            }
        });
}

pub(crate) fn parse_optional_usize(value: &str) -> Result<Option<usize>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        trimmed
            .parse::<usize>()
            .map(Some)
            .map_err(|err| anyhow!("Failed to parse integer {trimmed:?}: {err}"))
    }
}

pub(crate) fn suggested_output_path(tool: UtilityTool, input_path: &Path) -> PathBuf {
    match tool {
        UtilityTool::CountObjects => input_path.with_extension("csv"),
        UtilityTool::FillHoles => append_to_stem(input_path, "_filled"),
        UtilityTool::Connect3d => append_to_stem(input_path, "_connected3d"),
        UtilityTool::Stack2dTo3d => append_to_stem(input_path, "_stacked3d"),
        UtilityTool::CombineChannels => input_path.to_path_buf(),
    }
}

pub(crate) fn append_to_stem(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("npz");
    path.with_file_name(format!("{stem}{suffix}.{ext}"))
}

pub(crate) fn utility_tool_label(tool: UtilityTool) -> &'static str {
    match tool {
        UtilityTool::CountObjects => "Count Objects",
        UtilityTool::FillHoles => "Fill Holes",
        UtilityTool::Connect3d => "Connect 3D Segm",
        UtilityTool::Stack2dTo3d => "Stack 2D to 3D",
        UtilityTool::CombineChannels => "Combine Channels",
    }
}

pub(crate) fn selected_segm_label(position: &PositionSession, endname: &Option<String>) -> String {
    endname
        .clone()
        .map(|value| format!("segm_{value}"))
        .or_else(|| {
            position
                .segmentations
                .iter()
                .find(|asset| asset.endname.is_none())
                .map(|_| "segm".to_string())
        })
        .unwrap_or_else(|| "<none>".to_string())
}

pub(crate) fn format_utility_summary(action: &str, result: &UtilityOutputPaths) -> String {
    let mut summary = format!("{action} -> {}", result.primary_path.display());
    if !result.secondary_paths.is_empty() {
        summary.push_str(&format!(
            " (+{} sidecar file(s))",
            result.secondary_paths.len()
        ));
    }
    summary
}
