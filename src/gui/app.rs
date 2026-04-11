use super::jobs::JobHandle;
use super::persist::{self, PersistedState};
use super::state::{AppRoute, ProjectionMode, ViewKey};
use super::workspaces;
use anyhow::{anyhow, Result};
use cellacdc_rs::{
    open_experiment_session, ExperimentSession, FrameProjection, ImportSource, OverwritePolicy,
    PositionSession, SegmentationParams,
};
use eframe::egui::{self, TextureHandle};
use rfd::FileDialog;
use std::path::PathBuf;

const MAX_LOG_LINES: usize = 200;

pub fn launch_gui() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 920.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Cell-ACDC Rust",
        options,
        Box::new(|cc| Ok(Box::new(CellAcdcGui::new(cc)))),
    )
    .map_err(|err| anyhow!(err.to_string()))
}

pub(crate) struct CellAcdcGui {
    pub(crate) persisted: PersistedState,
    pub(crate) last_non_launcher_route: AppRoute,
    pub(crate) experiment: Option<ExperimentSession>,
    pub(crate) selected_position_idx: usize,
    pub(crate) selected_frame_idx: usize,
    pub(crate) texture: Option<TextureHandle>,
    pub(crate) texture_key: Option<ViewKey>,
    pub(crate) last_error: Option<String>,
    pub(crate) logs: Vec<String>,
    pub(crate) active_job: Option<JobHandle>,
    pub(crate) data_structure_scan_path: String,
    pub(crate) data_structure_scan_results: Vec<ImportSource>,
    pub(crate) data_structure_scan_error: Option<String>,
}

impl CellAcdcGui {
    pub(crate) fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = persist::load(cc.storage);
        let last_non_launcher_route = match persisted.route {
            AppRoute::Launcher => AppRoute::Segmentation,
            route => route,
        };
        let mut app = Self {
            persisted,
            last_non_launcher_route,
            experiment: None,
            selected_position_idx: 0,
            selected_frame_idx: 0,
            texture: None,
            texture_key: None,
            last_error: None,
            logs: Vec::new(),
            active_job: None,
            data_structure_scan_path: String::new(),
            data_structure_scan_results: Vec::new(),
            data_structure_scan_error: None,
        };
        if let Some(path) = app.persisted.last_opened_path.clone() {
            let path = PathBuf::from(path);
            if path.exists() {
                if let Err(err) = app.open_path(path) {
                    app.last_error = Some(err.to_string());
                }
            }
        }
        app
    }

    pub(crate) fn pick_and_open_session(&mut self) {
        if let Some(path) = FileDialog::new().pick_folder() {
            if let Err(err) = self.open_path(path) {
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn set_route(&mut self, route: AppRoute) {
        if route != AppRoute::Launcher {
            self.last_non_launcher_route = route;
        }
        self.persisted.route = route;
    }

    pub(crate) fn restore_current_session_route(&mut self) {
        let route = match self.last_non_launcher_route {
            AppRoute::Annotation => AppRoute::Annotation,
            AppRoute::Segmentation => AppRoute::Segmentation,
            _ => AppRoute::Segmentation,
        };
        self.set_route(route);
    }

    pub(crate) fn open_path(&mut self, path: PathBuf) -> Result<()> {
        let experiment = open_experiment_session(&path)?;
        self.experiment = Some(experiment);
        self.selected_position_idx = 0;
        self.selected_frame_idx = 0;
        self.persisted.last_opened_path = Some(path.display().to_string());
        self.set_route(AppRoute::Segmentation);
        self.push_recent_path(path);
        self.sync_selection_with_position();
        self.invalidate_texture();
        self.last_error = None;
        self.append_log("Opened Cell-ACDC session".to_string());
        Ok(())
    }

    pub(crate) fn reload_experiment(&mut self) {
        let Some(current) = self.experiment.as_ref() else {
            return;
        };
        match current.reload() {
            Ok(experiment) => {
                self.experiment = Some(experiment);
                self.sync_selection_with_position();
                self.invalidate_texture();
                self.append_log("Reloaded experiment session".to_string());
                self.last_error = None;
            }
            Err(err) => self.last_error = Some(err.to_string()),
        }
    }

    pub(crate) fn push_recent_path(&mut self, path: PathBuf) {
        let display = path.display().to_string();
        self.persisted.recent_paths.retain(|item| item != &display);
        self.persisted.recent_paths.insert(0, display);
        self.persisted.recent_paths.truncate(8);
    }

    pub(crate) fn append_log(&mut self, message: String) {
        self.logs.push(message);
        if self.logs.len() > MAX_LOG_LINES {
            let overflow = self.logs.len() - MAX_LOG_LINES;
            self.logs.drain(0..overflow);
        }
    }

    pub(crate) fn invalidate_texture(&mut self) {
        self.texture_key = None;
    }

    pub(crate) fn selected_position(&self) -> Option<&PositionSession> {
        self.experiment
            .as_ref()
            .and_then(|experiment| experiment.positions.get(self.selected_position_idx))
    }

    pub(crate) fn selected_segmentation_path(&self) -> Option<PathBuf> {
        let position = self.selected_position()?;
        position
            .segmentations
            .iter()
            .find(|asset| asset.endname == self.persisted.selected_segmentation_endname)
            .map(|asset| asset.path.clone())
    }

    pub(crate) fn sync_selection_with_position(&mut self) {
        let Some(position) = self.selected_position().cloned() else {
            return;
        };
        let channel_names = position.channel_names();
        let default_channel = position.default_channel_name();
        let default_phase = position.default_phase_channel_name();
        let default_fluo = position.default_fluo_channel_name();
        let first_segmentation = position
            .segmentations
            .first()
            .and_then(|asset| asset.endname.clone());
        let size_t = position.spec.size_t;
        let size_z = position.spec.size_z;

        if self.persisted.selected_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.selected_channel)
        {
            self.persisted.selected_channel =
                default_channel.unwrap_or_else(|| String::from("<missing>"));
        }

        if self.persisted.phase_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.phase_channel)
        {
            self.persisted.phase_channel =
                default_phase.unwrap_or_else(|| self.persisted.selected_channel.clone());
        }

        if self.persisted.fluo_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.fluo_channel)
        {
            self.persisted.fluo_channel =
                default_fluo.unwrap_or_else(|| self.persisted.selected_channel.clone());
        }

        let has_selected_segm = position
            .segmentations
            .iter()
            .any(|asset| asset.endname == self.persisted.selected_segmentation_endname);
        if !has_selected_segm {
            self.persisted.selected_segmentation_endname = first_segmentation;
        }

        if self.selected_frame_idx >= size_t {
            self.selected_frame_idx = size_t.saturating_sub(1);
        }
        if size_z <= 1 {
            self.persisted.z_index = 0;
        } else {
            self.persisted.z_index = self.persisted.z_index.min(size_z - 1);
        }
    }

    pub(crate) fn current_projection(&self) -> FrameProjection {
        match self.persisted.projection_mode {
            ProjectionMode::Max => FrameProjection::Max,
            ProjectionMode::ZSlice => FrameProjection::ZSlice(self.persisted.z_index),
        }
    }

    pub(crate) fn current_view_key(&self) -> Option<ViewKey> {
        let position = self.selected_position()?;
        Some(ViewKey {
            position_dir: position.spec.position_dir.clone(),
            channel: self.persisted.selected_channel.clone(),
            frame_index: self.selected_frame_idx,
            projection: self.current_projection(),
            segmentation_endname: self.persisted.selected_segmentation_endname.clone(),
            overlay_alpha_bits: self.persisted.overlay_alpha.to_bits(),
            show_overlay: self.persisted.show_segmentation_overlay,
        })
    }

    pub(crate) fn selected_measurement_suffix(&self) -> Option<String> {
        self.persisted.selected_segmentation_endname.clone()
    }

    pub(crate) fn run_output_suffix(&self) -> Option<String> {
        let trimmed = self.persisted.run_output_suffix.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    pub(crate) fn overwrite_policy(&self) -> OverwritePolicy {
        if self.persisted.overwrite_outputs {
            OverwritePolicy::Overwrite
        } else {
            OverwritePolicy::Refuse
        }
    }

    pub(crate) fn segmentation_params(&self) -> SegmentationParams {
        SegmentationParams {
            tile: self.persisted.tile,
            batch_size: self.persisted.batch_size,
            cellprob_threshold: self.persisted.cellprob_threshold,
            niter: self.persisted.niter,
            min_size: self.persisted.min_size,
        }
    }

    pub(crate) fn autofill_utility_from_selected_segmentation(&mut self) {
        let Some(segmentation_path) = self.selected_segmentation_path() else {
            return;
        };
        self.persisted.utility.segmentation_path = segmentation_path.display().to_string();
        if self.persisted.utility.output_path.trim().is_empty() {
            self.persisted.utility.output_path = workspaces::suggested_output_path(
                self.persisted.utility.selected_tool,
                &segmentation_path,
            )
            .display()
            .to_string();
        }
        if self.persisted.utility.scope_path.trim().is_empty() {
            if let Some(position) = self.selected_position() {
                self.persisted.utility.scope_path =
                    position.spec.position_dir.display().to_string();
            }
        }
    }
}

impl eframe::App for CellAcdcGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        persist::save(storage, &self.persisted);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_active_job();
        self.request_repaint_for_active_job(ctx);

        self.draw_shell_bar(ctx);
        match self.persisted.route {
            AppRoute::Launcher => self.draw_launcher_panel(ctx),
            AppRoute::DataStructure => self.draw_data_structure_panel(ctx),
            AppRoute::DataPrep => self.draw_data_prep_panel(ctx),
            AppRoute::Segmentation => self.render_segmentation_workspace(ctx),
            AppRoute::Annotation => self.draw_annotation_panel(ctx),
            AppRoute::Utilities => self.draw_utility_panel(ctx),
            AppRoute::Help => self.draw_help_panel(ctx),
        }
    }
}
