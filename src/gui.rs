use anyhow::{anyhow, bail, Result};
use cellacdc_rs::{
    measure_experiment, measure_position, open_experiment_session, resolve_position,
    run_experiment, run_position, ExperimentRunConfig, ExperimentSession, FrameData,
    FrameProjection, MeasurementExperimentConfig, MeasurementRunConfig, OverwritePolicy,
    PositionSession, SegmentationParams, TrackingConfig,
};
use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

const APP_KEY: &str = "cellacdc_rs_gui";
const MAX_LOG_LINES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ProjectionMode {
    Max,
    ZSlice,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    recent_paths: Vec<String>,
    last_opened_path: Option<String>,
    phase_channel: String,
    fluo_channel: String,
    model_path: String,
    run_output_suffix: String,
    cpu: bool,
    track: bool,
    track_ioa_threshold: f32,
    tile: usize,
    batch_size: usize,
    cellprob_threshold: f32,
    niter: usize,
    min_size: usize,
    overwrite_outputs: bool,
    overlay_alpha: f32,
    show_segmentation_overlay: bool,
    selected_channel: String,
    selected_segmentation_endname: Option<String>,
    projection_mode: ProjectionMode,
    z_index: usize,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            recent_paths: Vec::new(),
            last_opened_path: None,
            phase_channel: String::new(),
            fluo_channel: String::new(),
            model_path: String::new(),
            run_output_suffix: String::new(),
            cpu: false,
            track: false,
            track_ioa_threshold: 0.4,
            tile: 256,
            batch_size: 1,
            cellprob_threshold: 0.0,
            niter: 200,
            min_size: 15,
            overwrite_outputs: false,
            overlay_alpha: 0.45,
            show_segmentation_overlay: true,
            selected_channel: String::new(),
            selected_segmentation_endname: None,
            projection_mode: ProjectionMode::Max,
            z_index: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ViewKey {
    position_dir: PathBuf,
    channel: String,
    frame_index: usize,
    projection: FrameProjection,
    segmentation_endname: Option<String>,
    overlay_alpha_bits: u32,
    show_overlay: bool,
}

struct BackgroundJob {
    receiver: Receiver<JobEvent>,
    label: String,
}

enum JobEvent {
    Log(String),
    Finished(Result<JobResult, String>),
}

struct JobResult {
    summary: String,
    reload_session: bool,
    select_segmentation_endname: Option<Option<String>>,
}

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

struct CellAcdcGui {
    persisted: PersistedState,
    experiment: Option<ExperimentSession>,
    selected_position_idx: usize,
    selected_frame_idx: usize,
    texture: Option<TextureHandle>,
    texture_key: Option<ViewKey>,
    last_error: Option<String>,
    logs: Vec<String>,
    active_job: Option<BackgroundJob>,
}

impl CellAcdcGui {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = cc
            .storage
            .and_then(|storage| storage.get_string(APP_KEY))
            .and_then(|json| serde_json::from_str::<PersistedState>(&json).ok())
            .unwrap_or_default();

        let mut app = Self {
            persisted,
            experiment: None,
            selected_position_idx: 0,
            selected_frame_idx: 0,
            texture: None,
            texture_key: None,
            last_error: None,
            logs: Vec::new(),
            active_job: None,
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

    fn open_path(&mut self, path: PathBuf) -> Result<()> {
        let experiment = open_experiment_session(&path)?;
        self.experiment = Some(experiment);
        self.selected_position_idx = 0;
        self.selected_frame_idx = 0;
        self.persisted.last_opened_path = Some(path.display().to_string());
        self.push_recent_path(path);
        self.sync_selection_with_position();
        self.invalidate_texture();
        self.last_error = None;
        self.append_log("Opened Cell-ACDC session".to_string());
        Ok(())
    }

    fn reload_experiment(&mut self) {
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

    fn push_recent_path(&mut self, path: PathBuf) {
        let display = path.display().to_string();
        self.persisted.recent_paths.retain(|item| item != &display);
        self.persisted.recent_paths.insert(0, display);
        self.persisted.recent_paths.truncate(8);
    }

    fn append_log(&mut self, message: String) {
        self.logs.push(message);
        if self.logs.len() > MAX_LOG_LINES {
            let overflow = self.logs.len() - MAX_LOG_LINES;
            self.logs.drain(0..overflow);
        }
    }

    fn invalidate_texture(&mut self) {
        self.texture_key = None;
    }

    fn selected_position(&self) -> Option<&PositionSession> {
        self.experiment
            .as_ref()
            .and_then(|experiment| experiment.positions.get(self.selected_position_idx))
    }

    fn sync_selection_with_position(&mut self) {
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

    fn current_projection(&self) -> FrameProjection {
        match self.persisted.projection_mode {
            ProjectionMode::Max => FrameProjection::Max,
            ProjectionMode::ZSlice => FrameProjection::ZSlice(self.persisted.z_index),
        }
    }

    fn current_view_key(&self) -> Option<ViewKey> {
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

    fn refresh_texture_if_needed(&mut self, ctx: &egui::Context) {
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

    fn render_current_view(&self) -> Result<ColorImage> {
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

    fn selected_measurement_suffix(&self) -> Option<String> {
        self.persisted.selected_segmentation_endname.clone()
    }

    fn run_output_suffix(&self) -> Option<String> {
        let trimmed = self.persisted.run_output_suffix.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    fn overwrite_policy(&self) -> OverwritePolicy {
        if self.persisted.overwrite_outputs {
            OverwritePolicy::Overwrite
        } else {
            OverwritePolicy::Refuse
        }
    }

    fn segmentation_params(&self) -> SegmentationParams {
        SegmentationParams {
            tile: self.persisted.tile,
            batch_size: self.persisted.batch_size,
            cellprob_threshold: self.persisted.cellprob_threshold,
            niter: self.persisted.niter,
            min_size: self.persisted.min_size,
        }
    }

    fn tracking_config(&self) -> Option<TrackingConfig> {
        self.persisted.track.then(|| TrackingConfig {
            ioa_threshold: self.persisted.track_ioa_threshold,
        })
    }

    fn poll_active_job(&mut self) {
        let mut events = Vec::new();
        if let Some(job) = self.active_job.as_mut() {
            while let Ok(event) = job.receiver.try_recv() {
                events.push(event);
            }
        }
        let mut finished = false;
        for event in events {
            match event {
                JobEvent::Log(message) => self.append_log(message),
                JobEvent::Finished(result) => {
                    finished = true;
                    match result {
                        Ok(outcome) => {
                            self.append_log(outcome.summary);
                            if let Some(endname) = outcome.select_segmentation_endname {
                                self.persisted.selected_segmentation_endname = endname;
                            }
                            if outcome.reload_session {
                                self.reload_experiment();
                            }
                        }
                        Err(err) => {
                            self.last_error = Some(err.clone());
                            self.append_log(format!("Job failed: {err}"));
                        }
                    }
                }
            }
        }
        if finished {
            self.active_job = None;
        }
    }

    fn start_measure_position_job(&mut self) {
        let Some(position) = self.selected_position() else {
            return;
        };
        if self.active_job.is_some() {
            return;
        }
        let position_path = position.spec.position_dir.clone();
        let segm_endname = self.selected_measurement_suffix();
        let overwrite_policy = self.overwrite_policy();
        let (sender, receiver) = mpsc::channel();
        let label = format!("Measure {}", position_path.display());
        self.append_log(format!("Queued job: {label}"));
        std::thread::spawn(move || {
            let _ = sender.send(JobEvent::Log(format!(
                "Measuring {}",
                position_path.display()
            )));
            let outcome = measure_position(MeasurementRunConfig {
                position_path,
                segm_endname,
                overwrite_policy,
            })
            .map(|result| JobResult {
                summary: format!(
                    "Measured {} frame(s) -> {}",
                    result.frames_processed,
                    result.outputs.acdc_output_csv_path.display()
                ),
                reload_session: true,
                select_segmentation_endname: None,
            })
            .map_err(|err| err.to_string());
            let _ = sender.send(JobEvent::Finished(outcome));
        });
        self.active_job = Some(BackgroundJob { receiver, label });
    }

    fn start_measure_experiment_job(&mut self) {
        let Some(experiment) = self.experiment.as_ref() else {
            return;
        };
        if experiment.is_single_position || self.active_job.is_some() {
            return;
        }
        let experiment_dir = experiment.root_path.clone();
        let segm_endname = self.selected_measurement_suffix();
        let overwrite_policy = self.overwrite_policy();
        let (sender, receiver) = mpsc::channel();
        let label = format!("Measure experiment {}", experiment_dir.display());
        self.append_log(format!("Queued job: {label}"));
        std::thread::spawn(move || {
            let _ = sender.send(JobEvent::Log(format!(
                "Measuring experiment {}",
                experiment_dir.display()
            )));
            let outcome = measure_experiment(MeasurementExperimentConfig {
                experiment_dir,
                segm_endname,
                overwrite_policy,
            })
            .map(|results| JobResult {
                summary: format!("Measured {} position(s)", results.len()),
                reload_session: true,
                select_segmentation_endname: None,
            })
            .map_err(|err| err.to_string());
            let _ = sender.send(JobEvent::Finished(outcome));
        });
        self.active_job = Some(BackgroundJob { receiver, label });
    }

    fn start_run_position_job(&mut self) {
        let Some(position) = self.selected_position() else {
            return;
        };
        if self.active_job.is_some() {
            return;
        }
        let position_path = position.spec.position_dir.clone();
        let phase_channel = self.persisted.phase_channel.clone();
        let fluo_channel = self.persisted.fluo_channel.clone();
        let model_path = PathBuf::from(self.persisted.model_path.trim());
        let segm_endname = self.run_output_suffix();
        let overwrite_policy = self.overwrite_policy();
        let cpu = self.persisted.cpu;
        let params = self.segmentation_params();
        let tracking = self.tracking_config();
        let (sender, receiver) = mpsc::channel();
        let label = format!("Segment {}", position_path.display());
        self.append_log(format!("Queued job: {label}"));
        std::thread::spawn(move || {
            let _ = sender.send(JobEvent::Log(format!(
                "Segmenting {}",
                position_path.display()
            )));
            let outcome = (|| -> Result<JobResult> {
                if model_path.as_os_str().is_empty() {
                    bail!("Model path is required for segmentation");
                }
                let position = resolve_position(&position_path, phase_channel, fluo_channel)?;
                let result = run_position(cellacdc_rs::SegmentationRunConfig {
                    position,
                    model_path,
                    segm_endname: segm_endname.clone(),
                    overwrite_policy,
                    cpu,
                    params,
                    tracking,
                })?;
                Ok(JobResult {
                    summary: format!(
                        "Segmented {} frame(s) -> {}",
                        result.frames_processed,
                        result.outputs.segm_npz_path.display()
                    ),
                    reload_session: true,
                    select_segmentation_endname: Some(segm_endname),
                })
            })()
            .map_err(|err| err.to_string());
            let _ = sender.send(JobEvent::Finished(outcome));
        });
        self.active_job = Some(BackgroundJob { receiver, label });
    }

    fn start_run_experiment_job(&mut self) {
        let Some(experiment) = self.experiment.as_ref() else {
            return;
        };
        if experiment.is_single_position || self.active_job.is_some() {
            return;
        }
        let experiment_dir = experiment.root_path.clone();
        let phase_channel = self.persisted.phase_channel.clone();
        let fluo_channel = self.persisted.fluo_channel.clone();
        let model_path = PathBuf::from(self.persisted.model_path.trim());
        let segm_endname = self.run_output_suffix();
        let overwrite_policy = self.overwrite_policy();
        let cpu = self.persisted.cpu;
        let params = self.segmentation_params();
        let tracking = self.tracking_config();
        let (sender, receiver) = mpsc::channel();
        let label = format!("Segment experiment {}", experiment_dir.display());
        self.append_log(format!("Queued job: {label}"));
        std::thread::spawn(move || {
            let _ = sender.send(JobEvent::Log(format!(
                "Segmenting experiment {}",
                experiment_dir.display()
            )));
            let outcome = (|| -> Result<JobResult> {
                if model_path.as_os_str().is_empty() {
                    bail!("Model path is required for segmentation");
                }
                let results = run_experiment(ExperimentRunConfig {
                    experiment_dir,
                    phase_channel,
                    fluo_channel,
                    model_path,
                    segm_endname: segm_endname.clone(),
                    overwrite_policy,
                    cpu,
                    params,
                    tracking,
                })?;
                Ok(JobResult {
                    summary: format!("Segmented {} position(s)", results.len()),
                    reload_session: true,
                    select_segmentation_endname: Some(segm_endname),
                })
            })()
            .map_err(|err| err.to_string());
            let _ = sender.send(JobEvent::Finished(outcome));
        });
        self.active_job = Some(BackgroundJob { receiver, label });
    }

    fn draw_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open Experiment or Position").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        if let Err(err) = self.open_path(path) {
                            self.last_error = Some(err.to_string());
                        }
                    }
                }
                if ui
                    .add_enabled(self.experiment.is_some(), egui::Button::new("Reload"))
                    .clicked()
                {
                    self.reload_experiment();
                }

                if !self.persisted.recent_paths.is_empty() {
                    egui::ComboBox::from_label("Recent")
                        .selected_text("Select path")
                        .show_ui(ui, |ui| {
                            for recent in self.persisted.recent_paths.clone() {
                                if ui.selectable_label(false, &recent).clicked() {
                                    if let Err(err) = self.open_path(PathBuf::from(recent)) {
                                        self.last_error = Some(err.to_string());
                                    }
                                }
                            }
                        });
                }

                if let Some(job) = self.active_job.as_ref() {
                    ui.separator();
                    ui.label(RichText::new(format!("Running: {}", job.label)).strong());
                }
            });
        });
    }

    fn draw_left_panel(&mut self, ctx: &egui::Context) {
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
                    return;
                }

                ui.separator();
                ui.label(RichText::new("Positions").strong());
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
                    ui.label(RichText::new("Position Details").strong());
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
                        .selected_text(
                            self.persisted
                                .selected_segmentation_endname
                                .clone()
                                .map(|value| format!("segm_{value}"))
                                .or_else(|| {
                                    position
                                        .segmentations
                                        .iter()
                                        .find(|asset| asset.endname.is_none())
                                        .map(|_| "segm".to_string())
                                })
                                .unwrap_or_else(|| "<none>".to_string()),
                        )
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

    fn draw_right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("jobs_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("Jobs");
                let Some(position) = self.selected_position().cloned() else {
                    ui.label("Open a session to configure jobs.");
                    return;
                };
                let channel_names = position.channel_names();
                let has_position_segmentations = !position.segmentations.is_empty();
                let experiment_supports_batch = self
                    .experiment
                    .as_ref()
                    .map(|experiment| !experiment.is_single_position)
                    .unwrap_or(false);

                ui.label(RichText::new("Segmentation").strong());
                combo_for_channel(
                    ui,
                    "Phase channel",
                    &channel_names,
                    &mut self.persisted.phase_channel,
                );
                combo_for_channel(
                    ui,
                    "Fluorescence channel",
                    &channel_names,
                    &mut self.persisted.fluo_channel,
                );
                ui.horizontal(|ui| {
                    ui.label("Model");
                    ui.text_edit_singleline(&mut self.persisted.model_path);
                    if ui.button("Browse").clicked() {
                        if let Some(path) = FileDialog::new().pick_file() {
                            self.persisted.model_path = path.display().to_string();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Output suffix");
                    ui.text_edit_singleline(&mut self.persisted.run_output_suffix);
                });
                ui.checkbox(&mut self.persisted.cpu, "Run on CPU");
                ui.checkbox(&mut self.persisted.track, "Track after segmentation");
                if self.persisted.track {
                    ui.add(
                        egui::Slider::new(&mut self.persisted.track_ioa_threshold, 0.0..=1.0)
                            .text("Tracking IoA"),
                    );
                }
                ui.checkbox(&mut self.persisted.overwrite_outputs, "Overwrite outputs");
                ui.collapsing("Segmentation parameters", |ui| {
                    ui.add(egui::DragValue::new(&mut self.persisted.tile).prefix("tile "));
                    ui.add(egui::DragValue::new(&mut self.persisted.batch_size).prefix("batch "));
                    ui.add(
                        egui::DragValue::new(&mut self.persisted.cellprob_threshold)
                            .prefix("cellprob "),
                    );
                    ui.add(egui::DragValue::new(&mut self.persisted.niter).prefix("niter "));
                    ui.add(egui::DragValue::new(&mut self.persisted.min_size).prefix("min size "));
                });

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(self.active_job.is_none(), egui::Button::new("Run Position"))
                        .clicked()
                    {
                        self.start_run_position_job();
                    }
                    if ui
                        .add_enabled(
                            self.active_job.is_none() && experiment_supports_batch,
                            egui::Button::new("Run Experiment"),
                        )
                        .clicked()
                    {
                        self.start_run_experiment_job();
                    }
                });

                ui.separator();
                ui.label(RichText::new("Measurements").strong());
                ui.label(format!(
                    "Selected segmentation: {}",
                    self.persisted
                        .selected_segmentation_endname
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
                ));

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            self.active_job.is_none() && has_position_segmentations,
                            egui::Button::new("Measure Position"),
                        )
                        .clicked()
                    {
                        self.start_measure_position_job();
                    }
                    if ui
                        .add_enabled(
                            self.active_job.is_none()
                                && has_position_segmentations
                                && experiment_supports_batch,
                            egui::Button::new("Measure Experiment"),
                        )
                        .clicked()
                    {
                        self.start_measure_experiment_job();
                    }
                });

                ui.separator();
                ui.label(RichText::new("Logs").strong());
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for line in &self.logs {
                            ui.monospace(line);
                        }
                    });
            });
    }

    fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(position) = self.selected_position().cloned() else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an experiment or position to start.");
                });
                return;
            };
            let size_t = position.spec.size_t;
            let size_z = position.spec.size_z;

            ui.horizontal(|ui| {
                ui.label(RichText::new("Viewer").strong());
                ui.separator();
                ui.label(format!(
                    "{}",
                    position
                        .spec
                        .position_dir
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Position")
                ));
            });

            let mut frame_changed = false;
            if size_t > 1 {
                frame_changed = ui
                    .add(
                        egui::Slider::new(
                            &mut self.selected_frame_idx,
                            0..=size_t.saturating_sub(1),
                        )
                        .text("Frame"),
                    )
                    .changed();
            }

            if size_z > 1 {
                ui.horizontal(|ui| {
                    let max_changed = ui
                        .radio_value(
                            &mut self.persisted.projection_mode,
                            ProjectionMode::Max,
                            "Max projection",
                        )
                        .changed();
                    let z_changed = ui
                        .radio_value(
                            &mut self.persisted.projection_mode,
                            ProjectionMode::ZSlice,
                            "Z slice",
                        )
                        .changed();
                    if max_changed || z_changed {
                        self.invalidate_texture();
                    }
                });
                if matches!(self.persisted.projection_mode, ProjectionMode::ZSlice)
                    && ui
                        .add(
                            egui::Slider::new(
                                &mut self.persisted.z_index,
                                0..=size_z.saturating_sub(1),
                            )
                            .text("Z"),
                        )
                        .changed()
                {
                    self.invalidate_texture();
                }
            }

            if frame_changed {
                self.invalidate_texture();
            }

            self.refresh_texture_if_needed(ctx);

            if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
            }

            if let Some(texture) = self.texture.as_ref() {
                let available = ui.available_size();
                let image_size = texture.size_vec2();
                let scale = (available.x / image_size.x)
                    .min(available.y / image_size.y)
                    .max(1.0);
                let desired_size = image_size * scale;
                ui.centered_and_justified(|ui| {
                    ui.add(
                        egui::Image::from_texture(texture)
                            .fit_to_exact_size(desired_size)
                            .sense(egui::Sense::hover()),
                    );
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No image available for the current selection.");
                });
            }
        });
    }
}

impl eframe::App for CellAcdcGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(&self.persisted) {
            storage.set_string(APP_KEY, json);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_active_job();
        if self.active_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        self.draw_top_bar(ctx);
        self.draw_left_panel(ctx);
        self.draw_right_panel(ctx);
        self.draw_central_panel(ctx);
    }
}

fn combo_for_channel(
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
