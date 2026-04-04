use anyhow::{anyhow, bail, Result};
use cellacdc_rs::{
    combine_channels, connect_3d_segm, count_objects, fill_holes, measure_experiment,
    measure_position, open_experiment_session, open_position_session, resolve_position,
    run_experiment, run_position, stack_2d_segm_to_3d, CombineChannelsConfig, Connect3DSegmConfig,
    CountObjectsConfig, ExperimentRunConfig, ExperimentSession, FillHolesConfig, FrameData,
    FrameProjection, MaskPathResolution, MeasurementExperimentConfig, MeasurementRunConfig,
    OverwritePolicy, PositionSession, SegmentationLayout, SegmentationParams,
    Stack2DSegmTo3DConfig, TrackingConfig, UtilityOutputPaths,
};
use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

const APP_KEY: &str = "cellacdc_rs_gui";
const MAX_LOG_LINES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ProjectionMode {
    Max,
    ZSlice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum AppRoute {
    Launcher,
    Viewer,
    Utilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum UtilityTool {
    CountObjects,
    FillHoles,
    Connect3d,
    Stack2dTo3d,
    CombineChannels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum UtilityScopeMode {
    Auto,
    Position,
    Experiment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ResolutionLayoutChoice {
    Auto,
    Yx,
    Tyx,
    Zyx,
    Tzyx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct UtilityState {
    selected_tool: UtilityTool,
    segmentation_path: String,
    output_path: String,
    scope_path: String,
    recipe_path: String,
    append_name: String,
    resolution_size_t: String,
    resolution_size_z: String,
    resolution_layout: ResolutionLayoutChoice,
    stack_target_size_z: usize,
    scope_mode: UtilityScopeMode,
}

impl Default for UtilityState {
    fn default() -> Self {
        Self {
            selected_tool: UtilityTool::CountObjects,
            segmentation_path: String::new(),
            output_path: String::new(),
            scope_path: String::new(),
            recipe_path: String::new(),
            append_name: "combined".to_string(),
            resolution_size_t: String::new(),
            resolution_size_z: String::new(),
            resolution_layout: ResolutionLayoutChoice::Auto,
            stack_target_size_z: 3,
            scope_mode: UtilityScopeMode::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct PersistedState {
    route: AppRoute,
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
    utility: UtilityState,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            route: AppRoute::Launcher,
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
            utility: UtilityState::default(),
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
        self.persisted.route = AppRoute::Viewer;
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

    fn selected_segmentation_path(&self) -> Option<PathBuf> {
        let position = self.selected_position()?;
        position
            .segmentations
            .iter()
            .find(|asset| asset.endname == self.persisted.selected_segmentation_endname)
            .map(|asset| asset.path.clone())
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

    fn start_job<F>(&mut self, label: String, work: F)
    where
        F: FnOnce(mpsc::Sender<JobEvent>) -> Result<JobResult> + Send + 'static,
    {
        if self.active_job.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        self.append_log(format!("Queued job: {label}"));
        std::thread::spawn(move || {
            let outcome = work(sender.clone()).map_err(|err| err.to_string());
            let _ = sender.send(JobEvent::Finished(outcome));
        });
        self.active_job = Some(BackgroundJob { receiver, label });
    }

    fn start_measure_position_job(&mut self) {
        let Some(position) = self.selected_position() else {
            return;
        };
        let position_path = position.spec.position_dir.clone();
        let segm_endname = self.selected_measurement_suffix();
        let overwrite_policy = self.overwrite_policy();
        let label = format!("Measure {}", position_path.display());
        self.start_job(label, move |sender| {
            let _ = sender.send(JobEvent::Log(format!(
                "Measuring {}",
                position_path.display()
            )));
            let result = measure_position(MeasurementRunConfig {
                position_path,
                segm_endname,
                overwrite_policy,
            })?;
            Ok(JobResult {
                summary: format!(
                    "Measured {} frame(s) -> {}",
                    result.frames_processed,
                    result.outputs.acdc_output_csv_path.display()
                ),
                reload_session: true,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_measure_experiment_job(&mut self) {
        let Some(experiment) = self.experiment.as_ref() else {
            return;
        };
        if experiment.is_single_position {
            return;
        }
        let experiment_dir = experiment.root_path.clone();
        let segm_endname = self.selected_measurement_suffix();
        let overwrite_policy = self.overwrite_policy();
        let label = format!("Measure experiment {}", experiment_dir.display());
        self.start_job(label, move |sender| {
            let _ = sender.send(JobEvent::Log(format!(
                "Measuring experiment {}",
                experiment_dir.display()
            )));
            let results = measure_experiment(MeasurementExperimentConfig {
                experiment_dir,
                segm_endname,
                overwrite_policy,
            })?;
            Ok(JobResult {
                summary: format!("Measured {} position(s)", results.len()),
                reload_session: true,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_run_position_job(&mut self) {
        let Some(position) = self.selected_position() else {
            return;
        };
        let position_path = position.spec.position_dir.clone();
        let phase_channel = self.persisted.phase_channel.clone();
        let fluo_channel = self.persisted.fluo_channel.clone();
        let model_path = PathBuf::from(self.persisted.model_path.trim());
        let segm_endname = self.run_output_suffix();
        let overwrite_policy = self.overwrite_policy();
        let cpu = self.persisted.cpu;
        let params = self.segmentation_params();
        let tracking = self.tracking_config();
        let label = format!("Segment {}", position_path.display());
        self.start_job(label, move |sender| {
            let _ = sender.send(JobEvent::Log(format!(
                "Segmenting {}",
                position_path.display()
            )));
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
        });
    }

    fn start_run_experiment_job(&mut self) {
        let Some(experiment) = self.experiment.as_ref() else {
            return;
        };
        if experiment.is_single_position {
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
        let label = format!("Segment experiment {}", experiment_dir.display());
        self.start_job(label, move |sender| {
            let _ = sender.send(JobEvent::Log(format!(
                "Segmenting experiment {}",
                experiment_dir.display()
            )));
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
        });
    }

    fn start_count_objects_job(&mut self) {
        let segmentation_path = PathBuf::from(self.persisted.utility.segmentation_path.trim());
        let output_path = PathBuf::from(self.persisted.utility.output_path.trim());
        let resolution = match self.build_utility_resolution() {
            Ok(value) => value,
            Err(err) => {
                self.last_error = Some(err.to_string());
                return;
            }
        };
        let label = format!("Count objects {}", segmentation_path.display());
        self.start_job(label, move |sender| {
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobEvent::Log(format!(
                "Counting objects in {}",
                segmentation_path.display()
            )));
            let result = count_objects(CountObjectsConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobResult {
                summary: format!(
                    "Saved object counts to {} ({} entries)",
                    result.summary.output_path.display(),
                    result.summary.counts.len()
                ),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_fill_holes_job(&mut self) {
        let segmentation_path = PathBuf::from(self.persisted.utility.segmentation_path.trim());
        let output_path = PathBuf::from(self.persisted.utility.output_path.trim());
        let resolution = match self.build_utility_resolution() {
            Ok(value) => value,
            Err(err) => {
                self.last_error = Some(err.to_string());
                return;
            }
        };
        let label = format!("Fill holes {}", segmentation_path.display());
        self.start_job(label, move |sender| {
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobEvent::Log(format!(
                "Filling holes in {}",
                segmentation_path.display()
            )));
            let result = fill_holes(FillHolesConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobResult {
                summary: format_utility_summary("Filled holes", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_connect_3d_job(&mut self) {
        let segmentation_path = PathBuf::from(self.persisted.utility.segmentation_path.trim());
        let output_path = PathBuf::from(self.persisted.utility.output_path.trim());
        let resolution = match self.build_utility_resolution() {
            Ok(value) => value,
            Err(err) => {
                self.last_error = Some(err.to_string());
                return;
            }
        };
        let label = format!("Connect 3D {}", segmentation_path.display());
        self.start_job(label, move |sender| {
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobEvent::Log(format!(
                "Connecting 3D segmentation {}",
                segmentation_path.display()
            )));
            let result = connect_3d_segm(Connect3DSegmConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobResult {
                summary: format_utility_summary("Connected 3D segmentation", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_stack_2d_to_3d_job(&mut self) {
        let segmentation_path = PathBuf::from(self.persisted.utility.segmentation_path.trim());
        let output_path = PathBuf::from(self.persisted.utility.output_path.trim());
        let resolution = match self.build_utility_resolution() {
            Ok(value) => value,
            Err(err) => {
                self.last_error = Some(err.to_string());
                return;
            }
        };
        let size_z = self.persisted.utility.stack_target_size_z;
        let label = format!("Stack 2D to 3D {}", segmentation_path.display());
        self.start_job(label, move |sender| {
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            if size_z == 0 {
                bail!("Target size_z must be greater than 0");
            }
            let _ = sender.send(JobEvent::Log(format!(
                "Stacking {} into {} z-slices",
                segmentation_path.display(),
                size_z
            )));
            let result = stack_2d_segm_to_3d(Stack2DSegmTo3DConfig {
                segmentation_path,
                output_path,
                size_z,
                resolution,
            })?;
            Ok(JobResult {
                summary: format_utility_summary("Stacked 2D segmentation to 3D", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    fn start_combine_channels_job(&mut self) {
        let scope_path = PathBuf::from(self.persisted.utility.scope_path.trim());
        let recipe_path = PathBuf::from(self.persisted.utility.recipe_path.trim());
        let append_name = self.persisted.utility.append_name.trim().to_string();
        let (position_dir, experiment_dir) = match self.utility_scope_parts(&scope_path) {
            Ok(parts) => parts,
            Err(err) => {
                self.last_error = Some(err.to_string());
                return;
            }
        };
        let reload_session = self.scope_touches_current_session(&scope_path);
        let label = format!("Combine channels {}", scope_path.display());
        self.start_job(label, move |sender| {
            if scope_path.as_os_str().is_empty() {
                bail!("Scope path is required");
            }
            if recipe_path.as_os_str().is_empty() {
                bail!("Recipe path is required");
            }
            if append_name.is_empty() {
                bail!("Append name is required");
            }
            let _ = sender.send(JobEvent::Log(format!(
                "Combining channels in {}",
                scope_path.display()
            )));
            let result = combine_channels(CombineChannelsConfig {
                position_dir,
                experiment_dir,
                recipe_path,
                append_name: append_name.clone(),
            })?;
            Ok(JobResult {
                summary: format!("Created {} combined output(s)", result.output_paths.len()),
                reload_session,
                select_segmentation_endname: None,
            })
        });
    }

    fn build_utility_resolution(&self) -> Result<Option<MaskPathResolution>> {
        let size_t = parse_optional_usize(&self.persisted.utility.resolution_size_t)?;
        let size_z = parse_optional_usize(&self.persisted.utility.resolution_size_z)?;
        let layout = match self.persisted.utility.resolution_layout {
            ResolutionLayoutChoice::Auto => None,
            ResolutionLayoutChoice::Yx => Some(SegmentationLayout::YX),
            ResolutionLayoutChoice::Tyx => Some(SegmentationLayout::TYX),
            ResolutionLayoutChoice::Zyx => Some(SegmentationLayout::ZYX),
            ResolutionLayoutChoice::Tzyx => Some(SegmentationLayout::TZYX),
        };
        if size_t.is_none() && size_z.is_none() && layout.is_none() {
            Ok(None)
        } else {
            Ok(Some(MaskPathResolution {
                size_t,
                size_z,
                layout,
            }))
        }
    }

    fn utility_scope_parts(&self, scope_path: &Path) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
        if scope_path.as_os_str().is_empty() {
            bail!("Scope path is required");
        }
        match self.persisted.utility.scope_mode {
            UtilityScopeMode::Position => Ok((Some(scope_path.to_path_buf()), None)),
            UtilityScopeMode::Experiment => Ok((None, Some(scope_path.to_path_buf()))),
            UtilityScopeMode::Auto => {
                if open_position_session(scope_path).is_ok() {
                    Ok((Some(scope_path.to_path_buf()), None))
                } else {
                    Ok((None, Some(scope_path.to_path_buf())))
                }
            }
        }
    }

    fn scope_touches_current_session(&self, scope_path: &Path) -> bool {
        self.experiment
            .as_ref()
            .map(|experiment| {
                scope_path.starts_with(&experiment.root_path)
                    || experiment.root_path.starts_with(scope_path)
            })
            .unwrap_or(false)
    }

    fn autofill_utility_from_selected_segmentation(&mut self) {
        let Some(segmentation_path) = self.selected_segmentation_path() else {
            return;
        };
        self.persisted.utility.segmentation_path = segmentation_path.display().to_string();
        if self.persisted.utility.output_path.trim().is_empty() {
            self.persisted.utility.output_path =
                suggested_output_path(self.persisted.utility.selected_tool, &segmentation_path)
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

    fn draw_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.persisted.route, AppRoute::Launcher, "Home");
                ui.selectable_value(&mut self.persisted.route, AppRoute::Viewer, "Viewer");
                ui.selectable_value(&mut self.persisted.route, AppRoute::Utilities, "Utilities");
                ui.separator();

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

    fn draw_launcher_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cell-ACDC Rust");
            ui.label(
                "This launcher is the phase-1 shell for the Rust desktop app: session review, job execution, and a utility center on top of the existing Rust core.",
            );
            ui.add_space(8.0);

            if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                ui.add_space(4.0);
            }

            if let Some(experiment) = self.experiment.as_ref() {
                ui.group(|ui| {
                    ui.label(RichText::new("Current Session").strong());
                    ui.monospace(experiment.root_path.display().to_string());
                    ui.label(format!("Positions: {}", experiment.positions.len()));
                    ui.horizontal(|ui| {
                        if ui.button("Open Viewer").clicked() {
                            self.persisted.route = AppRoute::Viewer;
                        }
                        if ui.button("Open Utilities").clicked() {
                            self.persisted.route = AppRoute::Utilities;
                        }
                    });
                });
            } else {
                ui.group(|ui| {
                    ui.label(RichText::new("Current Session").strong());
                    ui.label("No structured experiment is open.");
                    if ui.button("Open Experiment or Position").clicked() {
                        if let Some(path) = FileDialog::new().pick_folder() {
                            if let Err(err) = self.open_path(path) {
                                self.last_error = Some(err.to_string());
                            }
                        }
                    }
                });
            }

            ui.add_space(10.0);
            ui.columns(2, |columns| {
                columns[0].group(|ui| {
                    ui.label(RichText::new("Viewer + Jobs").strong());
                    ui.label(
                        "Browse positions, frames, channels, z views, segmentation overlays, and run segmentation or measurement jobs.",
                    );
                    if ui
                        .add_enabled(self.experiment.is_some(), egui::Button::new("Open Viewer"))
                        .clicked()
                    {
                        self.persisted.route = AppRoute::Viewer;
                    }
                });

                columns[1].group(|ui| {
                    ui.label(RichText::new("Utility Center").strong());
                    ui.label(
                        "Run file-based Rust utilities such as object counting, hole filling, 3D connection, 2D-to-3D stacking, and channel combination.",
                    );
                    if ui.button("Open Utilities").clicked() {
                        self.persisted.route = AppRoute::Utilities;
                    }
                });
            });

            ui.add_space(10.0);
            ui.columns(3, |columns| {
                launcher_placeholder_card(
                    &mut columns[0],
                    "Data Prep",
                    "Planned next: alignment, crop ROIs, z-slice and projection selection, and background ROI workflows.",
                );
                launcher_placeholder_card(
                    &mut columns[1],
                    "Segmentation Module",
                    "Planned next: model/tracker selection, recipes, batch scope, and segmentation run UX.",
                );
                launcher_placeholder_card(
                    &mut columns[2],
                    "Annotation Workspace",
                    "Planned later: editing, repeat tracking, autosave, recovery, lineage, and cell-cycle annotations.",
                );
            });

            ui.add_space(10.0);
            draw_logs(ui, &self.logs, 220.0);
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
                    if ui.button("Open Session").clicked() {
                        if let Some(path) = FileDialog::new().pick_folder() {
                            if let Err(err) = self.open_path(path) {
                                self.last_error = Some(err.to_string());
                            }
                        }
                    }
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
                path_edit_row(
                    ui,
                    "Model",
                    &mut self.persisted.model_path,
                    PathEditKind::PickFile,
                );
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
                    selected_segm_label(&position, &self.persisted.selected_segmentation_endname)
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
                draw_logs(ui, &self.logs, 320.0);
            });
    }

    fn draw_viewer_panel(&mut self, ctx: &egui::Context) {
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
                ui.label(
                    position
                        .spec
                        .position_dir
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Position"),
                );
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

    fn draw_utility_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Utility Center");
            ui.label("These tools call the existing Rust utility functions directly. They are intended for structured Cell-ACDC files and related masks/tables.");
            ui.add_space(8.0);

            if let Some(error) = self.last_error.clone() {
                ui.colored_label(Color32::from_rgb(200, 60, 60), error);
                ui.add_space(4.0);
            }

            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(220.0);
                    ui.label(RichText::new("Tools").strong());
                    for tool in [
                        UtilityTool::CountObjects,
                        UtilityTool::FillHoles,
                        UtilityTool::Connect3d,
                        UtilityTool::Stack2dTo3d,
                        UtilityTool::CombineChannels,
                    ] {
                        if ui
                            .selectable_label(
                                self.persisted.utility.selected_tool == tool,
                                utility_tool_label(tool),
                            )
                            .clicked()
                        {
                            self.persisted.utility.selected_tool = tool;
                            if self.persisted.utility.output_path.trim().is_empty() {
                                if let Some(path) = self.selected_segmentation_path() {
                                    self.persisted.utility.output_path =
                                        suggested_output_path(tool, &path).display().to_string();
                                }
                            }
                        }
                    }

                    ui.add_space(12.0);
                    ui.label(RichText::new("Quick Fill").strong());
                    if ui
                        .add_enabled(
                            self.selected_segmentation_path().is_some(),
                            egui::Button::new("Use Selected Segmentation"),
                        )
                        .clicked()
                    {
                        self.autofill_utility_from_selected_segmentation();
                    }
                    if ui
                        .add_enabled(
                            self.selected_position().is_some(),
                            egui::Button::new("Use Selected Position Scope"),
                        )
                        .clicked()
                    {
                        if let Some(position) = self.selected_position() {
                            self.persisted.utility.scope_path =
                                position.spec.position_dir.display().to_string();
                            self.persisted.utility.scope_mode = UtilityScopeMode::Position;
                        }
                    }
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.set_width(ui.available_width());
                    match self.persisted.utility.selected_tool {
                        UtilityTool::CountObjects => self.draw_count_objects_form(ui),
                        UtilityTool::FillHoles => self.draw_fill_holes_form(ui),
                        UtilityTool::Connect3d => self.draw_connect_3d_form(ui),
                        UtilityTool::Stack2dTo3d => self.draw_stack_2d_to_3d_form(ui),
                        UtilityTool::CombineChannels => self.draw_combine_channels_form(ui),
                    }
                });
            });

            ui.add_space(10.0);
            draw_logs(ui, &self.logs, 240.0);
        });
    }

    fn draw_count_objects_form(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Count Objects").strong());
        ui.label("Count labelled objects in a segmentation file and write a summary table.");
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest Output Path").clicked() {
            let input = PathBuf::from(self.persisted.utility.segmentation_path.trim());
            if !input.as_os_str().is_empty() {
                self.persisted.utility.output_path =
                    suggested_output_path(UtilityTool::CountObjects, &input)
                        .display()
                        .to_string();
            }
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

    fn draw_fill_holes_form(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Fill Holes").strong());
        ui.label("Fill holes inside labelled objects and save a new segmentation file.");
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest Output Path").clicked() {
            let input = PathBuf::from(self.persisted.utility.segmentation_path.trim());
            if !input.as_os_str().is_empty() {
                self.persisted.utility.output_path =
                    suggested_output_path(UtilityTool::FillHoles, &input)
                        .display()
                        .to_string();
            }
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

    fn draw_connect_3d_form(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Connect 3D Segmentation").strong());
        ui.label("Reconnect labels across z boundaries for ZYX or TZYX segmentation data.");
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest Output Path").clicked() {
            let input = PathBuf::from(self.persisted.utility.segmentation_path.trim());
            if !input.as_os_str().is_empty() {
                self.persisted.utility.output_path =
                    suggested_output_path(UtilityTool::Connect3d, &input)
                        .display()
                        .to_string();
            }
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

    fn draw_stack_2d_to_3d_form(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Stack 2D Segmentation to 3D").strong());
        ui.label("Replicate 2D labels into a ZYX or TZYX segmentation with the requested z-depth.");
        path_edit_row(
            ui,
            "Segmentation",
            &mut self.persisted.utility.segmentation_path,
            PathEditKind::PickFile,
        );
        path_edit_row(
            ui,
            "Output",
            &mut self.persisted.utility.output_path,
            PathEditKind::SaveFile,
        );
        if ui.button("Suggest Output Path").clicked() {
            let input = PathBuf::from(self.persisted.utility.segmentation_path.trim());
            if !input.as_os_str().is_empty() {
                self.persisted.utility.output_path =
                    suggested_output_path(UtilityTool::Stack2dTo3d, &input)
                        .display()
                        .to_string();
            }
        }
        ui.add(
            egui::DragValue::new(&mut self.persisted.utility.stack_target_size_z)
                .range(1..=1024)
                .prefix("target z "),
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

    fn draw_combine_channels_form(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Combine Channels").strong());
        ui.label("Apply a Rust combine-channels recipe to one position or an entire experiment.");
        path_edit_row(
            ui,
            "Scope",
            &mut self.persisted.utility.scope_path,
            PathEditKind::PickFolder,
        );
        egui::ComboBox::from_label("Scope mode")
            .selected_text(match self.persisted.utility.scope_mode {
                UtilityScopeMode::Auto => "Auto detect",
                UtilityScopeMode::Position => "Position",
                UtilityScopeMode::Experiment => "Experiment",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    UtilityScopeMode::Auto,
                    "Auto detect",
                );
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    UtilityScopeMode::Position,
                    "Position",
                );
                ui.selectable_value(
                    &mut self.persisted.utility.scope_mode,
                    UtilityScopeMode::Experiment,
                    "Experiment",
                );
            });
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

    fn draw_resolution_controls(&mut self, ui: &mut egui::Ui) {
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
        match self.persisted.route {
            AppRoute::Launcher => self.draw_launcher_panel(ctx),
            AppRoute::Viewer => {
                self.draw_left_panel(ctx);
                self.draw_right_panel(ctx);
                self.draw_viewer_panel(ctx);
            }
            AppRoute::Utilities => self.draw_utility_panel(ctx),
        }
    }
}

#[derive(Clone, Copy)]
enum PathEditKind {
    PickFile,
    PickFolder,
    SaveFile,
}

fn launcher_placeholder_card(ui: &mut egui::Ui, title: &str, body: &str) {
    ui.group(|ui| {
        ui.label(RichText::new(title).strong());
        ui.label(body);
        ui.add_enabled(false, egui::Button::new("Planned"));
    });
}

fn path_edit_row(ui: &mut egui::Ui, label: &str, value: &mut String, kind: PathEditKind) {
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

fn draw_logs(ui: &mut egui::Ui, logs: &[String], max_height: f32) {
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

fn parse_optional_usize(value: &str) -> Result<Option<usize>> {
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

fn suggested_output_path(tool: UtilityTool, input_path: &Path) -> PathBuf {
    match tool {
        UtilityTool::CountObjects => input_path.with_extension("csv"),
        UtilityTool::FillHoles => append_to_stem(input_path, "_filled"),
        UtilityTool::Connect3d => append_to_stem(input_path, "_connected3d"),
        UtilityTool::Stack2dTo3d => append_to_stem(input_path, "_stacked3d"),
        UtilityTool::CombineChannels => input_path.to_path_buf(),
    }
}

fn append_to_stem(path: &Path, suffix: &str) -> PathBuf {
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

fn utility_tool_label(tool: UtilityTool) -> &'static str {
    match tool {
        UtilityTool::CountObjects => "Count Objects",
        UtilityTool::FillHoles => "Fill Holes",
        UtilityTool::Connect3d => "Connect 3D Segm",
        UtilityTool::Stack2dTo3d => "Stack 2D to 3D",
        UtilityTool::CombineChannels => "Combine Channels",
    }
}

fn selected_segm_label(position: &PositionSession, endname: &Option<String>) -> String {
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

fn format_utility_summary(action: &str, result: &UtilityOutputPaths) -> String {
    let mut summary = format!("{action} -> {}", result.primary_path.display());
    if !result.secondary_paths.is_empty() {
        summary.push_str(&format!(
            " (+{} sidecar file(s))",
            result.secondary_paths.len()
        ));
    }
    summary
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
