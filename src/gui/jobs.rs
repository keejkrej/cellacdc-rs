use super::app::CellAcdcGui;
use super::state::ResolutionLayoutChoice;
use crate::gui::workspaces::{format_utility_summary, parse_optional_usize};
use anyhow::{bail, Result};
use cellacdc_rs::{
    combine_channels, connect_3d_segm, count_objects, fill_holes, measure_experiment,
    measure_position, open_position_session, resolve_position, run_experiment, run_position,
    stack_2d_segm_to_3d, CombineChannelsConfig, Connect3DSegmConfig, CountObjectsConfig,
    ExperimentRunConfig, FillHolesConfig, MaskPathResolution, MeasurementExperimentConfig,
    MeasurementRunConfig, SegmentationLayout, Stack2DSegmTo3DConfig, TrackingConfig,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub(crate) fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct JobRequest {
    pub label: String,
}

#[derive(Debug, Clone)]
pub enum JobUpdate {
    Log(String),
    Finished(Result<JobSummary, String>),
}

#[derive(Debug, Clone)]
pub struct JobSummary {
    pub summary: String,
    pub reload_session: bool,
    pub select_segmentation_endname: Option<Option<String>>,
}

pub(crate) struct JobHandle {
    pub(crate) receiver: Receiver<JobUpdate>,
    pub(crate) label: String,
    cancellation: CancellationToken,
}

impl JobHandle {
    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl CellAcdcGui {
    pub(crate) fn poll_active_job(&mut self) {
        let mut events = Vec::new();
        if let Some(job) = self.active_job.as_mut() {
            while let Ok(event) = job.receiver.try_recv() {
                events.push(event);
            }
        }
        let mut finished = false;
        for event in events {
            match event {
                JobUpdate::Log(message) => self.append_log(message),
                JobUpdate::Finished(result) => {
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

    pub(crate) fn start_job<F>(&mut self, request: JobRequest, work: F)
    where
        F: FnOnce(mpsc::Sender<JobUpdate>, CancellationToken) -> Result<JobSummary>
            + Send
            + 'static,
    {
        if self.active_job.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let cancellation = CancellationToken::new();
        self.append_log(format!("Queued job: {}", request.label));
        let token = cancellation.clone();
        std::thread::spawn(move || {
            let outcome = work(sender.clone(), token).map_err(|err| err.to_string());
            let _ = sender.send(JobUpdate::Finished(outcome));
        });
        self.active_job = Some(JobHandle {
            receiver,
            label: request.label,
            cancellation,
        });
    }

    pub(crate) fn tracking_config(&self) -> Option<TrackingConfig> {
        self.persisted.track.then(|| TrackingConfig {
            ioa_threshold: self.persisted.track_ioa_threshold,
        })
    }

    pub(crate) fn build_utility_resolution(&self) -> Result<Option<MaskPathResolution>> {
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

    pub(crate) fn utility_scope_parts(
        &self,
        scope_path: &Path,
    ) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
        if scope_path.as_os_str().is_empty() {
            bail!("Scope path is required");
        }
        match self.persisted.utility.scope_mode {
            super::state::UtilityScopeMode::Position => Ok((Some(scope_path.to_path_buf()), None)),
            super::state::UtilityScopeMode::Experiment => {
                Ok((None, Some(scope_path.to_path_buf())))
            }
            super::state::UtilityScopeMode::Auto => {
                if open_position_session(scope_path).is_ok() {
                    Ok((Some(scope_path.to_path_buf()), None))
                } else {
                    Ok((None, Some(scope_path.to_path_buf())))
                }
            }
        }
    }

    pub(crate) fn scope_touches_current_session(&self, scope_path: &Path) -> bool {
        self.experiment
            .as_ref()
            .map(|experiment| {
                scope_path.starts_with(&experiment.root_path)
                    || experiment.root_path.starts_with(scope_path)
            })
            .unwrap_or(false)
    }

    pub(crate) fn start_measure_position_job(&mut self) {
        let Some(position) = self.selected_position() else {
            return;
        };
        let position_path = position.spec.position_dir.clone();
        let segm_endname = self.selected_measurement_suffix();
        let overwrite_policy = self.overwrite_policy();
        let label = format!("Measure {}", position_path.display());
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Measuring {}",
                position_path.display()
            )));
            let result = measure_position(MeasurementRunConfig {
                position_path,
                segm_endname,
                overwrite_policy,
            })?;
            Ok(JobSummary {
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

    pub(crate) fn start_measure_experiment_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Measuring experiment {}",
                experiment_dir.display()
            )));
            let results = measure_experiment(MeasurementExperimentConfig {
                experiment_dir,
                segm_endname,
                overwrite_policy,
            })?;
            Ok(JobSummary {
                summary: format!("Measured {} position(s)", results.len()),
                reload_session: true,
                select_segmentation_endname: None,
            })
        });
    }

    pub(crate) fn start_run_position_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            let _ = sender.send(JobUpdate::Log(format!(
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
            Ok(JobSummary {
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

    pub(crate) fn start_run_experiment_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            let _ = sender.send(JobUpdate::Log(format!(
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
            Ok(JobSummary {
                summary: format!("Segmented {} position(s)", results.len()),
                reload_session: true,
                select_segmentation_endname: Some(segm_endname),
            })
        });
    }

    pub(crate) fn start_count_objects_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Counting objects in {}",
                segmentation_path.display()
            )));
            let result = count_objects(CountObjectsConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobSummary {
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

    pub(crate) fn start_fill_holes_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Filling holes in {}",
                segmentation_path.display()
            )));
            let result = fill_holes(FillHolesConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobSummary {
                summary: format_utility_summary("Filled holes", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    pub(crate) fn start_connect_3d_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Connecting 3D segmentation {}",
                segmentation_path.display()
            )));
            let result = connect_3d_segm(Connect3DSegmConfig {
                segmentation_path,
                output_path,
                resolution,
            })?;
            Ok(JobSummary {
                summary: format_utility_summary("Connected 3D segmentation", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    pub(crate) fn start_stack_2d_to_3d_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            if segmentation_path.as_os_str().is_empty() {
                bail!("Segmentation path is required");
            }
            if output_path.as_os_str().is_empty() {
                bail!("Output path is required");
            }
            if size_z == 0 {
                bail!("Target size_z must be greater than 0");
            }
            let _ = sender.send(JobUpdate::Log(format!(
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
            Ok(JobSummary {
                summary: format_utility_summary("Stacked 2D segmentation to 3D", &result),
                reload_session: false,
                select_segmentation_endname: None,
            })
        });
    }

    pub(crate) fn start_combine_channels_job(&mut self) {
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
        self.start_job(JobRequest { label }, move |sender, token| {
            if token.is_cancelled() {
                bail!("Job cancelled before start");
            }
            if scope_path.as_os_str().is_empty() {
                bail!("Scope path is required");
            }
            if recipe_path.as_os_str().is_empty() {
                bail!("Recipe path is required");
            }
            if append_name.is_empty() {
                bail!("Append name is required");
            }
            let _ = sender.send(JobUpdate::Log(format!(
                "Combining channels in {}",
                scope_path.display()
            )));
            let result = combine_channels(CombineChannelsConfig {
                position_dir,
                experiment_dir,
                recipe_path,
                append_name: append_name.clone(),
            })?;
            Ok(JobSummary {
                summary: format!("Created {} combined output(s)", result.output_paths.len()),
                reload_session,
                select_segmentation_endname: None,
            })
        });
    }
}
