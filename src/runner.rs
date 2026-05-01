use anyhow::{bail, Context, Result};
use chrono::Local;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
use csv::Reader;
#[cfg(test)]
use ndarray::Array2;
#[cfg(test)]
use ndarray::Array3;
#[cfg(test)]
use ndarray::Array4;
#[cfg(test)]
use ndarray_npy::{NpzReader, NpzWriter};
#[cfg(test)]
use std::fs::File;

use crate::image_io::{load_image_stack_as_f32, load_image_volume_as_f32, write_mask_npz};
use crate::layout::{discover_experiment, resolve_position, ExperimentSpec, PositionSpec};
use crate::measure::{
    load_measurement_inputs, measurement_position_from_position, write_measurements,
};
use crate::metadata::ensure_position_metadata;
use crate::model::{CellposeModel, Segmenter};
use crate::prep::{load_crop_roi_coords_csv, read_freehand_roi_npz, CropRoiRect, FreehandRoiMask};
use crate::segm_info::load_segm_info;
use crate::tracking::{track_sequence, OverlapDenominator, TrackingConfig};
use crate::zstack::project_frame_f32;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationParams {
    pub tile: usize,
    pub batch_size: usize,
    pub cellprob_threshold: f32,
    pub niter: usize,
    pub min_size: usize,
}

impl Default for SegmentationParams {
    fn default() -> Self {
        Self {
            tile: 256,
            batch_size: 1,
            cellprob_threshold: 0.0,
            niter: 200,
            min_size: 15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverwritePolicy {
    Refuse,
    Overwrite,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostprocessConfig {
    pub min_area: Option<usize>,
    pub min_solidity: Option<f64>,
    pub max_elongation: Option<f64>,
    pub min_obj_no_zslices: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessStep {
    GaussianFilter {
        sigma_y: f32,
        sigma_x: f32,
    },
    RemoveHotPixels,
    SpotDetectorFilter {
        radius_y: f32,
        radius_x: f32,
    },
    RidgeFilter {
        sigmas: Vec<f32>,
    },
    EnhanceSpeckles {
        radius: usize,
    },
    CorrectIllumination {
        block_size: usize,
        approximate_object_diameter: f32,
        apply_gaussian_filter: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationRunConfig {
    pub position: PositionSpec,
    pub model_path: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub cpu: bool,
    pub params: SegmentationParams,
    pub preprocess_steps: Vec<PreprocessStep>,
    pub tracking: Option<TrackingConfig>,
    pub postprocess: Option<PostprocessConfig>,
    pub stop_frame: Option<usize>,
    pub save_outputs: bool,
    pub use_data_prep_roi: bool,
    pub use_data_prep_free_roi: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentRunConfig {
    pub experiment_dir: PathBuf,
    pub phase_channel: String,
    pub fluo_channel: String,
    pub model_path: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub cpu: bool,
    pub params: SegmentationParams,
    pub preprocess_steps: Vec<PreprocessStep>,
    pub tracking: Option<TrackingConfig>,
    pub postprocess: Option<PostprocessConfig>,
    pub stop_frame: Option<usize>,
    pub save_outputs: bool,
    pub use_data_prep_roi: bool,
    pub use_data_prep_free_roi: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutputPaths {
    pub segm_npz_path: PathBuf,
    pub acdc_output_csv_path: PathBuf,
    pub segm_hyperparams_ini_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub outputs: RunOutputPaths,
    pub labels_found: u32,
    pub frames_processed: usize,
}

pub fn run_position(config: SegmentationRunConfig) -> Result<RunResult> {
    let mut model = CellposeModel::new(&config.model_path, config.cpu)?;
    run_position_with_segmenter(config, &mut model)
}

pub fn run_experiment(config: ExperimentRunConfig) -> Result<Vec<RunResult>> {
    let experiment = discover_experiment(
        &config.experiment_dir,
        config.phase_channel.clone(),
        config.fluo_channel.clone(),
    )?;
    let mut model = CellposeModel::new(&config.model_path, config.cpu)?;
    run_experiment_with_segmenter(config, experiment, &mut model)
}

pub fn run_experiment_with_segmenter(
    config: ExperimentRunConfig,
    experiment: ExperimentSpec,
    segmenter: &mut impl Segmenter,
) -> Result<Vec<RunResult>> {
    let mut results = Vec::with_capacity(experiment.positions.len());
    for position in experiment.positions {
        let run_config = SegmentationRunConfig {
            position,
            model_path: config.model_path.clone(),
            segm_endname: config.segm_endname.clone(),
            overwrite_policy: config.overwrite_policy,
            cpu: config.cpu,
            params: config.params.clone(),
            preprocess_steps: config.preprocess_steps.clone(),
            tracking: config.tracking.clone(),
            postprocess: config.postprocess.clone(),
            stop_frame: config.stop_frame,
            save_outputs: config.save_outputs,
            use_data_prep_roi: config.use_data_prep_roi,
            use_data_prep_free_roi: config.use_data_prep_free_roi,
        };
        results.push(run_position_with_segmenter(run_config, segmenter)?);
    }
    Ok(results)
}

pub fn run_position_with_segmenter(
    config: SegmentationRunConfig,
    segmenter: &mut impl Segmenter,
) -> Result<RunResult> {
    let outputs = output_paths(
        &config.position.images_dir,
        &config.position.basename,
        config.segm_endname.as_deref(),
    );
    if config.save_outputs {
        guard_outputs(&outputs, config.overwrite_policy)?;
    }
    let frames_to_process = resolve_stop_frame_count(config.position.size_t, config.stop_frame)?;
    let mut raw_frame_masks = Vec::with_capacity(config.position.size_t);
    let (frame_height, frame_width) = if config.position.size_z > 1 {
        let (phase, phase_shape) = load_image_volume_as_f32(
            &config.position.phase_image,
            Some(config.position.size_t),
            Some(config.position.size_z),
        )
        .with_context(|| {
            format!(
                "Failed to load phase image {}",
                config.position.phase_image.display()
            )
        })?;
        let (fluo, fluo_shape) = load_image_volume_as_f32(
            &config.position.fluo_image,
            Some(config.position.size_t),
            Some(config.position.size_z),
        )
        .with_context(|| {
            format!(
                "Failed to load fluorescence image {}",
                config.position.fluo_image.display()
            )
        })?;
        if phase_shape != fluo_shape {
            bail!(
                "Input z-stack mismatch: phase is {}x{}x{}x{}, fluorescence is {}x{}x{}x{}",
                phase_shape.size_t,
                phase_shape.size_z,
                phase_shape.height,
                phase_shape.width,
                fluo_shape.size_t,
                fluo_shape.size_z,
                fluo_shape.height,
                fluo_shape.width
            );
        }
        let segm_info_path = config.position.segm_info_path.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "Position {} is a z-stack but no _segmInfo.csv was found. Run prepare-zstack-segm-info first.",
                config.position.position_dir.display()
            )
        })?;
        let segm_info = load_segm_info(segm_info_path)?;
        let phase_filename = config
            .position
            .phase_image
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid phase filename"))?;
        let fluo_filename = config
            .position
            .fluo_image
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid fluorescence filename"))?;
        for frame_i in 0..frames_to_process {
            let phase_record = segm_info.get(phase_filename, frame_i).ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing _segmInfo entry for {:?} frame {}",
                    phase_filename,
                    frame_i
                )
            })?;
            let fluo_record = segm_info.get(fluo_filename, frame_i).ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing _segmInfo entry for {:?} frame {}",
                    fluo_filename,
                    frame_i
                )
            })?;
            let phase_frame = project_frame_f32(
                &phase,
                phase_shape,
                frame_i,
                phase_record.z_slice_used_data_prep,
                phase_record.which_z_proj,
            )?;
            let fluo_frame = project_frame_f32(
                &fluo,
                fluo_shape,
                frame_i,
                fluo_record.z_slice_used_data_prep,
                fluo_record.which_z_proj,
            )?;
            let phase_frame = apply_preprocess_steps(
                phase_frame,
                phase_shape.height,
                phase_shape.width,
                &config.preprocess_steps,
            )?;
            let fluo_frame = apply_preprocess_steps(
                fluo_frame,
                fluo_shape.height,
                fluo_shape.width,
                &config.preprocess_steps,
            )?;
            raw_frame_masks.push(segmenter.segment_pair(
                phase_frame,
                fluo_frame,
                phase_shape.height,
                phase_shape.width,
                &config.params,
            )?);
        }
        for _ in frames_to_process..phase_shape.size_t {
            raw_frame_masks.push(vec![0; phase_shape.height * phase_shape.width]);
        }
        (phase_shape.height, phase_shape.width)
    } else {
        let (phase, phase_shape) = load_image_stack_as_f32(&config.position.phase_image)
            .with_context(|| {
                format!(
                    "Failed to load phase image {}",
                    config.position.phase_image.display()
                )
            })?;
        let (fluo, fluo_shape) = load_image_stack_as_f32(&config.position.fluo_image)
            .with_context(|| {
                format!(
                    "Failed to load fluorescence image {}",
                    config.position.fluo_image.display()
                )
            })?;

        if phase_shape != fluo_shape {
            bail!(
                "Input image stack mismatch: phase is {} frame(s) of {}x{}, fluorescence is {} frame(s) of {}x{}",
                phase_shape.frames,
                phase_shape.width,
                phase_shape.height,
                fluo_shape.frames,
                fluo_shape.width,
                fluo_shape.height
            );
        }

        if config.position.size_t != phase_shape.frames {
            bail!(
                "Resolved metadata SizeT ({}) does not match image frame count ({}) for {}",
                config.position.size_t,
                phase_shape.frames,
                config.position.phase_image.display()
            );
        }

        let frame_len = phase_shape.height * phase_shape.width;
        for frame_i in 0..frames_to_process {
            let start = frame_i * frame_len;
            let end = start + frame_len;
            let phase_frame = apply_preprocess_steps(
                phase[start..end].to_vec(),
                phase_shape.height,
                phase_shape.width,
                &config.preprocess_steps,
            )?;
            let fluo_frame = apply_preprocess_steps(
                fluo[start..end].to_vec(),
                fluo_shape.height,
                fluo_shape.width,
                &config.preprocess_steps,
            )?;
            let mask = segmenter.segment_pair(
                phase_frame,
                fluo_frame,
                phase_shape.height,
                phase_shape.width,
                &config.params,
            )?;
            raw_frame_masks.push(mask);
        }
        for _ in frames_to_process..phase_shape.frames {
            raw_frame_masks.push(vec![0; frame_len]);
        }
        (phase_shape.height, phase_shape.width)
    };

    if config.use_data_prep_roi {
        if let Some(path) = config.position.data_prep_roi_coords_path.as_deref() {
            let table = load_crop_roi_coords_csv(path)
                .with_context(|| format!("Failed to load data-prep crop ROI {}", path.display()))?;
            if let Some(roi) = active_data_prep_crop_roi(&table.rois, &table.cropped_roi_ids) {
                apply_data_prep_crop_roi_filter(
                    &mut raw_frame_masks,
                    frame_height,
                    frame_width,
                    roi,
                )?;
            }
        }
    }

    if config.use_data_prep_free_roi {
        if let Some(path) = config.position.data_prep_free_roi_path.as_deref() {
            if let Some(roi) = read_freehand_roi_npz(path)
                .with_context(|| format!("Failed to load data-prep free ROI {}", path.display()))?
            {
                apply_data_prep_free_roi_filter(
                    &mut raw_frame_masks,
                    frame_height,
                    frame_width,
                    &roi,
                )?;
            }
        }
    }

    if let Some(postprocess) = &config.postprocess {
        apply_standard_postprocess(&mut raw_frame_masks, frame_height, frame_width, postprocess)?;
    }

    let (tracked_frames, labels_found) = if let Some(tracking) = &config.tracking {
        let tracked = track_sequence(&raw_frame_masks, frame_height, frame_width, tracking);
        (tracked.frames, tracked.labels_found)
    } else {
        let labels_found = raw_frame_masks
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .max()
            .unwrap_or(0);
        (raw_frame_masks, labels_found)
    };

    let (labels_found, frames_processed) = if config.save_outputs {
        let masks = tracked_frames.into_iter().flatten().collect::<Vec<_>>();
        let segm_name = segmentation_name(config.segm_endname.as_deref());
        ensure_position_metadata(
            config.position.metadata_path.as_deref(),
            &config.position.images_dir,
            &config.position.basename,
            &config.position.phase_channel,
            &config.position.fluo_channel,
            config.position.size_t,
            config.position.size_z,
            frame_height,
            frame_width,
            config.position.time_increment,
            config.position.physical_size_z,
            config.position.physical_size_y,
            config.position.physical_size_x,
            &segm_name,
            false,
        )?;
        write_mask_npz(
            &outputs.segm_npz_path,
            &masks,
            config.position.size_t,
            frame_height,
            frame_width,
        )?;
        write_hyperparams_ini(
            &outputs.segm_hyperparams_ini_path,
            &config.position,
            &config.model_path,
            &config.params,
            config.cpu,
            config.segm_endname.as_deref(),
            config.tracking.as_ref(),
            &outputs.segm_npz_path,
        )?;

        let measurement_result = write_measurements(
            &load_measurement_inputs(
                measurement_position_from_position(&config.position),
                config.segm_endname.as_deref(),
                config.stop_frame,
                None,
            )?,
            None,
            false,
        )?;
        (
            measurement_result.labels_found.max(labels_found),
            frames_to_process.min(measurement_result.frames_processed),
        )
    } else {
        (labels_found, frames_to_process)
    };

    Ok(RunResult {
        position_dir: config.position.position_dir,
        images_dir: config.position.images_dir,
        outputs,
        labels_found,
        frames_processed,
    })
}

pub fn resolve_position_run_config(
    position_path: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
    model_path: impl Into<PathBuf>,
    segm_endname: Option<String>,
    overwrite_policy: OverwritePolicy,
    cpu: bool,
    params: SegmentationParams,
) -> Result<SegmentationRunConfig> {
    let position = resolve_position(position_path, phase_channel, fluo_channel)?;
    Ok(SegmentationRunConfig {
        position,
        model_path: model_path.into(),
        segm_endname,
        overwrite_policy,
        cpu,
        params,
        preprocess_steps: Vec::new(),
        tracking: None,
        postprocess: None,
        stop_frame: None,
        save_outputs: true,
        use_data_prep_roi: true,
        use_data_prep_free_roi: true,
    })
}

fn apply_standard_postprocess(
    masks: &mut [Vec<u32>],
    height: usize,
    width: usize,
    config: &PostprocessConfig,
) -> Result<()> {
    if let Some(min_area) = config.min_area {
        for mask in masks.iter_mut() {
            remove_labels_below_area(mask, min_area);
        }
    }
    if let Some(min_solidity) = config.min_solidity {
        for mask in masks.iter_mut() {
            remove_labels_below_solidity(mask, height, width, min_solidity)?;
        }
    }
    if let Some(max_elongation) = config.max_elongation {
        for mask in masks.iter_mut() {
            remove_labels_above_elongation(mask, height, width, max_elongation)?;
        }
    }
    if config.min_obj_no_zslices.is_some() {
        // Python applies this only to 3D z-stack labels. The Rust runner's
        // segmentation path produces one 2D mask per timepoint, so there is no
        // z-axis to filter here.
    }
    Ok(())
}

fn apply_preprocess_steps(
    mut frame: Vec<f32>,
    height: usize,
    width: usize,
    steps: &[PreprocessStep],
) -> Result<Vec<f32>> {
    for step in steps {
        match step {
            PreprocessStep::GaussianFilter { sigma_y, sigma_x } => {
                frame = gaussian_blur_frame_xy(&frame, height, width, *sigma_y, *sigma_x)?;
            }
            PreprocessStep::RemoveHotPixels => {
                frame = morphological_opening_frame(&frame, height, width)?;
            }
            PreprocessStep::SpotDetectorFilter { radius_y, radius_x } => {
                frame = spot_detector_frame(&frame, height, width, *radius_y, *radius_x)?;
            }
            PreprocessStep::RidgeFilter { sigmas } => {
                frame = ridge_filter_frame(&frame, height, width, sigmas)?;
            }
            PreprocessStep::EnhanceSpeckles { radius } => {
                frame = enhance_speckles_frame(&frame, height, width, *radius)?;
            }
            PreprocessStep::CorrectIllumination {
                block_size,
                approximate_object_diameter,
                apply_gaussian_filter,
            } => {
                frame = correct_illumination_frame(
                    &frame,
                    height,
                    width,
                    *block_size,
                    *approximate_object_diameter,
                    *apply_gaussian_filter,
                )?;
            }
        }
    }
    Ok(frame)
}

fn correct_illumination_frame(
    frame: &[f32],
    height: usize,
    width: usize,
    block_size: usize,
    approximate_object_diameter: f32,
    apply_gaussian_filter: bool,
) -> Result<Vec<f32>> {
    let background_source = if apply_gaussian_filter {
        gaussian_blur_frame_xy(
            frame,
            height,
            width,
            approximate_object_diameter / 2.0,
            approximate_object_diameter / 2.0,
        )?
    } else {
        frame.to_vec()
    };
    let background = morphological_opening_with_offsets(
        &background_source,
        height,
        width,
        &disk_offsets(block_size),
    )?;
    Ok(frame
        .iter()
        .zip(background.iter())
        .map(|(input, background)| input - background)
        .collect())
}

fn enhance_speckles_frame(
    frame: &[f32],
    height: usize,
    width: usize,
    radius: usize,
) -> Result<Vec<f32>> {
    let offsets = disk_offsets(radius);
    let opened = morphological_opening_with_offsets(frame, height, width, &offsets)?;
    Ok(frame
        .iter()
        .zip(opened.iter())
        .map(|(input, background)| input - background)
        .collect())
}

fn spot_detector_frame(
    frame: &[f32],
    height: usize,
    width: usize,
    radius_y: f32,
    radius_x: f32,
) -> Result<Vec<f32>> {
    if radius_y <= 0.0 || radius_x <= 0.0 {
        bail!(
            "Spot detector filter radii must be positive, got ({}, {})",
            radius_y,
            radius_x
        );
    }

    let sqrt_2 = std::f32::consts::SQRT_2;
    let sigma_y = radius_y / (1.0 + sqrt_2);
    let sigma_x = radius_x / (1.0 + sqrt_2);
    let blurred1 = gaussian_blur_frame_xy(frame, height, width, sigma_y, sigma_x)?;
    let blurred2 =
        gaussian_blur_frame_xy(frame, height, width, sqrt_2 * sigma_y, sqrt_2 * sigma_x)?;
    let sharpened = blurred1
        .iter()
        .zip(blurred2.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    rescale_to_input_range(&sharpened, frame)
}

fn ridge_filter_frame(
    frame: &[f32],
    height: usize,
    width: usize,
    sigmas: &[f32],
) -> Result<Vec<f32>> {
    if frame.len() != height * width {
        bail!(
            "Cannot preprocess frame with {} pixels as {}x{}",
            frame.len(),
            height,
            width
        );
    }
    if sigmas.iter().any(|sigma| *sigma <= 0.0) {
        bail!("Ridge filter sigmas must be positive, got {:?}", sigmas);
    }
    if sigmas.is_empty() {
        return Ok(frame.to_vec());
    }

    let mut best = vec![0.0; frame.len()];
    for sigma in sigmas {
        let blurred = gaussian_blur_frame_xy(frame, height, width, *sigma, *sigma)?;
        let scale = sigma * sigma;
        for y in 0..height {
            for x in 0..width {
                let center = sample_clamped(&blurred, height, width, y as isize, x as isize);
                let left = sample_clamped(&blurred, height, width, y as isize, x as isize - 1);
                let right = sample_clamped(&blurred, height, width, y as isize, x as isize + 1);
                let up = sample_clamped(&blurred, height, width, y as isize - 1, x as isize);
                let down = sample_clamped(&blurred, height, width, y as isize + 1, x as isize);
                let up_left =
                    sample_clamped(&blurred, height, width, y as isize - 1, x as isize - 1);
                let up_right =
                    sample_clamped(&blurred, height, width, y as isize - 1, x as isize + 1);
                let down_left =
                    sample_clamped(&blurred, height, width, y as isize + 1, x as isize - 1);
                let down_right =
                    sample_clamped(&blurred, height, width, y as isize + 1, x as isize + 1);

                let hxx = (left - 2.0 * center + right) * scale;
                let hyy = (up - 2.0 * center + down) * scale;
                let hxy = ((down_right - down_left - up_right + up_left) * 0.25) * scale;
                let trace = hxx + hyy;
                let delta = ((hxx - hyy) * (hxx - hyy) + 4.0 * hxy * hxy).sqrt();
                let lambda_a = 0.5 * (trace - delta);
                let lambda_b = 0.5 * (trace + delta);
                let response = (-lambda_a.min(lambda_b)).max(0.0);
                let idx = y * width + x;
                if response > best[idx] {
                    best[idx] = response;
                }
            }
        }
    }
    Ok(best)
}

fn sample_clamped(frame: &[f32], height: usize, width: usize, y: isize, x: isize) -> f32 {
    let yy = y.clamp(0, height.saturating_sub(1) as isize) as usize;
    let xx = x.clamp(0, width.saturating_sub(1) as isize) as usize;
    frame[yy * width + xx]
}

fn rescale_to_input_range(values: &[f32], input: &[f32]) -> Result<Vec<f32>> {
    if values.len() != input.len() {
        bail!(
            "Cannot rescale {} preprocessed pixels to {} input pixels",
            values.len(),
            input.len()
        );
    }

    let input_min = input.iter().copied().fold(f32::INFINITY, f32::min);
    let input_max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let value_min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let value_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (value_max - value_min).abs() <= f32::EPSILON {
        return Ok(vec![input_min; values.len()]);
    }
    let scale = (input_max - input_min) / (value_max - value_min);
    Ok(values
        .iter()
        .map(|value| input_min + (*value - value_min) * scale)
        .collect())
}

fn morphological_opening_frame(frame: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
    let offsets = square_offsets(1);
    morphological_opening_with_offsets(frame, height, width, &offsets)
}

fn morphological_opening_with_offsets(
    frame: &[f32],
    height: usize,
    width: usize,
    offsets: &[(isize, isize)],
) -> Result<Vec<f32>> {
    if frame.len() != height * width {
        bail!(
            "Cannot preprocess frame with {} pixels as {}x{}",
            frame.len(),
            height,
            width
        );
    }
    if offsets.is_empty() {
        return Ok(frame.to_vec());
    }

    let mut eroded = vec![0.0; frame.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = f32::INFINITY;
            for (dy, dx) in offsets {
                let yy = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                let xx = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                value = value.min(frame[yy * width + xx]);
            }
            eroded[y * width + x] = value;
        }
    }

    let mut opened = vec![0.0; frame.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = f32::NEG_INFINITY;
            for (dy, dx) in offsets {
                let yy = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                let xx = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                value = value.max(eroded[yy * width + xx]);
            }
            opened[y * width + x] = value;
        }
    }

    Ok(opened)
}

fn square_offsets(radius: usize) -> Vec<(isize, isize)> {
    let radius = radius as isize;
    let mut offsets = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            offsets.push((dy, dx));
        }
    }
    offsets
}

fn disk_offsets(radius: usize) -> Vec<(isize, isize)> {
    let radius = radius as isize;
    let radius_sq = radius * radius;
    let mut offsets = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dy * dy + dx * dx <= radius_sq {
                offsets.push((dy, dx));
            }
        }
    }
    offsets
}

fn gaussian_blur_frame_xy(
    frame: &[f32],
    height: usize,
    width: usize,
    sigma_y: f32,
    sigma_x: f32,
) -> Result<Vec<f32>> {
    if frame.len() != height * width {
        bail!(
            "Cannot preprocess frame with {} pixels as {}x{}",
            frame.len(),
            height,
            width
        );
    }
    if sigma_y <= 0.0 && sigma_x <= 0.0 {
        return Ok(frame.to_vec());
    }

    let kernel_x = gaussian_kernel_1d(sigma_x);
    let kernel_y = gaussian_kernel_1d(sigma_y);
    if kernel_x.is_empty() && kernel_y.is_empty() {
        return Ok(frame.to_vec());
    }

    let mut horizontal = frame.to_vec();
    if !kernel_x.is_empty() {
        horizontal = convolve_horizontal(frame, height, width, &kernel_x);
    }

    let mut blurred = horizontal;
    if !kernel_y.is_empty() {
        blurred = convolve_vertical(&blurred, height, width, &kernel_y);
    }

    Ok(blurred)
}

fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return Vec::new();
    }
    let radius = (sigma * 3.0).ceil() as usize;
    if radius == 0 {
        return Vec::new();
    }
    let mut kernel = Vec::with_capacity(radius * 2 + 1);
    let sigma2 = 2.0 * sigma * sigma;
    for offset in 0..=(radius * 2) {
        let distance = offset as isize - radius as isize;
        kernel.push((-(distance * distance) as f32 / sigma2).exp());
    }
    let sum = kernel.iter().sum::<f32>();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn convolve_horizontal(frame: &[f32], height: usize, width: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut horizontal = vec![0.0; frame.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for (kernel_idx, weight) in kernel.iter().enumerate() {
                let dx = kernel_idx as isize - radius as isize;
                let sample_x = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize);
                value += frame[y * width + sample_x as usize] * weight;
            }
            horizontal[y * width + x] = value;
        }
    }
    horizontal
}

fn convolve_vertical(frame: &[f32], height: usize, width: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut blurred = vec![0.0; frame.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for (kernel_idx, weight) in kernel.iter().enumerate() {
                let dy = kernel_idx as isize - radius as isize;
                let sample_y = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize);
                value += frame[sample_y as usize * width + x] * weight;
            }
            blurred[y * width + x] = value;
        }
    }
    blurred
}

fn remove_labels_below_area(mask: &mut [u32], min_area: usize) {
    if min_area == 0 {
        return;
    }
    let mut areas = std::collections::BTreeMap::<u32, usize>::new();
    for label in mask.iter().copied().filter(|label| *label != 0) {
        *areas.entry(label).or_default() += 1;
    }
    let labels_to_clear = areas
        .into_iter()
        .filter_map(|(label, area)| if area < min_area { Some(label) } else { None })
        .collect::<BTreeSet<_>>();
    if labels_to_clear.is_empty() {
        return;
    }
    for label in mask {
        if labels_to_clear.contains(label) {
            *label = 0;
        }
    }
}

fn remove_labels_below_solidity(
    mask: &mut [u32],
    height: usize,
    width: usize,
    min_solidity: f64,
) -> Result<()> {
    let expected_len = height * width;
    if mask.len() != expected_len {
        bail!(
            "Segmentation mask has {} pixels but expected {}",
            mask.len(),
            expected_len
        );
    }

    let stats = collect_label_shape_stats(mask, width);
    let labels_to_clear = stats
        .into_iter()
        .filter_map(|(label, stats)| {
            if stats.solidity() < min_solidity {
                Some(label)
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();
    clear_labels(mask, &labels_to_clear);
    Ok(())
}

fn remove_labels_above_elongation(
    mask: &mut [u32],
    height: usize,
    width: usize,
    max_elongation: f64,
) -> Result<()> {
    let expected_len = height * width;
    if mask.len() != expected_len {
        bail!(
            "Segmentation mask has {} pixels but expected {}",
            mask.len(),
            expected_len
        );
    }

    let stats = collect_label_shape_stats(mask, width);
    let labels_to_clear = stats
        .into_iter()
        .filter_map(|(label, stats)| {
            if stats.elongation() > max_elongation {
                Some(label)
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();
    clear_labels(mask, &labels_to_clear);
    Ok(())
}

fn collect_label_shape_stats(mask: &[u32], width: usize) -> BTreeMap<u32, LabelShapeStats> {
    let mut stats = BTreeMap::<u32, LabelShapeStats>::new();
    for (idx, label) in mask.iter().copied().enumerate() {
        if label == 0 {
            continue;
        }
        let y = idx / width;
        let x = idx % width;
        stats.entry(label).or_default().add(y, x);
    }
    stats
}

fn clear_labels(mask: &mut [u32], labels_to_clear: &BTreeSet<u32>) {
    if labels_to_clear.is_empty() {
        return;
    }
    for label in mask {
        if labels_to_clear.contains(label) {
            *label = 0;
        }
    }
}

#[derive(Debug, Default)]
struct LabelShapeStats {
    area: usize,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
    pixels: Vec<(usize, usize)>,
}

impl LabelShapeStats {
    fn add(&mut self, y: usize, x: usize) {
        self.pixels.push((y, x));
        let x = x as f64;
        let y = y as f64;
        self.area += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.sum_xy += x * y;
    }

    fn solidity(&self) -> f64 {
        let points = pixel_square_corners(&self.pixels);
        let hull = convex_hull(points);
        let convex_area = polygon_area(&hull);
        if convex_area > 0.0 {
            self.area as f64 / convex_area
        } else {
            f64::NAN
        }
    }

    fn elongation(&self) -> f64 {
        let area = self.area as f64;
        if area == 0.0 {
            return f64::NAN;
        }
        let cx = self.sum_x / area;
        let cy = self.sum_y / area;
        let mu20 = self.sum_x2 - 2.0 * cx * self.sum_x + area * cx * cx;
        let mu02 = self.sum_y2 - 2.0 * cy * self.sum_y + area * cy * cy;
        let mu11 = self.sum_xy - cx * self.sum_y - cy * self.sum_x + area * cx * cy;
        let cov_xx = mu20 / area;
        let cov_yy = mu02 / area;
        let cov_xy = mu11 / area;
        let trace = cov_xx + cov_yy;
        let delta = ((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy).sqrt();
        let eig0 = ((trace + delta) / 2.0).max(0.0);
        let eig1 = ((trace - delta) / 2.0).max(0.0);
        let major_axis_length = 4.0 * eig0.sqrt();
        let minor_axis_length = 4.0 * eig1.sqrt();
        major_axis_length / minor_axis_length.max(1.0)
    }
}

fn pixel_square_corners(pixels: &[(usize, usize)]) -> Vec<(f64, f64)> {
    let mut points = BTreeSet::new();
    for &(y, x) in pixels {
        let x = x as i64;
        let y = y as i64;
        points.insert((x, y));
        points.insert((x + 1, y));
        points.insert((x, y + 1));
        points.insert((x + 1, y + 1));
    }
    points
        .into_iter()
        .map(|(x, y)| (x as f64, y as f64))
        .collect()
}

fn convex_hull(mut points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if points.len() <= 1 {
        return points;
    }
    points.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap()
            .then_with(|| left.1.partial_cmp(&right.1).unwrap())
    });

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn polygon_area(points: &[(f64, f64)]) -> f64 {
    if points.len() < 3 {
        return f64::NAN;
    }
    let mut area = 0.0;
    for idx in 0..points.len() {
        let next = (idx + 1) % points.len();
        area += points[idx].0 * points[next].1 - points[next].0 * points[idx].1;
    }
    area.abs() / 2.0
}

fn active_data_prep_crop_roi<'a>(
    rois: &'a [CropRoiRect],
    cropped_roi_ids: &[usize],
) -> Option<&'a CropRoiRect> {
    rois.iter()
        .find(|roi| roi.roi_id == 0)
        .or_else(|| rois.first())
        .filter(|roi| !cropped_roi_ids.contains(&roi.roi_id))
}

fn apply_data_prep_crop_roi_filter(
    masks: &mut [Vec<u32>],
    height: usize,
    width: usize,
    roi: &CropRoiRect,
) -> Result<()> {
    let frame_len = height * width;
    let x0 = roi.x.min(width);
    let y0 = roi.y.min(height);
    let x1 = roi.x.saturating_add(roi.width).min(width);
    let y1 = roi.y.saturating_add(roi.height).min(height);
    if x0 >= x1 || y0 >= y1 {
        bail!("Data-prep crop ROI {} has zero width or height", roi.roi_id);
    }

    for mask in masks {
        if mask.len() != frame_len {
            bail!(
                "Segmentation mask has {} pixels but expected {}",
                mask.len(),
                frame_len
            );
        }
        for y in 0..height {
            for x in 0..width {
                if x < x0 || x >= x1 || y < y0 || y >= y1 {
                    mask[y * width + x] = 0;
                }
            }
        }
    }
    Ok(())
}

fn apply_data_prep_free_roi_filter(
    masks: &mut [Vec<u32>],
    height: usize,
    width: usize,
    roi: &FreehandRoiMask,
) -> Result<()> {
    let frame_len = height * width;
    let mut keep_mask = vec![false; frame_len];
    let (y0, x0, _y1, _x1) = roi.bbox_yxxy;

    for ((local_y, local_x), keep) in roi.local_mask.indexed_iter() {
        if !*keep {
            continue;
        }
        let y = y0 + local_y;
        let x = x0 + local_x;
        if y < height && x < width {
            keep_mask[y * width + x] = true;
        }
    }

    for mask in masks {
        if mask.len() != frame_len {
            bail!(
                "Segmentation mask has {} pixels but expected {}",
                mask.len(),
                frame_len
            );
        }
        let labels_to_clear = mask
            .iter()
            .zip(&keep_mask)
            .filter_map(|(label, keep)| {
                if *label != 0 && !*keep {
                    Some(*label)
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if labels_to_clear.is_empty() {
            continue;
        }
        for label in mask {
            if labels_to_clear.contains(label) {
                *label = 0;
            }
        }
    }
    Ok(())
}

fn resolve_stop_frame_count(total_frames: usize, stop_frame: Option<usize>) -> Result<usize> {
    match stop_frame {
        Some(limit) if limit > total_frames => bail!(
            "Requested stop_frame {} exceeds available frame count {}",
            limit,
            total_frames
        ),
        Some(limit) => Ok(limit),
        None => Ok(total_frames),
    }
}

fn segmentation_name(endname: Option<&str>) -> String {
    match endname {
        Some(value) if value.trim().trim_end_matches(".npz").starts_with("segm") => {
            value.trim().trim_end_matches(".npz").to_string()
        }
        Some(value) if !value.trim().is_empty() => {
            format!("segm_{}", value.trim().trim_end_matches(".npz"))
        }
        _ => "segm".to_string(),
    }
}

fn output_paths(images_dir: &Path, basename: &str, endname: Option<&str>) -> RunOutputPaths {
    let segm_name = segmentation_name(endname);
    let acdc_output_name = segm_name.replacen("segm", "acdc_output", 1);
    let segm_npz_path = images_dir.join(format!("{basename}{segm_name}.npz"));
    let acdc_output_csv_path = images_dir.join(format!("{basename}{acdc_output_name}.csv"));
    let segm_hyperparams_ini_path = images_dir.join(format!("{basename}segm_hyperparams.ini"));

    RunOutputPaths {
        segm_npz_path,
        acdc_output_csv_path,
        segm_hyperparams_ini_path,
    }
}

fn guard_outputs(paths: &RunOutputPaths, policy: OverwritePolicy) -> Result<()> {
    if policy == OverwritePolicy::Overwrite {
        return Ok(());
    }

    for path in [
        &paths.segm_npz_path,
        &paths.acdc_output_csv_path,
        &paths.segm_hyperparams_ini_path,
    ] {
        if path.exists() {
            bail!(
                "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
                path.display()
            );
        }
    }
    Ok(())
}

fn write_hyperparams_ini(
    path: &Path,
    position: &PositionSpec,
    model_path: &Path,
    params: &SegmentationParams,
    cpu: bool,
    segm_endname: Option<&str>,
    tracking: Option<&TrackingConfig>,
    segm_npz_path: &Path,
) -> Result<()> {
    let segm_name = segmentation_name(segm_endname);
    let existing = if path.exists() {
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?
    } else {
        String::new()
    };
    let run_prefix = format!("{segm_name}.metadata.run_number_");
    let run_number = existing
        .lines()
        .filter(|line| {
            line.strip_prefix('[')
                .and_then(|line| line.strip_suffix(']'))
                .map(|section| section.starts_with(&run_prefix))
                .unwrap_or(false)
        })
        .count()
        + 1;

    let mut content = existing;
    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.6f");
    let model_name = model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("cellpose");

    content.push_str(&format!("[{segm_name}.metadata.run_number_{run_number}]\n"));
    content.push_str(&format!(
        "segmentation_filename = {}\n",
        segm_npz_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("segm.npz")
    ));
    content.push_str(&format!("segmented_channel = {}\n", position.phase_channel));
    content.push_str(&format!("segmented_on = {}\n", timestamp));
    content.push_str(&format!("model_name = {}\n\n", model_name));

    content.push_str(&format!("[{segm_name}.init.run_number_{run_number}]\n"));
    content.push_str(&format!("model_path = {}\n", model_path.display()));
    content.push_str(&format!("cpu = {}\n", cpu));
    content.push_str(&format!("phase_channel = {}\n", position.phase_channel));
    content.push_str(&format!("fluo_channel = {}\n\n", position.fluo_channel));

    content.push_str(&format!("[{segm_name}.segment.run_number_{run_number}]\n"));
    content.push_str(&format!("tile = {}\n", params.tile));
    content.push_str(&format!("batch_size = {}\n", params.batch_size));
    content.push_str(&format!(
        "cellprob_threshold = {}\n",
        params.cellprob_threshold
    ));
    content.push_str(&format!("niter = {}\n", params.niter));
    content.push_str(&format!("min_size = {}\n", params.min_size));
    if let Some(tracking) = tracking {
        content.push_str("track = true\n");
        content.push_str(&format!("IoA_thresh = {}\n", tracking.ioa_threshold));
        content.push_str(&format!(
            "assign_unique_new_IDs = {}\n",
            tracking.assign_unique_new_ids
        ));
        content.push_str(&format!(
            "denom_overlap_matrix = {}\n",
            match tracking.overlap_denominator {
                OverlapDenominator::AreaPrev => "area_prev",
                OverlapDenominator::Union => "union",
            }
        ));
    }
    content.push('\n');

    fs::write(path, content).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prep::{save_crop_roi_coords_csv, write_freehand_roi_npz, CropRoiCoordsTable};
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    struct FakeSegmenter {
        masks_per_call: Vec<Vec<u32>>,
        calls: usize,
    }

    impl Segmenter for FakeSegmenter {
        fn segment_pair(
            &mut self,
            _phase: Vec<f32>,
            _fluo: Vec<f32>,
            _height: usize,
            _width: usize,
            _params: &SegmentationParams,
        ) -> Result<Vec<u32>> {
            let index = self.calls.min(self.masks_per_call.len().saturating_sub(1));
            self.calls += 1;
            Ok(self.masks_per_call[index].clone())
        }
    }

    struct RecordingSegmenter {
        phase_inputs: Vec<Vec<f32>>,
        fluo_inputs: Vec<Vec<f32>>,
        mask: Vec<u32>,
    }

    impl Segmenter for RecordingSegmenter {
        fn segment_pair(
            &mut self,
            phase: Vec<f32>,
            fluo: Vec<f32>,
            _height: usize,
            _width: usize,
            _params: &SegmentationParams,
        ) -> Result<Vec<u32>> {
            self.phase_inputs.push(phase);
            self.fluo_inputs.push(fluo);
            Ok(self.mask.clone())
        }
    }

    #[test]
    fn writes_outputs_for_single_position() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                0, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 2, 2, //
                0, 0, 2, 2, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 2);
        assert_eq!(result.frames_processed, 1);
        assert!(result.outputs.segm_npz_path.exists());
        assert!(result.outputs.acdc_output_csv_path.exists());
        assert!(result.outputs.segm_hyperparams_ini_path.exists());
        Ok(())
    }

    #[test]
    fn skips_writing_outputs_when_save_disabled() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preview".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                0, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 2, 2, //
                0, 0, 2, 2, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 2);
        assert_eq!(result.frames_processed, 1);
        assert!(!result.outputs.segm_npz_path.exists());
        assert!(!result.outputs.acdc_output_csv_path.exists());
        assert!(!result.outputs.segm_hyperparams_ini_path.exists());
        Ok(())
    }

    #[test]
    fn applies_gaussian_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![0u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &vec![10u16; 16])?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::GaussianFilter {
            sigma_y: 1.0,
            sigma_x: 1.0,
        }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert!(phase[5] < 100.0);
        assert!(phase[1] > 0.0);
        assert!(phase[4] > 0.0);
        let fluo = segmenter.fluo_inputs.first().expect("fluo input");
        assert!(fluo.iter().all(|value| (*value - 10.0).abs() < 0.001));
        Ok(())
    }

    #[test]
    fn applies_vector_gaussian_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![0u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::GaussianFilter {
            sigma_y: 1.0,
            sigma_x: 0.0,
        }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert!(phase[1] > 0.0);
        assert_eq!(phase[4], 0.0);
        Ok(())
    }

    #[test]
    fn applies_remove_hot_pixels_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![0u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::RemoveHotPixels];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert_eq!(phase[5], 0.0);
        assert!(phase.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn applies_spot_detector_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![0u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::SpotDetectorFilter {
            radius_y: 2.0,
            radius_x: 2.0,
        }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert!(phase.iter().all(|value| *value >= 0.0 && *value <= 100.0));
        assert!(phase[5] > phase[0]);
        assert!(phase.iter().any(|value| *value > 0.0 && *value < 100.0));
        Ok(())
    }

    #[test]
    fn applies_ridge_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let pixels = vec![
            0, 100, 0, 0, //
            0, 100, 0, 0, //
            0, 100, 0, 0, //
            0, 100, 0, 0, //
        ];
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::RidgeFilter { sigmas: vec![1.0] }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert!(phase.iter().all(|value| *value >= 0.0));
        let ridge_column = [phase[1], phase[5], phase[9], phase[13]];
        let off_column = [phase[3], phase[7], phase[11], phase[15]];
        assert!(ridge_column.iter().sum::<f32>() > off_column.iter().sum::<f32>());
        Ok(())
    }

    #[test]
    fn applies_enhance_speckles_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![10u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::EnhanceSpeckles { radius: 1 }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert_eq!(phase[5], 90.0);
        assert_eq!(phase[0], 0.0);
        assert!(phase.iter().all(|value| *value >= 0.0));
        Ok(())
    }

    #[test]
    fn applies_correct_illumination_preprocess_before_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let mut pixels = vec![10u16; 16];
        pixels[5] = 100;
        write_test_tiff_pixels(&images.join("demo_phase.tif"), &pixels)?;
        write_test_tiff_pixels(&images.join("demo_fluo.tif"), &pixels)?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("preprocessed".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.save_outputs = false;
        config.preprocess_steps = vec![PreprocessStep::CorrectIllumination {
            block_size: 1,
            approximate_object_diameter: 15.0,
            apply_gaussian_filter: false,
        }];

        let mut segmenter = RecordingSegmenter {
            phase_inputs: Vec::new(),
            fluo_inputs: Vec::new(),
            mask: vec![0; 16],
        };
        run_position_with_segmenter(config, &mut segmenter)?;

        let phase = segmenter.phase_inputs.first().expect("phase input");
        assert_eq!(phase[5], 90.0);
        assert_eq!(phase[0], 0.0);
        assert!(phase.iter().all(|value| *value >= 0.0));
        Ok(())
    }

    #[test]
    fn applies_min_area_standard_postprocess_before_saving() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.postprocess = Some(PostprocessConfig {
            min_area: Some(3),
            min_solidity: None,
            max_elongation: None,
            min_obj_no_zslices: None,
        });

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                1, 1, 0, 0, //
                1, 0, 2, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 1);
        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array2<u32> = npz.by_name("arr_0.npy")?;
        assert!(masks.iter().any(|value| *value == 1));
        assert!(masks.iter().all(|value| *value != 2));
        Ok(())
    }

    #[test]
    fn applies_min_solidity_standard_postprocess_before_saving() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.postprocess = Some(PostprocessConfig {
            min_area: None,
            min_solidity: Some(0.9),
            max_elongation: None,
            min_obj_no_zslices: None,
        });

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                1, 1, 0, 0, //
                1, 0, 2, 2, //
                0, 0, 2, 2, //
                0, 0, 0, 0, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 2);
        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array2<u32> = npz.by_name("arr_0.npy")?;
        assert!(masks.iter().all(|value| *value != 1));
        assert!(masks.iter().any(|value| *value == 2));
        Ok(())
    }

    #[test]
    fn applies_max_elongation_standard_postprocess_before_saving() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;

        let mut config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        config.postprocess = Some(PostprocessConfig {
            min_area: None,
            min_solidity: None,
            max_elongation: Some(2.0),
            min_obj_no_zslices: None,
        });

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                1, 1, 1, 1, //
                0, 0, 0, 0, //
                2, 2, 0, 0, //
                2, 2, 0, 0, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 2);
        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array2<u32> = npz.by_name("arr_0.npy")?;
        assert!(masks.iter().all(|value| *value != 1));
        assert!(masks.iter().any(|value| *value == 2));
        Ok(())
    }

    #[test]
    fn normalizes_python_segmentation_end_filename() {
        let images = Path::new("/tmp/images");
        let from_full_name = output_paths(images, "demo_", Some("segm_rust.npz"));
        assert_eq!(
            from_full_name.segm_npz_path,
            images.join("demo_segm_rust.npz")
        );
        assert_eq!(
            from_full_name.acdc_output_csv_path,
            images.join("demo_acdc_output_rust.csv")
        );

        let from_suffix = output_paths(images, "demo_", Some("rust"));
        assert_eq!(from_suffix, from_full_name);

        let default = output_paths(images, "demo_", Some("segm.npz"));
        assert_eq!(default.segm_npz_path, images.join("demo_segm.npz"));
        assert_eq!(
            default.acdc_output_csv_path,
            images.join("demo_acdc_output.csv")
        );
    }

    #[test]
    fn applies_data_prep_free_roi_to_segmented_masks() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;
        write_freehand_roi_npz(
            images.join("demo_dataPrepFreeRoi.npz"),
            &FreehandRoiMask {
                bbox_yxxy: (0, 0, 1, 1),
                local_mask: Array2::from_elem((2, 2), true),
            },
        )?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                1, 1, 2, 0, //
                1, 1, 2, 0, //
                0, 0, 2, 2, //
                0, 0, 0, 0, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 1);

        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array2<u32> = npz.by_name("arr_0.npy")?;
        assert!(masks.iter().any(|value| *value == 1));
        assert!(masks.iter().all(|value| *value != 2));
        Ok(())
    }

    #[test]
    fn applies_data_prep_crop_roi_to_segmented_masks() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;
        save_crop_roi_coords_csv(
            images.join("demo_dataPrepROIs_coords.csv"),
            &CropRoiCoordsTable {
                rois: vec![CropRoiRect {
                    roi_id: 0,
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 2,
                }],
                cropped_roi_ids: Vec::new(),
            },
        )?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![
                2, 2, 0, 0, //
                2, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 0, 0, //
            ]],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 1);

        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array2<u32> = npz.by_name("arr_0.npy")?;
        assert_eq!(masks[[1, 1]], 1);
        assert_eq!(masks[[2, 2]], 1);
        assert!(masks.iter().all(|value| *value != 2));
        assert_eq!(masks[[0, 0]], 0);
        assert_eq!(masks[[1, 3]], 0);
        Ok(())
    }

    #[test]
    fn refuses_overwrite_without_flag() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"))?;
        write_test_tiff(&images.join("demo_fluo.tif"))?;
        fs::write(images.join("demo_segm.npz"), b"occupied")?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![0; 16]],
            calls: 0,
        };
        let err = run_position_with_segmenter(config, &mut segmenter).unwrap_err();
        assert!(err.to_string().contains("Refusing to overwrite"));
        Ok(())
    }

    #[test]
    fn runs_experiment_across_positions() -> Result<()> {
        let temp = tempdir()?;
        for idx in 1..=2 {
            let images = temp.path().join(format!("Position_{idx}")).join("Images");
            fs::create_dir_all(&images)?;
            write_test_tiff(&images.join("demo_phase.tif"))?;
            write_test_tiff(&images.join("demo_fluo.tif"))?;
        }

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![0; 16]],
            calls: 0,
        };
        let results = run_experiment_with_segmenter(
            ExperimentRunConfig {
                experiment_dir: temp.path().to_path_buf(),
                phase_channel: "phase".into(),
                fluo_channel: "fluo".into(),
                model_path: PathBuf::from("unused-model.onnx"),
                segm_endname: Some("rust".into()),
                overwrite_policy: OverwritePolicy::Refuse,
                cpu: true,
                params: SegmentationParams::default(),
                preprocess_steps: Vec::new(),
                tracking: None,
                postprocess: None,
                stop_frame: None,
                save_outputs: true,
                use_data_prep_roi: true,
                use_data_prep_free_roi: true,
            },
            discover_experiment(temp.path(), "phase", "fluo")?,
            &mut segmenter,
        )?;

        assert_eq!(results.len(), 2);
        assert!(results[0]
            .outputs
            .segm_npz_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap()
            .contains("segm_rust"));
        Ok(())
    }

    #[test]
    fn writes_timelapse_outputs_for_multi_page_position() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 20, 30])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[15, 25, 35])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,3\nSizeZ,1\nTimeIncrement,30\n",
        )?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![
                vec![
                    0, 1, 1, 0, //
                    0, 1, 1, 0, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
                vec![
                    0, 0, 0, 0, //
                    0, 2, 2, 0, //
                    0, 2, 2, 0, //
                    0, 0, 0, 0, //
                ],
                vec![
                    0, 0, 3, 3, //
                    0, 0, 3, 3, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
            ],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.frames_processed, 3);
        assert_eq!(result.labels_found, 3);
        assert_eq!(segmenter.calls, 3);

        let mut npz = NpzReader::new(File::open(&result.outputs.segm_npz_path)?)?;
        let masks: Array3<u32> = npz.by_name("arr_0.npy")?;
        assert_eq!(masks.shape(), &[3, 4, 4]);

        let mut reader = Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader.headers()?.clone();
        assert_eq!(
            headers.iter().take(5).collect::<Vec<_>>(),
            vec![
                "frame_i",
                "time_seconds",
                "time_minutes",
                "time_hours",
                "z_slice_used"
            ]
        );

        let frame_i = header_index(&headers, "frame_i");
        let time_seconds = header_index(&headers, "time_seconds");
        let time_minutes = header_index(&headers, "time_minutes");
        let cell_id = header_index(&headers, "Cell_ID");
        let rows = reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()?;

        assert_eq!(rows.len(), 3);
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("0")
                && row.get(cell_id) == Some("1")
                && row.get(time_seconds) == Some("0")
                && row.get(time_minutes) == Some("0")
        }));
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("1")
                && row.get(cell_id) == Some("2")
                && row.get(time_seconds) == Some("30")
                && row.get(time_minutes) == Some("0.5")
        }));
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("2")
                && row.get(cell_id) == Some("3")
                && row.get(time_seconds) == Some("60")
                && row.get(time_minutes) == Some("1")
        }));
        Ok(())
    }

    #[test]
    fn runs_zstack_position_using_segm_info_projection() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(&images.join("demo_phase_aligned.npz"), &[1.0, 5.0], 2, 2)?;
        write_test_volume_npz(&images.join("demo_fluo_aligned.npz"), &[2.0, 6.0], 2, 2)?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,2\nTimeIncrement,30\nPhysicalSizeZ,1.5\nPhysicalSizeY,0.5\nPhysicalSizeX,0.25\n",
        )?;
        fs::write(
            images.join("demo_segmInfo.csv"),
            concat!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n",
                "demo_phase_aligned.npz,0,1,single z-slice,1,1,single z-slice,0\n",
                "demo_phase_aligned.npz,1,1,single z-slice,1,1,single z-slice,0\n",
                "demo_fluo_aligned.npz,0,1,single z-slice,1,1,single z-slice,0\n",
                "demo_fluo_aligned.npz,1,1,single z-slice,1,1,single z-slice,0\n",
            ),
        )?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![
                vec![
                    0, 1, 0, 0, //
                    0, 1, 0, 0, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
                vec![
                    0, 0, 2, 0, //
                    0, 0, 2, 0, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
            ],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        let csv = fs::read_to_string(result.outputs.acdc_output_csv_path)?;
        assert!(csv.contains("z_slice_used"));
        assert!(csv.contains("single z-slice"));
        assert!(csv.contains(",1,"));
        Ok(())
    }

    #[test]
    fn rejects_mismatched_channel_frame_counts() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 20, 30])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[15, 25])?;

        let config = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            None,
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![vec![0; 16]],
            calls: 0,
        };
        let err = run_position_with_segmenter(config, &mut segmenter).unwrap_err();
        assert!(err.to_string().contains("Input image stack mismatch"));
        Ok(())
    }

    #[test]
    fn tracks_ids_across_frames_and_marks_disappearances() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 20, 30])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[15, 25, 35])?;

        let base = resolve_position_run_config(
            temp.path().join("Position_1"),
            "phase",
            "fluo",
            "unused-model.onnx",
            Some("tracked".into()),
            OverwritePolicy::Refuse,
            true,
            SegmentationParams::default(),
        )?;
        let config = SegmentationRunConfig {
            tracking: Some(TrackingConfig {
                ioa_threshold: 0.4,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::AreaPrev,
            }),
            ..base
        };

        let mut segmenter = FakeSegmenter {
            masks_per_call: vec![
                vec![
                    1, 1, 0, 0, //
                    1, 1, 0, 0, //
                    0, 0, 2, 2, //
                    0, 0, 2, 2, //
                ],
                vec![
                    0, 3, 3, 0, //
                    0, 3, 3, 0, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
                vec![
                    0, 0, 4, 4, //
                    0, 0, 4, 4, //
                    0, 0, 0, 0, //
                    0, 0, 0, 0, //
                ],
            ],
            calls: 0,
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        let mut reader = Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader.headers()?.clone();
        let frame_i = header_index(&headers, "frame_i");
        let cell_id = header_index(&headers, "Cell_ID");
        let disappears_before_end = header_index(&headers, "disappears_before_end");
        let rows = reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()?;

        assert_eq!(rows.len(), 4);
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("0")
                && row.get(cell_id) == Some("1")
                && row.get(disappears_before_end) == Some("0")
        }));
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("1")
                && row.get(cell_id) == Some("1")
                && row.get(disappears_before_end) == Some("0")
        }));
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("2")
                && row.get(cell_id) == Some("1")
                && row.get(disappears_before_end) == Some("0")
        }));
        assert!(rows.iter().any(|row| {
            row.get(frame_i) == Some("0")
                && row.get(cell_id) == Some("2")
                && row.get(disappears_before_end) == Some("1")
        }));
        Ok(())
    }

    #[test]
    fn writes_python_compatible_hyperparams_sections() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("segm_hyperparams.ini");
        let position = PositionSpec {
            position_dir: temp.path().join("Position_1"),
            images_dir: temp.path().to_path_buf(),
            basename: "demo_".into(),
            channels: Vec::new(),
            phase_channel: "phase".into(),
            fluo_channel: "fluo".into(),
            phase_image: temp.path().join("demo_phase.tif"),
            fluo_image: temp.path().join("demo_fluo.tif"),
            metadata_path: None,
            data_prep_background_rois_path: None,
            data_prep_roi_coords_path: None,
            data_prep_free_roi_path: None,
            segm_info_path: None,
            size_t: 1,
            size_z: 1,
            time_increment: 1.0,
            physical_size_z: 1.0,
            physical_size_x: 1.0,
            physical_size_y: 1.0,
            segm_is_3d: BTreeMap::new(),
        };

        write_hyperparams_ini(
            &path,
            &position,
            Path::new("/tmp/model.onnx"),
            &SegmentationParams::default(),
            true,
            Some("rust"),
            None,
            Path::new("/tmp/demo_segm_rust.npz"),
        )?;
        write_hyperparams_ini(
            &path,
            &position,
            Path::new("/tmp/model.onnx"),
            &SegmentationParams::default(),
            true,
            Some("rust"),
            None,
            Path::new("/tmp/demo_segm_rust.npz"),
        )?;

        let text = fs::read_to_string(&path)?;
        assert!(text.contains("[segm_rust.metadata.run_number_1]"));
        assert!(text.contains("[segm_rust.init.run_number_1]"));
        assert!(text.contains("[segm_rust.segment.run_number_1]"));
        assert!(text.contains("[segm_rust.metadata.run_number_2]"));
        assert!(!text.contains("track_max_distance_px"));
        assert!(!text.contains("track_min_overlap_px"));
        Ok(())
    }

    #[test]
    fn writes_tracking_hyperparams_using_ioa_threshold() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("segm_hyperparams.ini");
        let position = PositionSpec {
            position_dir: temp.path().join("Position_1"),
            images_dir: temp.path().to_path_buf(),
            basename: "demo_".into(),
            channels: Vec::new(),
            phase_channel: "phase".into(),
            fluo_channel: "fluo".into(),
            phase_image: temp.path().join("demo_phase.tif"),
            fluo_image: temp.path().join("demo_fluo.tif"),
            metadata_path: None,
            data_prep_background_rois_path: None,
            data_prep_roi_coords_path: None,
            data_prep_free_roi_path: None,
            segm_info_path: None,
            size_t: 1,
            size_z: 1,
            time_increment: 1.0,
            physical_size_z: 1.0,
            physical_size_x: 1.0,
            physical_size_y: 1.0,
            segm_is_3d: BTreeMap::new(),
        };

        write_hyperparams_ini(
            &path,
            &position,
            Path::new("/tmp/model.onnx"),
            &SegmentationParams::default(),
            true,
            Some("tracked"),
            Some(&TrackingConfig {
                ioa_threshold: 0.55,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::AreaPrev,
            }),
            Path::new("/tmp/demo_segm_tracked.npz"),
        )?;

        let text = fs::read_to_string(&path)?;
        assert!(text.contains("track = true"));
        assert!(text.contains("IoA_thresh = 0.55"));
        assert!(text.contains("assign_unique_new_IDs = true"));
        assert!(text.contains("denom_overlap_matrix = area_prev"));
        Ok(())
    }

    fn write_test_tiff(path: &Path) -> Result<()> {
        write_test_stack(path, &[1])?;
        Ok(())
    }

    fn write_test_stack(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frame_values {
            let data = vec![*value; 16];
            encoder.write_image::<colortype::Gray16>(4, 4, &data)?;
        }
        Ok(())
    }

    fn write_test_tiff_pixels(path: &Path, pixels: &[u16]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        encoder.write_image::<colortype::Gray16>(4, 4, pixels)?;
        Ok(())
    }

    fn write_test_volume_npz(
        path: &Path,
        frame_values: &[f32],
        size_t: usize,
        size_z: usize,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = NpzWriter::new(file);
        let mut values = Vec::new();
        for value in frame_values {
            for _ in 0..size_z {
                values.extend(vec![*value; 16]);
            }
        }
        let array = Array4::from_shape_vec((size_t, size_z, 4, 4), values)?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;
        Ok(())
    }

    fn header_index(headers: &csv::StringRecord, name: &str) -> usize {
        headers
            .iter()
            .position(|header| header == name)
            .unwrap_or_else(|| panic!("missing CSV header {name}"))
    }
}
