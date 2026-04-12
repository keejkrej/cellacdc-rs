use anyhow::{bail, Context, Result};
use chrono::Local;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
use csv::Reader;
#[cfg(test)]
use ndarray::Array3;
#[cfg(test)]
use ndarray::Array4;
#[cfg(test)]
use ndarray_npy::{NpzReader, NpzWriter};
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::fs::File;

use crate::image_io::{load_image_stack_as_f32, load_image_volume_as_f32, write_mask_npz};
use crate::layout::{discover_experiment, resolve_position, ExperimentSpec, PositionSpec};
use crate::measure::{
    load_measurement_inputs, measurement_position_from_position, write_measurements,
};
use crate::metadata::ensure_position_metadata;
use crate::model::{CellposeModel, Segmenter};
use crate::segm_info::load_segm_info;
use crate::tracking::{track_sequence, TrackingConfig};
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

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationRunConfig {
    pub position: PositionSpec,
    pub model_path: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub cpu: bool,
    pub params: SegmentationParams,
    pub tracking: Option<TrackingConfig>,
    pub stop_frame: Option<usize>,
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
    pub tracking: Option<TrackingConfig>,
    pub stop_frame: Option<usize>,
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
            tracking: config.tracking.clone(),
            stop_frame: config.stop_frame,
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
    guard_outputs(&outputs, config.overwrite_policy)?;
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
            let mask = segmenter.segment_pair(
                phase[start..end].to_vec(),
                fluo[start..end].to_vec(),
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

    let measurement_result = write_measurements(&load_measurement_inputs(
        measurement_position_from_position(&config.position),
        config.segm_endname.as_deref(),
        config.stop_frame,
    )?)?;

    Ok(RunResult {
        position_dir: config.position.position_dir,
        images_dir: config.position.images_dir,
        outputs,
        labels_found: measurement_result.labels_found.max(labels_found),
        frames_processed: frames_to_process.min(measurement_result.frames_processed),
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
        tracking: None,
        stop_frame: None,
    })
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
        Some(value) if !value.trim().is_empty() => format!("segm_{value}"),
        _ => "segm".to_string(),
    }
}

fn output_paths(images_dir: &Path, basename: &str, endname: Option<&str>) -> RunOutputPaths {
    let suffix = match endname {
        Some(value) if !value.trim().is_empty() => format!("_{value}"),
        _ => String::new(),
    };
    let segm_npz_path = images_dir.join(format!("{basename}segm{suffix}.npz"));
    let acdc_output_csv_path = images_dir.join(format!("{basename}acdc_output{suffix}.csv"));
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
    }
    content.push('\n');

    fs::write(path, content).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
                tracking: None,
                stop_frame: None,
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
            tracking: Some(TrackingConfig { ioa_threshold: 0.4 }),
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
            }),
            Path::new("/tmp/demo_segm_tracked.npz"),
        )?;

        let text = fs::read_to_string(&path)?;
        assert!(text.contains("track = true"));
        assert!(text.contains("IoA_thresh = 0.55"));
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
