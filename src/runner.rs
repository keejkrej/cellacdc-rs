use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use crate::image_io::{ensure_metadata_file, load_tiff_as_f32, write_mask_npz};
use crate::layout::{discover_experiment, resolve_position, ExperimentSpec, PositionSpec};
use crate::measurements::{rows_from_mask, write_acdc_output_csv};
use crate::model::{CellposeModel, Segmenter};

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
        };
        results.push(run_position_with_segmenter(run_config, segmenter)?);
    }
    Ok(results)
}

pub fn run_position_with_segmenter(
    config: SegmentationRunConfig,
    segmenter: &mut impl Segmenter,
) -> Result<RunResult> {
    let (phase, height, width) =
        load_tiff_as_f32(&config.position.phase_image).with_context(|| {
            format!(
                "Failed to load phase image {}",
                config.position.phase_image.display()
            )
        })?;
    let (fluo, fluo_height, fluo_width) = load_tiff_as_f32(&config.position.fluo_image)
        .with_context(|| {
            format!(
                "Failed to load fluorescence image {}",
                config.position.fluo_image.display()
            )
        })?;

    if (height, width) != (fluo_height, fluo_width) {
        bail!(
            "Input image size mismatch: phase is {}x{}, fluorescence is {}x{}",
            width,
            height,
            fluo_width,
            fluo_height
        );
    }

    let outputs = output_paths(
        &config.position.images_dir,
        &config.position.basename,
        config.segm_endname.as_deref(),
    );
    guard_outputs(&outputs, config.overwrite_policy)?;

    let masks = segmenter.segment_pair(phase, fluo, height, width, &config.params)?;
    let labels_found = masks.iter().copied().max().unwrap_or(0);

    ensure_metadata_file(&config.position, height, width)?;
    write_mask_npz(&outputs.segm_npz_path, &masks, height, width)?;

    let rows = rows_from_mask(&masks, height, width, 0);
    write_acdc_output_csv(&outputs.acdc_output_csv_path, &rows)?;
    write_hyperparams_ini(
        &outputs.segm_hyperparams_ini_path,
        &config.position,
        &config.model_path,
        &config.params,
        config.cpu,
        config.segm_endname.as_deref(),
    )?;

    Ok(RunResult {
        position_dir: config.position.position_dir,
        images_dir: config.position.images_dir,
        outputs,
        labels_found,
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
    })
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
) -> Result<()> {
    let mut content = String::new();
    content.push_str("[workflow]\n");
    content.push_str("type = segmentation\n");
    content.push_str(&format!("phase_channel = {}\n", position.phase_channel));
    content.push_str(&format!("fluo_channel = {}\n", position.fluo_channel));
    content.push_str(&format!("cpu = {}\n", cpu));
    if let Some(endname) = segm_endname {
        if !endname.is_empty() {
            content.push_str(&format!("segm_endname = {}\n", endname));
        }
    }
    content.push_str("\n[cellpose]\n");
    content.push_str(&format!("model_path = {}\n", model_path.display()));
    content.push_str(&format!("tile = {}\n", params.tile));
    content.push_str(&format!("batch_size = {}\n", params.batch_size));
    content.push_str(&format!(
        "cellprob_threshold = {}\n",
        params.cellprob_threshold
    ));
    content.push_str(&format!("niter = {}\n", params.niter));
    content.push_str(&format!("min_size = {}\n", params.min_size));

    fs::write(path, content).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};
    use tempfile::tempdir;

    struct FakeSegmenter {
        mask: Vec<u32>,
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
            mask: vec![
                0, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 2, 2, //
                0, 0, 2, 2, //
            ],
        };

        let result = run_position_with_segmenter(config, &mut segmenter)?;
        assert_eq!(result.labels_found, 2);
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

        let mut segmenter = FakeSegmenter { mask: vec![0; 16] };
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

        let mut segmenter = FakeSegmenter { mask: vec![0; 16] };
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

    fn write_test_tiff(path: &Path) -> Result<()> {
        let image: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_fn(4, 4, |x, y| Luma([((x + y) * 100) as u16]));
        image.save(path)?;
        Ok(())
    }
}
