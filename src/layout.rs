use crate::image_io::inspect_image_stack;
use crate::metadata::{read_metadata_summary, DEFAULT_TIME_INCREMENT};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelSpec {
    pub name: String,
    pub image_path: PathBuf,
    pub background_data_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementPositionSpec {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub basename: String,
    pub channels: Vec<ChannelSpec>,
    pub metadata_path: Option<PathBuf>,
    pub data_prep_background_rois_path: Option<PathBuf>,
    pub size_t: usize,
    pub time_increment: f64,
    pub physical_size_x: f64,
    pub physical_size_y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositionSpec {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub basename: String,
    pub channels: Vec<ChannelSpec>,
    pub phase_channel: String,
    pub fluo_channel: String,
    pub phase_image: PathBuf,
    pub fluo_image: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub data_prep_background_rois_path: Option<PathBuf>,
    pub size_t: usize,
    pub time_increment: f64,
    pub physical_size_x: f64,
    pub physical_size_y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSpec {
    pub experiment_dir: PathBuf,
    pub positions: Vec<PositionSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementExperimentSpec {
    pub experiment_dir: PathBuf,
    pub positions: Vec<MeasurementPositionSpec>,
}

pub fn resolve_position(
    path: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
) -> Result<PositionSpec> {
    let phase_channel = phase_channel.into();
    let fluo_channel = fluo_channel.into();
    let base = resolve_measurement_position(path)?;
    let phase_image = channel_image_path(&base.channels, &phase_channel)
        .with_context(|| format!("Failed to resolve phase channel \"{phase_channel}\""))?;
    let fluo_image = channel_image_path(&base.channels, &fluo_channel)
        .with_context(|| format!("Failed to resolve fluorescence channel \"{fluo_channel}\""))?;

    Ok(PositionSpec {
        position_dir: base.position_dir,
        images_dir: base.images_dir,
        basename: base.basename,
        channels: base.channels,
        phase_channel,
        fluo_channel,
        phase_image,
        fluo_image,
        metadata_path: base.metadata_path,
        data_prep_background_rois_path: base.data_prep_background_rois_path,
        size_t: base.size_t,
        time_increment: base.time_increment,
        physical_size_x: base.physical_size_x,
        physical_size_y: base.physical_size_y,
    })
}

pub fn resolve_measurement_position(path: impl AsRef<Path>) -> Result<MeasurementPositionSpec> {
    let input = path.as_ref();
    let (position_dir, images_dir) = normalize_position_path(input)?;
    let files = list_dir_sorted(&images_dir)?;

    let metadata_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("metadata.csv"))
                .unwrap_or(false)
        })
        .cloned();
    let data_prep_background_rois_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("dataPrep_bkgrROIs.json"))
                .unwrap_or(false)
        })
        .cloned();

    let metadata = metadata_path
        .as_ref()
        .map(|path| read_metadata_summary(path))
        .transpose()?
        .unwrap_or_default();

    let basename = metadata
        .basename
        .or_else(|| infer_basename(&files))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Failed to determine Cell-ACDC basename in {}",
                images_dir.display()
            )
        })?;

    let channels = discover_channels(&images_dir, &files, &basename)?;

    if metadata.size_z.unwrap_or(1) > 1 {
        bail!(
            "Metadata in {} declares SizeZ > 1, but this phase only supports 2D timelapse inputs.",
            images_dir.display()
        );
    }

    let first_shape = inspect_image_stack(&channels[0].image_path)?;
    if let Some(size_t) = metadata.size_t {
        if size_t != first_shape.frames {
            bail!(
                "Metadata SizeT ({size_t}) does not match image frame count ({}) in {}",
                first_shape.frames,
                channels[0].image_path.display()
            );
        }
    }

    Ok(MeasurementPositionSpec {
        position_dir,
        images_dir,
        basename,
        channels,
        metadata_path,
        data_prep_background_rois_path,
        size_t: metadata.size_t.unwrap_or(first_shape.frames),
        time_increment: metadata.time_increment.unwrap_or(DEFAULT_TIME_INCREMENT),
        physical_size_x: metadata.physical_size_x.unwrap_or(1.0),
        physical_size_y: metadata.physical_size_y.unwrap_or(1.0),
    })
}

pub fn discover_experiment(
    experiment_dir: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
) -> Result<ExperimentSpec> {
    let experiment_dir = experiment_dir.as_ref().to_path_buf();
    if !experiment_dir.is_dir() {
        bail!(
            "Experiment path is not a directory: {}",
            experiment_dir.display()
        );
    }

    let phase_channel = phase_channel.into();
    let fluo_channel = fluo_channel.into();
    let mut positions = Vec::new();

    for entry in fs::read_dir(&experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let file_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(value) => value,
            None => continue,
        };

        if !file_name.starts_with("Position_") {
            continue;
        }

        positions.push(resolve_position(
            &path,
            phase_channel.clone(),
            fluo_channel.clone(),
        )?);
    }

    positions.sort_by(|a, b| a.position_dir.cmp(&b.position_dir));
    if positions.is_empty() {
        bail!(
            "No Cell-ACDC positions found under {}",
            experiment_dir.display()
        );
    }

    Ok(ExperimentSpec {
        experiment_dir,
        positions,
    })
}

pub fn discover_measurement_experiment(
    experiment_dir: impl AsRef<Path>,
) -> Result<MeasurementExperimentSpec> {
    let experiment_dir = experiment_dir.as_ref().to_path_buf();
    if !experiment_dir.is_dir() {
        bail!(
            "Experiment path is not a directory: {}",
            experiment_dir.display()
        );
    }

    let mut positions = Vec::new();
    for entry in fs::read_dir(&experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let file_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(value) => value,
            None => continue,
        };
        if !file_name.starts_with("Position_") {
            continue;
        }

        positions.push(resolve_measurement_position(&path)?);
    }

    positions.sort_by(|a, b| a.position_dir.cmp(&b.position_dir));
    if positions.is_empty() {
        bail!(
            "No Cell-ACDC positions found under {}",
            experiment_dir.display()
        );
    }

    Ok(MeasurementExperimentSpec {
        experiment_dir,
        positions,
    })
}

fn normalize_position_path(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if !path.exists() {
        bail!("Path does not exist: {}", path.display());
    }

    if path.file_name().and_then(|name| name.to_str()) == Some("Images") {
        let position_dir = path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("Images directory has no parent: {}", path.display()))?;
        return Ok((position_dir, path.to_path_buf()));
    }

    let images_dir = path.join("Images");
    if images_dir.is_dir() {
        return Ok((path.to_path_buf(), images_dir));
    }

    bail!(
        "Expected a Cell-ACDC position directory or Images directory, got {}",
        path.display()
    )
}

fn list_dir_sorted(path: &Path) -> Result<Vec<PathBuf>> {
    let mut items = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("Failed to read {}", path.display()))? {
        items.push(entry?.path());
    }
    items.sort();
    Ok(items)
}

fn infer_basename(files: &[PathBuf]) -> Option<String> {
    let mut candidates: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let is_supported = name.ends_with(".tif")
                || name.ends_with(".tiff")
                || name.ends_with("_aligned.npz")
                || name.ends_with(".h5");
            if !is_supported || name.contains("segm") || name.contains("acdc_output") {
                return None;
            }
            let stem = Path::new(name).file_stem()?.to_str()?;
            Some(stem.to_string())
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let mut prefix = candidates.remove(0);
    for candidate in candidates {
        prefix = longest_common_prefix(&prefix, &candidate);
        if prefix.is_empty() {
            break;
        }
    }

    if prefix.is_empty() {
        None
    } else if prefix.ends_with('_') {
        Some(prefix)
    } else {
        Some(format!("{prefix}_"))
    }
}

fn longest_common_prefix(left: &str, right: &str) -> String {
    let mut out = String::new();
    for (l, r) in left.chars().zip(right.chars()) {
        if l != r {
            break;
        }
        out.push(l);
    }
    out
}

fn discover_channels(
    images_dir: &Path,
    files: &[PathBuf],
    basename: &str,
) -> Result<Vec<ChannelSpec>> {
    let mut candidates = BTreeMap::<String, (usize, PathBuf)>::new();

    for path in files {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(suffix) = file_name.strip_prefix(basename) else {
            continue;
        };

        if should_skip_channel_candidate(suffix) {
            continue;
        }

        let parsed = if let Some(channel) = suffix.strip_suffix("_aligned.h5") {
            Some((channel.to_string(), 0usize))
        } else if let Some(channel) = suffix.strip_suffix(".h5") {
            Some((channel.to_string(), 1usize))
        } else if let Some(channel) = suffix.strip_suffix("_aligned.npz") {
            Some((channel.to_string(), 2usize))
        } else if let Some(channel) = suffix.strip_suffix(".tif") {
            Some((channel.to_string(), 3usize))
        } else {
            suffix
                .strip_suffix(".tiff")
                .map(|channel| (channel.to_string(), 4usize))
        };

        let Some((channel_name, priority)) = parsed else {
            continue;
        };
        if channel_name.trim().is_empty() {
            continue;
        }

        let entry = candidates
            .entry(channel_name)
            .or_insert_with(|| (priority, path.clone()));
        if priority < entry.0 {
            *entry = (priority, path.clone());
        }
    }

    if candidates.is_empty() {
        bail!(
            "No supported Cell-ACDC channel files found under {}",
            images_dir.display()
        );
    }

    let mut channels = Vec::with_capacity(candidates.len());
    for (name, (_, image_path)) in candidates {
        let background_data_path = find_background_data_path(images_dir, &image_path);
        channels.push(ChannelSpec {
            name,
            image_path,
            background_data_path,
        });
    }
    Ok(channels)
}

fn should_skip_channel_candidate(suffix: &str) -> bool {
    suffix.ends_with("metadata.csv")
        || suffix.ends_with("dataPrep_bkgrROIs.json")
        || suffix.ends_with("bkgrRoiData.npz")
        || suffix.starts_with("segm")
        || suffix.starts_with("acdc_output")
        || suffix.starts_with("segm_hyperparams")
}

fn find_background_data_path(images_dir: &Path, image_path: &Path) -> Option<PathBuf> {
    let file_name = image_path.file_name()?.to_str()?;
    let stem = Path::new(file_name).file_stem()?.to_str()?;
    let mut candidates = vec![
        images_dir.join(format!("{stem}_bkgrRoiData.npz")),
        images_dir.join(format!("{file_name}_bkgrRoiData.npz")),
    ];

    if let Some(stem_without_aligned) = stem.strip_suffix("_aligned") {
        candidates.push(images_dir.join(format!(
            "{stem_without_aligned}_aligned.npz_bkgrRoiData.npz"
        )));
        candidates.push(images_dir.join(format!("{stem_without_aligned}_bkgrRoiData.npz")));
    }

    candidates.into_iter().find(|path| path.exists())
}

fn channel_image_path(channels: &[ChannelSpec], channel_name: &str) -> Result<PathBuf> {
    channels
        .iter()
        .find(|channel| channel.name == channel_name)
        .map(|channel| channel.image_path.clone())
        .ok_or_else(|| anyhow::anyhow!("No supported file found for channel \"{channel_name}\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_npy::NpzWriter;
    use std::io::Write;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn resolves_position_from_images_dir() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("test_phase.tif"), &[1])?;
        write_test_stack(&images.join("test_fluo.tif"), &[1])?;
        let mut metadata = fs::File::create(images.join("test_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,test_")?;

        let spec = resolve_position(images, "phase", "fluo")?;
        assert_eq!(spec.basename, "test_");
        assert_eq!(spec.size_t, 1);
        assert_eq!(spec.channels.len(), 2);
        assert!(spec.phase_image.ends_with("test_phase.tif"));
        Ok(())
    }

    #[test]
    fn infers_basename_when_metadata_missing() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("abc_phase.tif"), &[1])?;
        write_test_stack(&images.join("abc_fluo.tif"), &[1])?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert_eq!(spec.basename, "abc_");
        Ok(())
    }

    #[test]
    fn reads_metadata_time_increment_frame_count_and_pixel_size() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_3");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1, 2, 3])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1, 2, 3])?;

        let mut metadata = fs::File::create(images.join("demo_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,demo_")?;
        writeln!(metadata, "SizeT,3")?;
        writeln!(metadata, "SizeZ,1")?;
        writeln!(metadata, "TimeIncrement,30")?;
        writeln!(metadata, "PhysicalSizeX,0.25")?;
        writeln!(metadata, "PhysicalSizeY,0.5")?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert_eq!(spec.size_t, 3);
        assert_eq!(spec.time_increment, 30.0);
        assert_eq!(spec.physical_size_x, 0.25);
        assert_eq!(spec.physical_size_y, 0.5);
        Ok(())
    }

    #[test]
    fn discovers_positions_in_experiment() -> Result<()> {
        let temp = tempdir()?;
        for idx in 1..=2 {
            let images = temp.path().join(format!("Position_{idx}")).join("Images");
            fs::create_dir_all(&images)?;
            write_test_stack(&images.join("demo_phase.tif"), &[1])?;
            write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        }

        let experiment = discover_experiment(temp.path(), "phase", "fluo")?;
        assert_eq!(experiment.positions.len(), 2);
        Ok(())
    }

    #[test]
    fn prefers_aligned_h5_then_h5_then_npz_then_tiff() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_4");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        write_test_npz(&images.join("demo_phase_aligned.npz"), vec![1u16; 6])?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert!(spec.phase_image.ends_with("demo_phase_aligned.npz"));
        Ok(())
    }

    #[test]
    fn inventories_all_channels_and_background_sidecars() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_5");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_gfp.tif"), &[1])?;
        write_test_stack(&images.join("demo_mcherry.tif"), &[1])?;
        fs::write(images.join("demo_gfp_bkgrRoiData.npz"), b"placeholder")?;
        fs::write(images.join("demo_dataPrep_bkgrROIs.json"), b"[]")?;

        let spec = resolve_position(&position, "phase", "gfp")?;
        assert_eq!(spec.channels.len(), 3);
        let gfp = spec
            .channels
            .iter()
            .find(|channel| channel.name == "gfp")
            .unwrap();
        assert!(gfp
            .background_data_path
            .as_ref()
            .unwrap()
            .ends_with("demo_gfp_bkgrRoiData.npz"));
        assert!(spec
            .data_prep_background_rois_path
            .as_ref()
            .unwrap()
            .ends_with("demo_dataPrep_bkgrROIs.json"));
        Ok(())
    }

    fn write_test_stack(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = fs::File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frame_values {
            let data = vec![*value; 6];
            encoder.write_image::<colortype::Gray16>(3, 2, &data)?;
        }
        Ok(())
    }

    fn write_test_npz(path: &Path, data: Vec<u16>) -> Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = NpzWriter::new(file);
        let array = Array2::from_shape_vec((2, 3), data)?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;
        Ok(())
    }
}
