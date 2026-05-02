use crate::image_io::inspect_image_volume;
use crate::metadata::{read_metadata_summary, DEFAULT_TIME_INCREMENT};
use anyhow::{bail, Context, Result};
use std::cmp::Ordering;
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
    pub data_prep_roi_coords_path: Option<PathBuf>,
    pub data_prep_free_roi_path: Option<PathBuf>,
    pub segm_info_path: Option<PathBuf>,
    pub size_t: usize,
    pub size_z: usize,
    pub time_increment: f64,
    pub physical_size_z: f64,
    pub physical_size_x: f64,
    pub physical_size_y: f64,
    pub segm_is_3d: BTreeMap<String, bool>,
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
    pub data_prep_roi_coords_path: Option<PathBuf>,
    pub data_prep_free_roi_path: Option<PathBuf>,
    pub segm_info_path: Option<PathBuf>,
    pub size_t: usize,
    pub size_z: usize,
    pub time_increment: f64,
    pub physical_size_z: f64,
    pub physical_size_x: f64,
    pub physical_size_y: f64,
    pub segm_is_3d: BTreeMap<String, bool>,
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

pub fn discover_experiment_positions(experiment_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let experiment_dir = experiment_dir.as_ref();
    if !experiment_dir.is_dir() {
        bail!(
            "Experiment path is not a directory: {}",
            experiment_dir.display()
        );
    }
    let mut positions = Vec::new();
    for entry in fs::read_dir(experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.starts_with("Position_") && path.join("Images").is_dir() {
            positions.push(path);
        }
    }
    positions.sort_by(|left, right| compare_position_paths(left, right));
    Ok(positions)
}

pub fn validate_imported_experiment(experiment_dir: impl AsRef<Path>) -> Result<()> {
    let experiment_dir = experiment_dir.as_ref();
    let positions = discover_experiment_positions(experiment_dir)?;
    if positions.is_empty() {
        bail!(
            "No Cell-ACDC positions were discovered under {}",
            experiment_dir.display()
        );
    }
    for position in positions {
        resolve_measurement_position(&position)?;
    }
    Ok(())
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
        data_prep_roi_coords_path: base.data_prep_roi_coords_path,
        data_prep_free_roi_path: base.data_prep_free_roi_path,
        segm_info_path: base.segm_info_path,
        size_t: base.size_t,
        size_z: base.size_z,
        time_increment: base.time_increment,
        physical_size_z: base.physical_size_z,
        physical_size_x: base.physical_size_x,
        physical_size_y: base.physical_size_y,
        segm_is_3d: base.segm_is_3d,
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
    let data_prep_roi_coords_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("dataPrepROIs_coords.csv"))
                .unwrap_or(false)
        })
        .cloned();
    let segm_info_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("segmInfo.csv"))
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

    let data_prep_free_roi_path = find_data_prep_free_roi_path(&files, &basename);
    let channels = discover_channels(&position_dir, &images_dir, &files, &basename)?;

    let first_shape =
        inspect_image_volume(&channels[0].image_path, metadata.size_t, metadata.size_z)?;
    if let Some(size_t) = metadata.size_t {
        if size_t != first_shape.size_t {
            bail!(
                "Metadata SizeT ({size_t}) does not match image frame count ({}) in {}",
                first_shape.size_t,
                channels[0].image_path.display()
            );
        }
    }
    if let Some(size_z) = metadata.size_z {
        if size_z != first_shape.size_z {
            bail!(
                "Metadata SizeZ ({size_z}) does not match image depth ({}) in {}",
                first_shape.size_z,
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
        data_prep_roi_coords_path,
        data_prep_free_roi_path,
        segm_info_path,
        size_t: metadata.size_t.unwrap_or(first_shape.size_t),
        size_z: metadata.size_z.unwrap_or(first_shape.size_z),
        time_increment: metadata.time_increment.unwrap_or(DEFAULT_TIME_INCREMENT),
        physical_size_z: metadata.physical_size_z.unwrap_or(1.0),
        physical_size_x: metadata.physical_size_x.unwrap_or(1.0),
        physical_size_y: metadata.physical_size_y.unwrap_or(1.0),
        segm_is_3d: metadata.segm_is_3d,
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

    positions.sort_by(|a, b| compare_position_paths(&a.position_dir, &b.position_dir));
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

    positions.sort_by(|a, b| compare_position_paths(&a.position_dir, &b.position_dir));
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

pub fn resolve_workflow_targets(
    path: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
) -> Result<Vec<PositionSpec>> {
    let path = path.as_ref();
    let phase_channel = phase_channel.into();
    let fluo_channel = fluo_channel.into();

    if path.is_file() {
        let parent = path.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "Workflow target file has no parent directory: {}",
                path.display()
            )
        })?;
        return Ok(vec![resolve_position(parent, phase_channel, fluo_channel)?]);
    }

    if !path.is_dir() {
        bail!("Workflow target does not exist: {}", path.display());
    }

    if path.file_name().and_then(|name| name.to_str()) == Some("Images")
        || path.join("Images").is_dir()
    {
        return Ok(vec![resolve_position(path, phase_channel, fluo_channel)?]);
    }

    Ok(discover_experiment(path, phase_channel, fluo_channel)?.positions)
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
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(should_skip_python_listdir_entry)
        {
            continue;
        }
        items.push(path);
    }
    items.sort_by(|left, right| compare_paths_by_file_name_natural(left, right));
    Ok(items)
}

fn should_skip_python_listdir_entry(name: &str) -> bool {
    name.starts_with('.')
        || name == "desktop.ini"
        || name == "recovery"
        || name.ends_with(".new.npz")
}

fn find_data_prep_free_roi_path(files: &[PathBuf], basename: &str) -> Option<PathBuf> {
    let expected = format!("{basename}dataPrepFreeRoi.npz");
    files
        .iter()
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some(expected.as_str()))
        .cloned()
        .or_else(|| {
            files
                .iter()
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.ends_with("dataPrepFreeRoi.npz"))
                        .unwrap_or(false)
                })
                .cloned()
        })
}

fn compare_paths_by_file_name_natural(left: &Path, right: &Path) -> Ordering {
    let left_name = left
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let right_name = right
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    natural_compare(left_name, right_name).then_with(|| left.cmp(right))
}

fn natural_compare(left: &str, right: &str) -> Ordering {
    let mut left_iter = NaturalParts::new(left);
    let mut right_iter = NaturalParts::new(right);
    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(NaturalPart::Number(a)), Some(NaturalPart::Number(b))) => match a.cmp(&b) {
                Ordering::Equal => {}
                other => return other,
            },
            (Some(NaturalPart::Text(a)), Some(NaturalPart::Text(b))) => match a.cmp(b) {
                Ordering::Equal => {}
                other => return other,
            },
            (Some(NaturalPart::Number(_)), Some(NaturalPart::Text(_))) => return Ordering::Less,
            (Some(NaturalPart::Text(_)), Some(NaturalPart::Number(_))) => return Ordering::Greater,
            (Some(_), None) => return Ordering::Greater,
            (None, Some(_)) => return Ordering::Less,
            (None, None) => return left.cmp(right),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NaturalPart<'a> {
    Text(&'a str),
    Number(u64),
}

struct NaturalParts<'a> {
    text: &'a str,
    index: usize,
}

impl<'a> NaturalParts<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, index: 0 }
    }
}

impl<'a> Iterator for NaturalParts<'a> {
    type Item = NaturalPart<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.text.len() {
            return None;
        }
        let start = self.index;
        let first = self.text[start..].chars().next()?;
        let is_digit = first.is_ascii_digit();
        while self.index < self.text.len() {
            let ch = self.text[self.index..].chars().next()?;
            if ch.is_ascii_digit() != is_digit {
                break;
            }
            self.index += ch.len_utf8();
        }
        let part = &self.text[start..self.index];
        if is_digit {
            Some(NaturalPart::Number(part.parse().unwrap_or(u64::MAX)))
        } else {
            Some(NaturalPart::Text(part))
        }
    }
}

fn infer_basename(files: &[PathBuf]) -> Option<String> {
    let mut candidates: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let is_supported = name.ends_with(".tif")
                || name.ends_with(".tiff")
                || name.ends_with("_aligned.npz")
                || name.ends_with("_aligned.npy")
                || name.ends_with(".npy")
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
    position_dir: &Path,
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

        let suffix = strip_legacy_position_prefix(suffix, position_dir);

        if should_skip_channel_candidate(suffix) {
            continue;
        }

        let parsed = if let Some(channel) = suffix.strip_suffix("_aligned.h5") {
            Some((channel.to_string(), 0usize))
        } else if let Some(channel) = suffix.strip_suffix(".h5") {
            Some((channel.to_string(), 1usize))
        } else if let Some(channel) = suffix.strip_suffix("_aligned.npz") {
            Some((channel.to_string(), 2usize))
        } else if let Some(channel) = suffix.strip_suffix("_aligned.npy") {
            Some((channel.to_string(), 3usize))
        } else if let Some(channel) = suffix.strip_suffix(".tif") {
            Some((channel.to_string(), 4usize))
        } else if let Some(channel) = suffix.strip_suffix(".tiff") {
            Some((channel.to_string(), 5usize))
        } else {
            suffix
                .strip_suffix(".npy")
                .map(|channel| (channel.to_string(), 6usize))
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

fn strip_legacy_position_prefix<'a>(suffix: &'a str, position_dir: &Path) -> &'a str {
    let Some(position_number) = position_number(position_dir) else {
        return suffix;
    };
    let Some(rest) = suffix.strip_prefix('s') else {
        return suffix;
    };
    let Some((digits, after_digits)) = rest.split_once('_') else {
        return suffix;
    };
    if digits.is_empty() {
        return suffix;
    }
    match digits.parse::<usize>() {
        Ok(value) if value == position_number => after_digits,
        _ => suffix,
    }
}

fn position_number(position_dir: &Path) -> Option<usize> {
    position_dir
        .file_name()
        .and_then(|name| name.to_str())?
        .strip_prefix("Position_")?
        .parse::<usize>()
        .ok()
}

fn compare_position_paths(left: &Path, right: &Path) -> Ordering {
    match (position_number(left), position_number(right)) {
        (Some(left_number), Some(right_number)) => {
            left_number.cmp(&right_number).then_with(|| left.cmp(right))
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left.cmp(right),
    }
}

fn should_skip_channel_candidate(suffix: &str) -> bool {
    suffix.ends_with("metadata.csv")
        || suffix.ends_with("dataPrep_bkgrROIs.json")
        || suffix.ends_with("dataPrepROIs_coords.csv")
        || suffix.ends_with("dataPrepFreeRoi.npz")
        || suffix.ends_with("bkgrRoiData.npz")
        || suffix.ends_with("align_shift.npy")
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
    fn ignores_python_listdir_excluded_files_when_inferring_basename() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join(".hidden_phase.tif"), &[1])?;
        write_test_stack(&images.join("desktop.ini"), &[1])?;
        fs::create_dir_all(images.join("recovery"))?;
        write_test_npz(&images.join("abc_phase.new.npz"), vec![1u16; 6])?;
        write_test_stack(&images.join("abc_phase.tif"), &[1])?;
        write_test_stack(&images.join("abc_fluo.tif"), &[1])?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert_eq!(spec.basename, "abc_");
        assert_eq!(spec.channels.len(), 2);
        Ok(())
    }

    #[test]
    fn discovers_sidecars_in_python_listdir_order() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        let mut metadata = fs::File::create(images.join("demo_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,demo_")?;
        fs::write(images.join("sample10_dataPrepROIs_coords.csv"), b"late")?;
        fs::write(images.join("sample2_dataPrepROIs_coords.csv"), b"early")?;
        fs::write(images.join("sample10_dataPrep_bkgrROIs.json"), b"[]")?;
        fs::write(images.join("sample2_dataPrep_bkgrROIs.json"), b"[]")?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert!(spec
            .data_prep_roi_coords_path
            .as_ref()
            .unwrap()
            .ends_with("sample2_dataPrepROIs_coords.csv"));
        assert!(spec
            .data_prep_background_rois_path
            .as_ref()
            .unwrap()
            .ends_with("sample2_dataPrep_bkgrROIs.json"));
        Ok(())
    }

    #[test]
    fn dataprep_free_roi_prefers_python_basename_path() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        let mut metadata = fs::File::create(images.join("demo_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,demo_")?;
        fs::write(images.join("other_dataPrepFreeRoi.npz"), b"wrong")?;
        fs::write(images.join("demo_dataPrepFreeRoi.npz"), b"right")?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert!(spec
            .data_prep_free_roi_path
            .as_ref()
            .unwrap()
            .ends_with("demo_dataPrepFreeRoi.npz"));
        Ok(())
    }

    #[test]
    fn strips_legacy_position_token_from_channel_names() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_s01_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_s01_fluo.tif"), &[1])?;
        let mut metadata = fs::File::create(images.join("demo_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,demo_")?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert_eq!(spec.basename, "demo_");
        assert!(spec.phase_image.ends_with("demo_s01_phase.tif"));
        assert!(spec.fluo_image.ends_with("demo_s01_fluo.tif"));
        assert!(spec.channels.iter().any(|channel| channel.name == "phase"));
        assert!(spec.channels.iter().any(|channel| channel.name == "fluo"));
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
    fn discovers_position_paths_in_natural_order() -> Result<()> {
        let temp = tempdir()?;
        for idx in [10, 2, 1] {
            fs::create_dir_all(temp.path().join(format!("Position_{idx}")).join("Images"))?;
        }

        let positions = discover_experiment_positions(temp.path())?;
        let names = positions
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Position_1", "Position_2", "Position_10"]);
        Ok(())
    }

    #[test]
    fn prefers_aligned_h5_then_h5_then_npz_then_npy_then_tiff() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_4");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        fs::write(images.join("demo_phase_aligned.npy"), b"placeholder")?;
        write_test_npz(&images.join("demo_phase_aligned.npz"), vec![1u16; 6])?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert!(spec.phase_image.ends_with("demo_phase_aligned.npz"));
        Ok(())
    }

    #[test]
    fn discovers_old_python_aligned_npy_channels() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_4");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1])?;
        write_test_stack(&images.join("demo_fluo.tif"), &[1])?;
        write_test_npy(&images.join("demo_phase_aligned.npy"), vec![1u16; 6])?;

        let spec = resolve_position(&position, "phase", "fluo")?;

        assert!(spec.phase_image.ends_with("demo_phase_aligned.npy"));
        assert!(spec.fluo_image.ends_with("demo_fluo.tif"));
        Ok(())
    }

    #[test]
    fn discovers_plain_npy_channels_and_infers_basename() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_4");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_npy(&images.join("demo_phase.npy"), vec![1u16; 6])?;
        write_test_npy(&images.join("demo_fluo.npy"), vec![2u16; 6])?;
        write_test_npy(&images.join("demo_align_shift.npy"), vec![0u16; 6])?;

        let spec = resolve_position(&position, "phase", "fluo")?;

        assert_eq!(spec.basename, "demo_");
        assert_eq!(spec.channels.len(), 2);
        assert!(spec.phase_image.ends_with("demo_phase.npy"));
        assert!(spec.fluo_image.ends_with("demo_fluo.npy"));
        assert!(!spec
            .channels
            .iter()
            .any(|channel| channel.name == "align_shift"));
        Ok(())
    }

    #[test]
    fn channel_discovery_uses_python_file_priority() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_4");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        let files = vec![
            images.join("demo_phase.tif"),
            images.join("demo_phase.npy"),
            images.join("demo_phase_aligned.npy"),
            images.join("demo_phase_aligned.npz"),
            images.join("demo_phase.h5"),
        ];

        let channels = discover_channels(&position, &images, &files, "demo_")?;
        assert_eq!(channels[0].name, "phase");
        assert!(channels[0].image_path.ends_with("demo_phase.h5"));

        let files = vec![
            images.join("demo_phase.tif"),
            images.join("demo_phase.npy"),
            images.join("demo_phase_aligned.npy"),
            images.join("demo_phase_aligned.npz"),
            images.join("demo_phase.h5"),
            images.join("demo_phase_aligned.h5"),
        ];
        let channels = discover_channels(&position, &images, &files, "demo_")?;
        assert!(channels[0].image_path.ends_with("demo_phase_aligned.h5"));

        let files = vec![images.join("demo_phase.npy"), images.join("demo_phase.tif")];
        let channels = discover_channels(&position, &images, &files, "demo_")?;
        assert!(channels[0].image_path.ends_with("demo_phase.tif"));
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

    fn write_test_npy(path: &Path, data: Vec<u16>) -> Result<()> {
        let array = Array2::from_shape_vec((2, 3), data)?;
        ndarray_npy::write_npy(path, &array)?;
        Ok(())
    }
}
