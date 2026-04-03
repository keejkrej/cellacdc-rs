use anyhow::{bail, Context, Result};
use csv::{ReaderBuilder, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::layout::{discover_measurement_experiment, resolve_measurement_position, MeasurementPositionSpec};
use crate::runner::OverwritePolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZProjectionMode {
    SingleZSlice,
    MaxZProjection,
    MeanZProjection,
    MedianZProjection,
}

impl ZProjectionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleZSlice => "single z-slice",
            Self::MaxZProjection => "max z-projection",
            Self::MeanZProjection => "mean z-projection",
            Self::MedianZProjection => "median z-proj.",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "single z-slice" => Some(Self::SingleZSlice),
            "max z-projection" => Some(Self::MaxZProjection),
            "mean z-projection" => Some(Self::MeanZProjection),
            "median z-proj." => Some(Self::MedianZProjection),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmInfoRecord {
    pub filename: String,
    pub frame_i: usize,
    pub z_slice_used_data_prep: usize,
    pub which_z_proj: ZProjectionMode,
    pub is_from_data_prep: bool,
    pub z_slice_used_gui: usize,
    pub which_z_proj_gui: ZProjectionMode,
    pub resegmented_in_gui: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegmInfoTable {
    pub records: BTreeMap<(String, usize), SegmInfoRecord>,
}

impl SegmInfoTable {
    pub fn get(&self, filename: &str, frame_i: usize) -> Option<&SegmInfoRecord> {
        self.records.get(&(filename.to_string(), frame_i))
    }

    pub fn insert(&mut self, record: SegmInfoRecord) {
        self.records
            .insert((record.filename.clone(), record.frame_i), record);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareSegmInfoTarget {
    Position(PathBuf),
    Experiment(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareZStackSegmInfoConfig {
    pub target: PrepareSegmInfoTarget,
    pub overwrite_policy: OverwritePolicy,
}

pub fn load_segm_info(path: &Path) -> Result<SegmInfoTable> {
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let required = [
        "filename",
        "frame_i",
        "z_slice_used_dataPrep",
        "which_z_proj",
        "is_from_dataPrep",
        "z_slice_used_gui",
        "which_z_proj_gui",
        "resegmented_in_gui",
    ];
    for name in required {
        if !headers.iter().any(|header| header == name) {
            bail!("Missing _segmInfo column {name:?} in {}", path.display());
        }
    }

    let mut table = SegmInfoTable::default();
    for row in reader.deserialize::<BTreeMap<String, String>>() {
        let row = row?;
        let filename = row
            .get("filename")
            .cloned()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing filename in {}", path.display()))?;
        let record = SegmInfoRecord {
            filename,
            frame_i: parse_usize(&row, "frame_i", path)?,
            z_slice_used_data_prep: parse_usize(&row, "z_slice_used_dataPrep", path)?,
            which_z_proj: parse_proj(&row, "which_z_proj", path)?,
            is_from_data_prep: parse_bool(&row, "is_from_dataPrep", path)?,
            z_slice_used_gui: parse_usize(&row, "z_slice_used_gui", path)?,
            which_z_proj_gui: parse_proj(&row, "which_z_proj_gui", path)?,
            resegmented_in_gui: parse_bool(&row, "resegmented_in_gui", path)?,
        };
        table.insert(record);
    }
    Ok(table)
}

pub fn prepare_zstack_segm_info(config: PrepareZStackSegmInfoConfig) -> Result<Vec<PathBuf>> {
    match config.target {
        PrepareSegmInfoTarget::Position(path) => {
            let spec = resolve_measurement_position(path)?;
            Ok(vec![prepare_position_zstack_segm_info(
                &spec,
                config.overwrite_policy,
            )?])
        }
        PrepareSegmInfoTarget::Experiment(path) => {
            let experiment = discover_measurement_experiment(path)?;
            let mut paths = Vec::with_capacity(experiment.positions.len());
            for position in experiment.positions {
                paths.push(prepare_position_zstack_segm_info(
                    &position,
                    config.overwrite_policy,
                )?);
            }
            Ok(paths)
        }
    }
}

pub fn prepare_position_zstack_segm_info(
    spec: &MeasurementPositionSpec,
    overwrite_policy: OverwritePolicy,
) -> Result<PathBuf> {
    if spec.size_z <= 1 {
        bail!(
            "Position {} is not a z-stack (SizeZ <= 1)",
            spec.position_dir.display()
        );
    }
    let path = spec
        .segm_info_path
        .clone()
        .unwrap_or_else(|| spec.images_dir.join(format!("{}segmInfo.csv", spec.basename)));
    if overwrite_policy == OverwritePolicy::Refuse && path.exists() {
        bail!(
            "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
            path.display()
        );
    }

    let middle_z = spec.size_z / 2;
    let mut writer =
        Writer::from_path(&path).with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record([
        "filename",
        "frame_i",
        "z_slice_used_dataPrep",
        "which_z_proj",
        "is_from_dataPrep",
        "z_slice_used_gui",
        "which_z_proj_gui",
        "resegmented_in_gui",
    ])?;
    for channel in &spec.channels {
        let filename = channel
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid channel filename in {}", channel.image_path.display()))?;
        for frame_i in 0..spec.size_t {
            writer.write_record([
                filename,
                &frame_i.to_string(),
                &middle_z.to_string(),
                ZProjectionMode::SingleZSlice.as_str(),
                "1",
                &middle_z.to_string(),
                ZProjectionMode::SingleZSlice.as_str(),
                "0",
            ])?;
        }
    }
    writer.flush()?;
    Ok(path)
}

fn parse_usize(row: &BTreeMap<String, String>, key: &str, path: &Path) -> Result<usize> {
    row.get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing {key:?} in {}", path.display()))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Failed to parse {key:?} in {}", path.display()))
}

fn parse_bool(row: &BTreeMap<String, String>, key: &str, path: &Path) -> Result<bool> {
    let value = row
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing {key:?} in {}", path.display()))?;
    Ok(matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    ))
}

fn parse_proj(row: &BTreeMap<String, String>, key: &str, path: &Path) -> Result<ZProjectionMode> {
    let value = row
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing {key:?} in {}", path.display()))?;
    ZProjectionMode::parse(value).ok_or_else(|| {
        anyhow::anyhow!(
            "Unsupported z projection {:?} for column {:?} in {}",
            value,
            key,
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_python_compatible_template_for_position() -> Result<()> {
        let temp = tempdir()?;
        let images_dir = temp.path().join("Position_1").join("Images");
        std::fs::create_dir_all(&images_dir)?;
        let spec = MeasurementPositionSpec {
            position_dir: temp.path().join("Position_1"),
            images_dir: images_dir.clone(),
            basename: "demo_".into(),
            channels: vec![crate::layout::ChannelSpec {
                name: "phase".into(),
                image_path: images_dir.join("demo_phase.tif"),
                background_data_path: None,
            }],
            metadata_path: None,
            data_prep_background_rois_path: None,
            segm_info_path: None,
            size_t: 3,
            size_z: 5,
            time_increment: 1.0,
            physical_size_z: 1.0,
            physical_size_x: 1.0,
            physical_size_y: 1.0,
            segm_is_3d: BTreeMap::new(),
        };

        let path = prepare_position_zstack_segm_info(&spec, OverwritePolicy::Refuse)?;
        let table = load_segm_info(&path)?;
        assert_eq!(table.records.len(), 3);
        let record = table.get("demo_phase.tif", 0).unwrap();
        assert_eq!(record.z_slice_used_data_prep, 2);
        assert_eq!(record.which_z_proj, ZProjectionMode::SingleZSlice);
        Ok(())
    }
}
