use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::{NpzReader, NpzWriter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentConfig {
    pub position_dir: PathBuf,
    pub target_channel: String,
    pub reference_channel: Option<String>,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CropConfig {
    pub position_dir: PathBuf,
    pub channels: Vec<String>,
    pub output_dir: PathBuf,
    pub x_range: (usize, usize),
    pub y_range: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeCropConfig {
    pub position_dir: PathBuf,
    pub channels: Vec<String>,
    pub output_dir: PathBuf,
    pub frame_range: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZCropConfig {
    pub position_dir: PathBuf,
    pub channels: Vec<String>,
    pub output_dir: PathBuf,
    pub z_range: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundRoiRect {
    pub pos: [f32; 2],
    pub size: [f32; 2],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackgroundRoiSet {
    pub items: Vec<BackgroundRoiRect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepOutputPaths {
    pub primary_path: PathBuf,
    pub secondary_paths: Vec<PathBuf>,
}

pub type BackgroundRoiArchive = BTreeMap<String, ArrayD<f32>>;

pub fn read_background_roi_json(path: impl AsRef<Path>) -> Result<BackgroundRoiSet> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    let mut items = Vec::new();
    if let Some(array) = value.as_array() {
        for item in array {
            let pos = parse_pair(
                item.get("pos")
                    .or_else(|| item.get("state").and_then(|state| state.get("pos"))),
            );
            let size = parse_pair(
                item.get("size")
                    .or_else(|| item.get("state").and_then(|state| state.get("size"))),
            );
            if let (Some(pos), Some(size)) = (pos, size) {
                items.push(BackgroundRoiRect { pos, size });
            }
        }
    }
    Ok(BackgroundRoiSet { items })
}

pub fn write_background_roi_json(path: impl AsRef<Path>, rois: &BackgroundRoiSet) -> Result<()> {
    let path = path.as_ref();
    let payload = serde_json::to_string_pretty(&rois.items)
        .with_context(|| format!("Failed to serialize ROI data for {}", path.display()))?;
    std::fs::write(path, payload).with_context(|| format!("Failed to write {}", path.display()))
}

pub fn read_background_roi_npz(path: impl AsRef<Path>) -> Result<BackgroundRoiArchive> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader =
        NpzReader::new(file).with_context(|| format!("Failed to read {}", path.display()))?;
    let mut arrays = BTreeMap::new();
    for name in reader.names()? {
        let array: ArrayD<f32> = reader.by_name(&name)?;
        arrays.insert(name.trim_end_matches(".npy").to_string(), array);
    }
    Ok(arrays)
}

pub fn write_background_roi_npz(
    path: impl AsRef<Path>,
    arrays: &BackgroundRoiArchive,
) -> Result<()> {
    let path = path.as_ref();
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = NpzWriter::new(file);
    for (name, array) in arrays {
        writer.add_array(name, array)?;
    }
    writer.finish()?;
    Ok(())
}

fn parse_pair(value: Option<&Value>) -> Option<[f32; 2]> {
    let array = value?.as_array()?;
    if array.len() < 2 {
        return None;
    }
    Some([array[0].as_f64()? as f32, array[1].as_f64()? as f32])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use tempfile::tempdir;

    #[test]
    fn roundtrips_background_roi_json() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo_dataPrep_bkgrROIs.json");
        let rois = BackgroundRoiSet {
            items: vec![BackgroundRoiRect {
                pos: [10.0, 12.0],
                size: [25.0, 18.0],
            }],
        };
        write_background_roi_json(&path, &rois)?;
        let loaded = read_background_roi_json(&path)?;
        assert_eq!(loaded, rois);
        Ok(())
    }

    #[test]
    fn roundtrips_background_roi_npz() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo_phase_bkgrRoiData.npz");
        let mut archive = BackgroundRoiArchive::new();
        archive.insert(
            "base".to_string(),
            Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn(),
        );
        write_background_roi_npz(&path, &archive)?;
        let loaded = read_background_roi_npz(&path)?;
        assert_eq!(loaded["base"].shape(), &[3]);
        Ok(())
    }
}
