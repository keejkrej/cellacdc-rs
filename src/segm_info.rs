use anyhow::{bail, Context, Result};
use csv::{ReaderBuilder, StringRecord, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::layout::{
    discover_measurement_experiment, resolve_measurement_position, MeasurementPositionSpec,
};
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
    pub crop_lower_z_slice: Option<usize>,
    pub crop_upper_z_slice: Option<usize>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmInfoEdit {
    pub filename: String,
    pub frame_i: usize,
    pub z_slice_used_data_prep: Option<usize>,
    pub which_z_proj: Option<ZProjectionMode>,
    pub crop_lower_z_slice: Option<usize>,
    pub crop_upper_z_slice: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmInfoInterpolationMode {
    ForwardFill,
    BackwardFill,
    LinearFrames,
}

pub fn load_segm_info(path: &Path) -> Result<SegmInfoTable> {
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let original_headers = reader.headers()?.clone();
    let headers = normalize_segm_info_headers(&original_headers);
    reader.set_headers(headers.clone());
    let required = [
        "filename",
        "frame_i",
        "z_slice_used_dataPrep",
        "which_z_proj",
        "is_from_dataPrep",
    ];
    for name in required {
        if !headers.iter().any(|header| header == name) {
            bail!("Missing _segmInfo column {name:?} in {}", path.display());
        }
    }

    let mut table = SegmInfoTable::default();
    for row in reader.records() {
        let record = row?;
        if is_repeated_header_record(&record, &original_headers) {
            break;
        }
        let row = segm_info_row_map(&headers, &record);
        if required.iter().any(|column| {
            row.get(*column)
                .is_none_or(|value| is_missing_csv_value(value))
        }) {
            continue;
        }
        let filename = row
            .get("filename")
            .cloned()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing filename in {}", path.display()))?;
        let z_slice_used_data_prep = parse_usize(&row, "z_slice_used_dataPrep", path)?;
        let which_z_proj = parse_proj(&row, "which_z_proj", path)?;
        let record = SegmInfoRecord {
            filename,
            frame_i: parse_usize(&row, "frame_i", path)?,
            z_slice_used_data_prep,
            which_z_proj,
            is_from_data_prep: parse_bool(&row, "is_from_dataPrep", path)?,
            z_slice_used_gui: parse_usize_or_default(
                &row,
                "z_slice_used_gui",
                z_slice_used_data_prep,
                path,
            )?,
            which_z_proj_gui: parse_proj_or_default(&row, "which_z_proj_gui", which_z_proj, path)?,
            resegmented_in_gui: parse_bool_or_default(&row, "resegmented_in_gui", false, path)?,
            crop_lower_z_slice: parse_optional_usize(&row, "crop_lower_z_slice")?,
            crop_upper_z_slice: parse_optional_usize(&row, "crop_upper_z_slice")?,
        };
        table
            .records
            .entry((record.filename.clone(), record.frame_i))
            .or_insert(record);
    }
    Ok(table)
}

fn is_repeated_header_record(record: &StringRecord, headers: &StringRecord) -> bool {
    record.len() == headers.len()
        && record
            .iter()
            .zip(headers.iter())
            .all(|(value, header)| value == header)
}

fn segm_info_row_map(headers: &StringRecord, record: &StringRecord) -> BTreeMap<String, String> {
    headers
        .iter()
        .zip(record.iter())
        .map(|(header, value)| (header.to_string(), value.to_string()))
        .collect()
}

fn normalize_segm_info_headers(headers: &StringRecord) -> StringRecord {
    headers
        .iter()
        .map(|header| match header {
            "Unnamed: 0" => "filename",
            "Unnamed: 1" => "frame_i",
            other => other,
        })
        .collect()
}

fn is_missing_csv_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "nan" | "na" | "n/a" | "none" | "null" | "<na>"
    )
}

pub fn save_segm_info(path: impl AsRef<Path>, table: &SegmInfoTable) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Output path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
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
        "crop_lower_z_slice",
        "crop_upper_z_slice",
    ])?;
    for record in table.records.values() {
        writer.write_record([
            record.filename.as_str(),
            &record.frame_i.to_string(),
            &record.z_slice_used_data_prep.to_string(),
            record.which_z_proj.as_str(),
            if record.is_from_data_prep { "1" } else { "0" },
            &record.z_slice_used_gui.to_string(),
            record.which_z_proj_gui.as_str(),
            if record.resegmented_in_gui { "1" } else { "0" },
            &record
                .crop_lower_z_slice
                .map(|value| value.to_string())
                .unwrap_or_default(),
            &record
                .crop_upper_z_slice
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ])?;
    }
    writer.flush()?;
    Ok(path)
}

pub fn clamp_segm_info_z_slices(table: &mut SegmInfoTable, size_z: usize) {
    if size_z <= 1 {
        return;
    }
    let middle_z = size_z / 2;
    for record in table.records.values_mut() {
        if record.z_slice_used_data_prep >= size_z {
            record.z_slice_used_data_prep = middle_z;
        }
        if record.z_slice_used_gui >= size_z {
            record.z_slice_used_gui = middle_z;
        }
    }
}

pub fn apply_segm_info_edit(table: &SegmInfoTable, edit: SegmInfoEdit) -> Result<SegmInfoTable> {
    let mut updated = table.clone();
    let key = (edit.filename.clone(), edit.frame_i);
    let record = updated.records.get_mut(&key).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing _segmInfo entry for file {:?} frame {}",
            edit.filename,
            edit.frame_i
        )
    })?;
    if let Some(z_slice) = edit.z_slice_used_data_prep {
        record.z_slice_used_data_prep = z_slice;
        record.z_slice_used_gui = z_slice;
        record.is_from_data_prep = true;
    }
    if let Some(which_z_proj) = edit.which_z_proj {
        record.which_z_proj = which_z_proj;
        record.which_z_proj_gui = which_z_proj;
        record.is_from_data_prep = true;
    }
    record.crop_lower_z_slice = edit.crop_lower_z_slice;
    record.crop_upper_z_slice = edit.crop_upper_z_slice;
    Ok(updated)
}

pub fn propagate_segm_info_selection(
    table: &SegmInfoTable,
    filename: &str,
    anchor_frame: usize,
    mode: SegmInfoInterpolationMode,
) -> Result<SegmInfoTable> {
    let anchor = table.get(filename, anchor_frame).cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "Missing _segmInfo anchor for {:?} frame {}",
            filename,
            anchor_frame
        )
    })?;
    let mut updated = table.clone();
    let frames = updated
        .records
        .keys()
        .filter_map(|(name, frame)| (name == filename).then_some(*frame))
        .collect::<Vec<_>>();
    for frame in frames {
        let should_apply = match mode {
            SegmInfoInterpolationMode::ForwardFill => frame >= anchor_frame,
            SegmInfoInterpolationMode::BackwardFill => frame <= anchor_frame,
            SegmInfoInterpolationMode::LinearFrames => false,
        };
        if should_apply {
            if let Some(record) = updated.records.get_mut(&(filename.to_string(), frame)) {
                record.z_slice_used_data_prep = anchor.z_slice_used_data_prep;
                record.z_slice_used_gui = anchor.z_slice_used_gui;
                record.which_z_proj = anchor.which_z_proj;
                record.which_z_proj_gui = anchor.which_z_proj_gui;
                record.is_from_data_prep = true;
            }
        }
    }

    if mode == SegmInfoInterpolationMode::LinearFrames {
        let mut same_file = table
            .records
            .values()
            .filter(|record| record.filename == filename)
            .cloned()
            .collect::<Vec<_>>();
        same_file.sort_by_key(|record| record.frame_i);
        let prev = same_file
            .iter()
            .rev()
            .find(|record| record.frame_i < anchor_frame)
            .cloned();
        let next = same_file
            .iter()
            .find(|record| record.frame_i > anchor_frame)
            .cloned();
        match (prev, next) {
            (Some(prev), Some(next)) if next.frame_i > prev.frame_i => {
                let span = (next.frame_i - prev.frame_i) as f32;
                for frame in prev.frame_i..=next.frame_i {
                    let t = (frame - prev.frame_i) as f32 / span;
                    let z = ((1.0 - t) * prev.z_slice_used_data_prep as f32
                        + t * next.z_slice_used_data_prep as f32)
                        .round() as usize;
                    if let Some(record) = updated.records.get_mut(&(filename.to_string(), frame)) {
                        record.z_slice_used_data_prep = z;
                        record.z_slice_used_gui = z;
                        record.which_z_proj = ZProjectionMode::SingleZSlice;
                        record.which_z_proj_gui = ZProjectionMode::SingleZSlice;
                        record.is_from_data_prep = true;
                    }
                }
            }
            (Some(_), Some(_)) => {}
            (Some(_), None) => {
                return propagate_segm_info_selection(
                    table,
                    filename,
                    anchor_frame,
                    SegmInfoInterpolationMode::ForwardFill,
                );
            }
            (None, Some(_)) => {
                return propagate_segm_info_selection(
                    table,
                    filename,
                    anchor_frame,
                    SegmInfoInterpolationMode::BackwardFill,
                );
            }
            (None, None) => {}
        }
    }

    Ok(updated)
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
    let path = spec.segm_info_path.clone().unwrap_or_else(|| {
        spec.images_dir
            .join(format!("{}segmInfo.csv", spec.basename))
    });
    if overwrite_policy == OverwritePolicy::Refuse && path.exists() {
        bail!(
            "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
            path.display()
        );
    }

    let table = build_default_segm_info_table(spec)?;
    save_segm_info(&path, &table)
}

fn parse_usize(row: &BTreeMap<String, String>, key: &str, path: &Path) -> Result<usize> {
    row.get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing {key:?} in {}", path.display()))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Failed to parse {key:?} in {}", path.display()))
}

fn parse_usize_or_default(
    row: &BTreeMap<String, String>,
    key: &str,
    default: usize,
    path: &Path,
) -> Result<usize> {
    match row.get(key).map(|value| value.trim()) {
        Some(value) if !is_missing_csv_value(value) => value
            .parse::<usize>()
            .with_context(|| format!("Failed to parse {key:?} in {}", path.display())),
        _ => Ok(default),
    }
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

fn parse_bool_or_default(
    row: &BTreeMap<String, String>,
    key: &str,
    default: bool,
    path: &Path,
) -> Result<bool> {
    match row.get(key).map(|value| value.trim()) {
        Some(value) if !is_missing_csv_value(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => Ok(true),
            "0" | "false" | "no" => Ok(false),
            _ => Err(anyhow::anyhow!(
                "Invalid boolean value {:?} for column {:?} in {}",
                value,
                key,
                path.display()
            )),
        },
        _ => Ok(default),
    }
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

fn parse_proj_or_default(
    row: &BTreeMap<String, String>,
    key: &str,
    default: ZProjectionMode,
    path: &Path,
) -> Result<ZProjectionMode> {
    match row.get(key).map(|value| value.trim()) {
        Some(value) if !is_missing_csv_value(value) => {
            ZProjectionMode::parse(value).ok_or_else(|| {
                anyhow::anyhow!(
                    "Unsupported z projection {:?} for column {:?} in {}",
                    value,
                    key,
                    path.display()
                )
            })
        }
        _ => Ok(default),
    }
}

fn parse_optional_usize(row: &BTreeMap<String, String>, key: &str) -> Result<Option<usize>> {
    match row.get(key).map(|value| value.trim()) {
        Some("") | None => Ok(None),
        Some(value) => value
            .parse::<usize>()
            .map(Some)
            .with_context(|| format!("Failed to parse optional _segmInfo value {key:?}={value:?}")),
    }
}

pub(crate) fn build_default_segm_info_table(
    spec: &MeasurementPositionSpec,
) -> Result<SegmInfoTable> {
    let middle_z = spec.size_z / 2;
    let mut table = SegmInfoTable::default();
    for channel in &spec.channels {
        let filename = channel
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid channel filename in {}",
                    channel.image_path.display()
                )
            })?;
        for frame_i in 0..spec.size_t {
            table.insert(SegmInfoRecord {
                filename: filename.to_string(),
                frame_i,
                z_slice_used_data_prep: middle_z,
                which_z_proj: ZProjectionMode::SingleZSlice,
                is_from_data_prep: true,
                z_slice_used_gui: middle_z,
                which_z_proj_gui: ZProjectionMode::SingleZSlice,
                resegmented_in_gui: false,
                crop_lower_z_slice: None,
                crop_upper_z_slice: None,
            });
        }
    }
    Ok(table)
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
            data_prep_roi_coords_path: None,
            data_prep_free_roi_path: None,
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

    #[test]
    fn loads_legacy_unnamed_index_segm_info_columns() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_segmInfo.csv");
        std::fs::write(
            &path,
            concat!(
                "Unnamed: 0,Unnamed: 1,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n",
                "demo_phase.tif,0,2,single z-slice,1,2,single z-slice,0\n",
                "demo_phase.tif,0,4,max z-projection,0,4,max z-projection,1\n",
                "demo_phase.tif,1,3,median z-proj.,1,3,median z-proj.,0\n",
            ),
        )?;

        let table = load_segm_info(&path)?;

        assert_eq!(table.records.len(), 2);
        let first = table.get("demo_phase.tif", 0).expect("frame 0 record");
        assert_eq!(first.z_slice_used_data_prep, 2);
        assert_eq!(first.which_z_proj, ZProjectionMode::SingleZSlice);
        assert!(first.is_from_data_prep);
        assert!(!first.resegmented_in_gui);
        let second = table.get("demo_phase.tif", 1).expect("frame 1 record");
        assert_eq!(second.z_slice_used_gui, 3);
        assert_eq!(second.which_z_proj_gui, ZProjectionMode::MedianZProjection);
        Ok(())
    }

    #[test]
    fn ignores_duplicate_appended_segm_info_after_repeated_header() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_segmInfo.csv");
        std::fs::write(
            &path,
            concat!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n",
                "demo_phase.tif,0,2,single z-slice,1,2,single z-slice,0\n",
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n",
                "demo_phase.tif,1,4,max z-projection,0,4,max z-projection,1\n",
            ),
        )?;

        let table = load_segm_info(&path)?;

        assert_eq!(table.records.len(), 1);
        let record = table.get("demo_phase.tif", 0).expect("first table row");
        assert_eq!(record.z_slice_used_data_prep, 2);
        assert_eq!(record.which_z_proj, ZProjectionMode::SingleZSlice);
        assert!(table.get("demo_phase.tif", 1).is_none());
        Ok(())
    }

    #[test]
    fn clamps_out_of_range_z_slices_to_middle_slice() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_segmInfo.csv");
        std::fs::write(
            &path,
            concat!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n",
                "demo_phase.tif,0,9,single z-slice,1,10,single z-slice,0\n",
            ),
        )?;
        let mut table = load_segm_info(&path)?;

        clamp_segm_info_z_slices(&mut table, 5);

        let record = table.get("demo_phase.tif", 0).expect("record");
        assert_eq!(record.z_slice_used_data_prep, 2);
        assert_eq!(record.z_slice_used_gui, 2);
        Ok(())
    }

    #[test]
    fn skips_segm_info_rows_with_missing_required_values() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_segmInfo.csv");
        std::fs::write(
            &path,
            concat!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui,crop_lower_z_slice,crop_upper_z_slice\n",
                "demo_phase.tif,0,2,single z-slice,1,2,single z-slice,0,,\n",
                "demo_phase.tif,1,,single z-slice,1,2,single z-slice,0,,\n",
                "demo_phase.tif,nan,3,single z-slice,1,3,single z-slice,0,,\n",
                ",2,3,single z-slice,1,3,single z-slice,0,,\n",
            ),
        )?;

        let table = load_segm_info(&path)?;

        assert_eq!(table.records.len(), 1);
        let record = table.get("demo_phase.tif", 0).expect("valid row");
        assert_eq!(record.z_slice_used_data_prep, 2);
        assert_eq!(record.crop_lower_z_slice, None);
        assert_eq!(record.crop_upper_z_slice, None);
        Ok(())
    }

    #[test]
    fn loads_old_segm_info_without_gui_mirror_columns() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_segmInfo.csv");
        std::fs::write(
            &path,
            concat!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep\n",
                "demo_phase.tif,0,2,single z-slice,1\n",
                "demo_phase.tif,1,3,max z-projection,0\n",
            ),
        )?;

        let table = load_segm_info(&path)?;

        assert_eq!(table.records.len(), 2);
        let first = table.get("demo_phase.tif", 0).expect("frame 0 record");
        assert_eq!(first.z_slice_used_gui, 2);
        assert_eq!(first.which_z_proj_gui, ZProjectionMode::SingleZSlice);
        assert!(!first.resegmented_in_gui);
        let second = table.get("demo_phase.tif", 1).expect("frame 1 record");
        assert_eq!(second.z_slice_used_gui, 3);
        assert_eq!(second.which_z_proj_gui, ZProjectionMode::MaxZProjection);
        assert!(!second.is_from_data_prep);
        Ok(())
    }
}
