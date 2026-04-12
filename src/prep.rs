use crate::image_io::{load_image_stack_as_f32, load_image_volume_as_f32, StackShape, VolumeShape};
use crate::layout::resolve_measurement_position;
use crate::metadata::read_metadata_map;
use crate::segm_info::{
    build_default_segm_info_table, load_segm_info, save_segm_info, SegmInfoTable,
};
use anyhow::{anyhow, bail, Context, Result};
use csv::Writer;
use ndarray::{s, Array2, Array3, Array4, ArrayD, Ix2, Ix3, Ix4};
use ndarray_npy::{write_npy, NpzReader, NpzWriter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tiff::encoder::{colortype, TiffEncoder};

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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CropRoiRect {
    pub roi_id: usize,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CropRoiCoordsTable {
    pub rois: Vec<CropRoiRect>,
    pub cropped_roi_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreehandRoiMask {
    pub bbox_yxxy: (usize, usize, usize, usize),
    pub local_mask: Array2<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentShiftSet {
    pub shifts_xy: Vec<[i32; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentRunConfig {
    pub position_dir: PathBuf,
    pub reference_channel: String,
    pub channels_to_align: Vec<String>,
    pub frame_range: Option<(usize, usize)>,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentRunResult {
    pub aligned_outputs: Vec<PathBuf>,
    pub shifts_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CropPreview {
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CropSaveConfig {
    pub position_dir: PathBuf,
    pub channels: Vec<String>,
    pub frame_range: Option<(usize, usize)>,
    pub z_range: Option<(usize, usize)>,
    pub crop_rois: Vec<CropRoiRect>,
    pub background_rois: BackgroundRoiSet,
    pub free_roi: Option<FreehandRoiMask>,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CropSaveResult {
    pub written_files: Vec<PathBuf>,
    pub updated_metadata_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataPrepState {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub available_channels: Vec<String>,
    pub active_channel: String,
    pub crop_rois: Vec<CropRoiRect>,
    pub background_rois: BackgroundRoiSet,
    pub free_roi: Option<FreehandRoiMask>,
    pub segm_info: SegmInfoTable,
    pub aligned_channel_paths: BTreeMap<String, PathBuf>,
    pub alignment_shifts_path: Option<PathBuf>,
}

pub type BackgroundRoiArchive = BTreeMap<String, ArrayD<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageScalarType {
    U8,
    U16,
    U32,
    F32,
}

pub fn load_data_prep_state(
    position_dir: impl AsRef<Path>,
    active_channel: Option<&str>,
) -> Result<DataPrepState> {
    let spec = resolve_measurement_position(position_dir)?;
    let crop_rois = spec
        .data_prep_roi_coords_path
        .as_ref()
        .map(load_crop_roi_coords_csv)
        .transpose()?
        .unwrap_or_default()
        .rois;
    let background_rois = spec
        .data_prep_background_rois_path
        .as_ref()
        .map(read_background_roi_json)
        .transpose()?
        .unwrap_or_default();
    let free_roi = spec
        .data_prep_free_roi_path
        .as_ref()
        .map(read_freehand_roi_npz)
        .transpose()?
        .flatten();
    let segm_info = if let Some(path) = spec.segm_info_path.as_ref() {
        load_segm_info(path)?
    } else if spec.size_z > 1 {
        build_default_segm_info_table(&spec)?
    } else {
        SegmInfoTable::default()
    };
    let available_channels = spec
        .channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect::<Vec<_>>();
    let active_channel = active_channel
        .and_then(|selected| {
            available_channels
                .iter()
                .find(|channel| channel.as_str() == selected)
                .cloned()
        })
        .or_else(|| available_channels.first().cloned())
        .unwrap_or_default();
    let aligned_channel_paths = spec
        .channels
        .iter()
        .filter_map(|channel| {
            channel
                .image_path
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| name.contains("_aligned."))
                .map(|_| (channel.name.clone(), channel.image_path.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let alignment_shifts_path = {
        let path = spec
            .images_dir
            .join(format!("{}align_shift.npy", spec.basename));
        path.exists().then_some(path)
    };

    Ok(DataPrepState {
        position_dir: spec.position_dir,
        images_dir: spec.images_dir,
        available_channels,
        active_channel,
        crop_rois,
        background_rois,
        free_roi,
        segm_info,
        aligned_channel_paths,
        alignment_shifts_path,
    })
}

pub fn load_crop_roi_coords_csv(path: impl AsRef<Path>) -> Result<CropRoiCoordsTable> {
    let path = path.as_ref();
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let mut rows = BTreeMap::<usize, BTreeMap<String, usize>>::new();
    for record in reader.deserialize::<BTreeMap<String, String>>() {
        let record = record?;
        let roi_id = record
            .get("roi_id")
            .map(String::as_str)
            .unwrap_or("0")
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Failed to parse roi_id in {}", path.display()))?;
        let description = record
            .get("description")
            .cloned()
            .ok_or_else(|| anyhow!("Missing ROI description in {}", path.display()))?;
        let value = record
            .get("value")
            .map(String::as_str)
            .unwrap_or("0")
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Failed to parse ROI value in {}", path.display()))?;
        rows.entry(roi_id).or_default().insert(description, value);
    }
    let mut table = CropRoiCoordsTable::default();
    for (roi_id, values) in rows {
        let x_left = values.get("x_left").copied().unwrap_or(0);
        let x_right = values.get("x_right").copied().unwrap_or(x_left);
        let y_top = values.get("y_top").copied().unwrap_or(0);
        let y_bottom = values.get("y_bottom").copied().unwrap_or(y_top);
        if x_right <= x_left || y_bottom <= y_top {
            continue;
        }
        table.rois.push(CropRoiRect {
            roi_id,
            x: x_left,
            y: y_top,
            width: x_right - x_left,
            height: y_bottom - y_top,
        });
        if values.get("cropped").copied().unwrap_or(0) > 0 {
            table.cropped_roi_ids.push(roi_id);
        }
    }
    table.rois.sort_by_key(|roi| roi.roi_id);
    table.cropped_roi_ids.sort_unstable();
    Ok(table)
}

pub fn save_crop_roi_coords_csv(
    path: impl AsRef<Path>,
    table: &CropRoiCoordsTable,
) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    let tmp_path = temp_write_path(&path);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let mut writer = Writer::from_path(&tmp_path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record(["roi_id", "description", "value"])?;
    for roi in &table.rois {
        let cropped = usize::from(table.cropped_roi_ids.contains(&roi.roi_id));
        for (description, value) in [
            ("x_left", roi.x),
            ("x_right", roi.x + roi.width),
            ("y_top", roi.y),
            ("y_bottom", roi.y + roi.height),
            ("cropped", cropped),
        ] {
            writer.write_record([
                roi.roi_id.to_string(),
                description.to_string(),
                value.to_string(),
            ])?;
        }
    }
    writer.flush()?;
    promote_temp_file(&tmp_path, &path)?;
    Ok(path)
}

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
    let tmp_path = temp_write_path(path);
    let payload = serde_json::to_string_pretty(&rois.items)
        .with_context(|| format!("Failed to serialize ROI data for {}", path.display()))?;
    std::fs::write(&tmp_path, payload)
        .with_context(|| format!("Failed to write {}", tmp_path.display()))?;
    promote_temp_file(&tmp_path, path)
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
    let tmp_path = temp_write_path(path);
    let file =
        File::create(&tmp_path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = NpzWriter::new(file);
    for (name, array) in arrays {
        writer.add_array(name, array)?;
    }
    writer.finish()?;
    promote_temp_file(&tmp_path, path)
}

pub fn read_freehand_roi_npz(path: impl AsRef<Path>) -> Result<Option<FreehandRoiMask>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader =
        NpzReader::new(file).with_context(|| format!("Failed to read {}", path.display()))?;
    let names = reader.names()?;
    let Some(name) = names.first() else {
        return Ok(None);
    };
    let bbox_key = name.trim_end_matches(".npy");
    let coords = bbox_key
        .split('_')
        .map(|coord| coord.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| {
            format!(
                "Invalid free ROI bbox key {bbox_key:?} in {}",
                path.display()
            )
        })?;
    if coords.len() != 4 {
        bail!(
            "Invalid free ROI bbox key {bbox_key:?} in {}",
            path.display()
        );
    }
    let mask: Array2<bool> = reader.by_name(name)?;
    Ok(Some(FreehandRoiMask {
        bbox_yxxy: (coords[1], coords[0], coords[3], coords[2]),
        local_mask: mask,
    }))
}

pub fn write_freehand_roi_npz(path: impl AsRef<Path>, roi: &FreehandRoiMask) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    let tmp_path = temp_write_path(&path);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let file = File::create(&tmp_path)?;
    let mut writer = NpzWriter::new_compressed(file);
    let (y0, x0, y1, x1) = roi.bbox_yxxy;
    let key = format!("{x0}_{y0}_{x1}_{y1}");
    writer.add_array(&key, &roi.local_mask)?;
    writer.finish()?;
    promote_temp_file(&tmp_path, &path)?;
    Ok(path)
}

pub fn remove_freehand_roi_npz(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("Failed to remove {}", path.display()))?;
    }
    Ok(())
}

pub fn compute_alignment_shifts(config: &AlignmentRunConfig) -> Result<AlignmentShiftSet> {
    let spec = resolve_measurement_position(&config.position_dir)?;
    let channel = spec
        .channels
        .iter()
        .find(|channel| channel.name == config.reference_channel)
        .ok_or_else(|| {
            anyhow!(
                "Unknown alignment reference channel {:?}",
                config.reference_channel
            )
        })?;
    let (frames, _, _, _) = load_channel_cube(&channel.image_path, spec.size_t, spec.size_z)?;
    if frames.is_empty() {
        return Ok(AlignmentShiftSet {
            shifts_xy: Vec::new(),
        });
    }
    let mut shifts_xy = vec![[0, 0]; frames.len()];
    let reference = &frames[0];
    let max_shift = reference.shape()[0]
        .min(reference.shape()[1])
        .saturating_div(8)
        .clamp(1, 32) as i32;
    let (range_start, range_end) = normalize_range(config.frame_range, frames.len())?;
    for frame_i in range_start..range_end {
        if frame_i == 0 {
            continue;
        }
        let (dy, dx) = estimate_xy_shift(reference, &frames[frame_i], max_shift);
        shifts_xy[frame_i] = [dx, dy];
    }
    Ok(AlignmentShiftSet { shifts_xy })
}

pub fn apply_alignment(
    config: AlignmentRunConfig,
    shifts: &AlignmentShiftSet,
) -> Result<AlignmentRunResult> {
    let spec = resolve_measurement_position(&config.position_dir)?;
    if shifts.shifts_xy.len() != spec.size_t.max(1) {
        bail!(
            "Alignment shifts count {} does not match SizeT {} in {}",
            shifts.shifts_xy.len(),
            spec.size_t,
            spec.position_dir.display()
        );
    }
    let shifts_path = spec
        .images_dir
        .join(format!("{}align_shift.npy", spec.basename));
    if shifts_path.exists() && !config.overwrite {
        bail!(
            "Alignment shifts already exist at {}. Re-run with overwrite enabled to replace them.",
            shifts_path.display()
        );
    }
    let mut aligned_outputs = Vec::new();
    for channel_name in &config.channels_to_align {
        let channel = spec
            .channels
            .iter()
            .find(|item| item.name == *channel_name)
            .ok_or_else(|| anyhow!("Unknown channel {:?}", channel_name))?;
        let data = load_channel_array(&channel.image_path, spec.size_t, spec.size_z)?;
        let aligned = apply_shift_set_to_array(&data, shifts)?;
        let output_path = spec
            .images_dir
            .join(format!("{}{}_aligned.npz", spec.basename, channel.name));
        if output_path.exists() && !config.overwrite {
            bail!(
                "Aligned output already exists at {}. Re-run with overwrite enabled to replace it.",
                output_path.display()
            );
        }
        write_array_npz_atomic(&output_path, &aligned)?;
        aligned_outputs.push(output_path);
    }
    let shifts_array = Array2::from_shape_vec(
        (shifts.shifts_xy.len(), 2),
        shifts
            .shifts_xy
            .iter()
            .flat_map(|shift| [shift[0], shift[1]])
            .collect(),
    )?;
    write_npy_atomic(&shifts_path, &shifts_array)?;
    Ok(AlignmentRunResult {
        aligned_outputs,
        shifts_path,
    })
}

pub fn preview_crop(config: &CropSaveConfig) -> Result<CropPreview> {
    let spec = resolve_measurement_position(&config.position_dir)?;
    let rois = normalized_crop_rois(&config.crop_rois, &spec)?;
    let (frame_start, frame_end) = normalize_range(config.frame_range, spec.size_t.max(1))?;
    let (z_start, z_end) = normalize_range(config.z_range, spec.size_z.max(1))?;
    let frame_len = frame_end.saturating_sub(frame_start);
    let z_len = z_end.saturating_sub(z_start);
    let output_shapes = rois
        .iter()
        .map(|roi| match (frame_len > 1, z_len > 1) {
            (true, true) => vec![frame_len, z_len, roi.height, roi.width],
            (true, false) => vec![frame_len, roi.height, roi.width],
            (false, true) => vec![z_len, roi.height, roi.width],
            (false, false) => vec![roi.height, roi.width],
        })
        .collect();
    Ok(CropPreview { output_shapes })
}

pub fn save_cropped_data(config: CropSaveConfig) -> Result<CropSaveResult> {
    let spec = resolve_measurement_position(&config.position_dir)?;
    let rois = normalized_crop_rois(&config.crop_rois, &spec)?;
    let positions = build_crop_targets(&spec.position_dir, rois.len())?;
    let (frame_start, frame_end) = normalize_range(config.frame_range, spec.size_t.max(1))?;
    let (z_start, z_end) = normalize_range(config.z_range, spec.size_z.max(1))?;
    let mut written_files = Vec::new();
    let mut updated_metadata_paths = Vec::new();
    let metadata_values: BTreeMap<String, String> = spec
        .metadata_path
        .as_ref()
        .map(|path| read_metadata_map(path.as_path()))
        .transpose()?
        .unwrap_or_default();

    for (roi_index, target_dir) in positions.iter().enumerate() {
        let roi = &rois[roi_index.min(rois.len().saturating_sub(1))];
        let images_dir = target_dir.join("Images");
        fs::create_dir_all(&images_dir)?;
        let mut target_crop_table = CropRoiCoordsTable {
            rois: vec![CropRoiRect {
                roi_id: 0,
                x: roi.x,
                y: roi.y,
                width: roi.width,
                height: roi.height,
            }],
            cropped_roi_ids: vec![0],
        };
        if positions.len() == 1 && rois.len() == 1 {
            target_crop_table = CropRoiCoordsTable {
                rois: vec![roi.clone()],
                cropped_roi_ids: vec![roi.roi_id],
            };
        }

        for channel_name in &config.channels {
            let channel = spec
                .channels
                .iter()
                .find(|item| item.name == *channel_name)
                .ok_or_else(|| anyhow!("Unknown channel {:?}", channel_name))?;
            let data = load_channel_array(&channel.image_path, spec.size_t, spec.size_z)?;
            let cropped = crop_array(&data, roi, (frame_start, frame_end), (z_start, z_end))?;
            let file_name = if positions.len() == 1 {
                channel
                    .image_path
                    .file_name()
                    .map(|name| name.to_os_string())
                    .ok_or_else(|| {
                        anyhow!("Invalid channel path {}", channel.image_path.display())
                    })?
            } else {
                OsString::from(format!("{}{}.tif", spec.basename, channel.name))
            };
            let target_path = images_dir.join(file_name);
            write_channel_array_atomic(&target_path, &cropped, &channel.image_path)?;
            written_files.push(target_path);
        }

        let metadata_path = images_dir.join(format!("{}metadata.csv", spec.basename));
        let size_t = frame_end.saturating_sub(frame_start).max(1);
        let size_z = z_end.saturating_sub(z_start).max(1);
        let size_y = roi.height;
        let size_x = roi.width;
        write_metadata_map(
            &metadata_path,
            metadata_values.clone(),
            &spec.basename,
            size_t,
            size_z,
            size_y,
            size_x,
        )?;
        updated_metadata_paths.push(metadata_path.clone());

        let roi_coords_path = images_dir.join(format!("{}dataPrepROIs_coords.csv", spec.basename));
        save_crop_roi_coords_csv(&roi_coords_path, &target_crop_table)?;
        written_files.push(roi_coords_path);

        let adjusted_background = translate_background_rois_for_crop(&config.background_rois, roi);
        let bkgr_json_path = images_dir.join(format!("{}dataPrep_bkgrROIs.json", spec.basename));
        write_background_roi_json(&bkgr_json_path, &adjusted_background)?;
        written_files.push(bkgr_json_path);

        let adjusted_free_roi = if positions.len() > 1 {
            None
        } else {
            adjust_free_roi_for_crop(config.free_roi.as_ref(), roi)
        };
        let free_roi_path = images_dir.join(format!("{}dataPrepFreeRoi.npz", spec.basename));
        if let Some(free_roi) = adjusted_free_roi.as_ref() {
            write_freehand_roi_npz(&free_roi_path, free_roi)?;
            written_files.push(free_roi_path);
        } else {
            let _ = remove_freehand_roi_npz(&free_roi_path);
        }

        if spec.size_z > 1 {
            let segm_info_path = images_dir.join(format!("{}segmInfo.csv", spec.basename));
            let mut segm_info = if let Some(path) = spec.segm_info_path.as_ref() {
                load_segm_info(path)?
            } else {
                build_default_segm_info_table(&spec)?
            };
            if z_start > 0 {
                for record in segm_info.records.values_mut() {
                    record.crop_lower_z_slice = Some(z_start);
                    record.crop_upper_z_slice = Some(z_end.saturating_sub(1));
                    record.z_slice_used_data_prep = record
                        .z_slice_used_data_prep
                        .saturating_sub(z_start)
                        .min(size_z - 1);
                    record.z_slice_used_gui = record
                        .z_slice_used_gui
                        .saturating_sub(z_start)
                        .min(size_z - 1);
                }
            }
            save_segm_info(&segm_info_path, &segm_info)?;
            written_files.push(segm_info_path);
        }

        let archives =
            compute_background_roi_archives(target_dir, &config.channels, &adjusted_background)?;
        written_files.extend(archives);
    }

    Ok(CropSaveResult {
        written_files,
        updated_metadata_paths,
    })
}

pub fn compute_background_roi_archives(
    position_dir: impl AsRef<Path>,
    channels: &[String],
    background_rois: &BackgroundRoiSet,
) -> Result<Vec<PathBuf>> {
    let spec = resolve_measurement_position(position_dir)?;
    let mut outputs = Vec::new();
    for channel_name in channels {
        let Some(channel) = spec.channels.iter().find(|item| item.name == *channel_name) else {
            continue;
        };
        let data = load_channel_array(&channel.image_path, spec.size_t, spec.size_z)?;
        let mut archive = BackgroundRoiArchive::new();
        for (roi_index, roi) in background_rois.items.iter().enumerate() {
            let x = roi.pos[0].round().max(0.0) as usize;
            let y = roi.pos[1].round().max(0.0) as usize;
            let width = roi.size[0].round().max(0.0) as usize;
            let height = roi.size[1].round().max(0.0) as usize;
            if width == 0 || height == 0 {
                continue;
            }
            let cropped = crop_array(
                &data,
                &CropRoiRect {
                    roi_id: roi_index,
                    x,
                    y,
                    width,
                    height,
                },
                (0, spec.size_t.max(1)),
                (0, spec.size_z.max(1)),
            )?;
            archive.insert(format!("roi{roi_index}_data"), cropped);
        }
        if archive.is_empty() {
            continue;
        }
        let stem = channel
            .image_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("Invalid channel filename {}", channel.image_path.display()))?;
        let output_path = spec.images_dir.join(format!("{stem}_bkgrRoiData.npz"));
        write_background_roi_npz(&output_path, &archive)?;
        outputs.push(output_path);
    }
    Ok(outputs)
}

fn parse_pair(value: Option<&Value>) -> Option<[f32; 2]> {
    let array = value?.as_array()?;
    if array.len() < 2 {
        return None;
    }
    Some([array[0].as_f64()? as f32, array[1].as_f64()? as f32])
}

fn normalize_range(range: Option<(usize, usize)>, max_len: usize) -> Result<(usize, usize)> {
    let max_len = max_len.max(1);
    match range {
        Some((start, end)) => {
            if start >= end || end > max_len {
                bail!("Invalid crop range ({start}, {end}) for length {max_len}");
            }
            Ok((start, end))
        }
        None => Ok((0, max_len)),
    }
}

fn normalized_crop_rois(
    rois: &[CropRoiRect],
    spec: &crate::layout::MeasurementPositionSpec,
) -> Result<Vec<CropRoiRect>> {
    let mut normalized = if rois.is_empty() {
        vec![CropRoiRect {
            roi_id: 0,
            x: 0,
            y: 0,
            width: infer_spatial_shape(&spec.channels[0].image_path, spec.size_t, spec.size_z)?.1,
            height: infer_spatial_shape(&spec.channels[0].image_path, spec.size_t, spec.size_z)?.0,
        }]
    } else {
        rois.to_vec()
    };
    let (height, width) =
        infer_spatial_shape(&spec.channels[0].image_path, spec.size_t, spec.size_z)?;
    for roi in &mut normalized {
        roi.x = roi.x.min(width);
        roi.y = roi.y.min(height);
        roi.width = roi.width.min(width.saturating_sub(roi.x));
        roi.height = roi.height.min(height.saturating_sub(roi.y));
        if roi.width == 0 || roi.height == 0 {
            bail!("Crop ROI {} has zero width or height", roi.roi_id);
        }
    }
    Ok(normalized)
}

fn infer_spatial_shape(path: &Path, size_t: usize, size_z: usize) -> Result<(usize, usize)> {
    let data = load_channel_array(path, size_t, size_z)?;
    match data.ndim() {
        2 => Ok((data.shape()[0], data.shape()[1])),
        3 => Ok((data.shape()[1], data.shape()[2])),
        4 => Ok((data.shape()[2], data.shape()[3])),
        ndim => bail!("Unsupported image ndim {} in {}", ndim, path.display()),
    }
}

fn load_channel_array(path: &Path, size_t: usize, size_z: usize) -> Result<ArrayD<f32>> {
    if size_z > 1 {
        let (values, shape) = load_image_volume_as_f32(path, Some(size_t), Some(size_z))?;
        return volume_to_array(values, shape);
    }
    let (values, shape) = load_image_stack_as_f32(path)?;
    stack_to_array(values, shape)
}

fn load_channel_cube(
    path: &Path,
    size_t: usize,
    size_z: usize,
) -> Result<(Vec<Array2<f32>>, usize, usize, usize)> {
    let data = load_channel_array(path, size_t, size_z)?;
    match data.ndim() {
        2 => {
            let plane = data.into_dimensionality::<Ix2>()?;
            let shape = plane.raw_dim();
            Ok((vec![plane], 1, shape[0], shape[1]))
        }
        3 => {
            let arr = data.into_dimensionality::<Ix3>()?;
            if size_z > 1 && size_t == 1 {
                let frames = vec![max_project_3d(&arr)];
                let projected = frames[0].shape().to_vec();
                Ok((frames, 1, projected[0], projected[1]))
            } else {
                let height = arr.shape()[1];
                let width = arr.shape()[2];
                Ok((
                    arr.outer_iter().map(|frame| frame.to_owned()).collect(),
                    arr.shape()[0],
                    height,
                    width,
                ))
            }
        }
        4 => {
            let arr = data.into_dimensionality::<Ix4>()?;
            let frames = arr
                .outer_iter()
                .map(|stack| max_project_3d(&stack.to_owned()))
                .collect::<Vec<_>>();
            let projected = frames[0].shape().to_vec();
            Ok((frames, arr.shape()[0], projected[0], projected[1]))
        }
        ndim => bail!("Unsupported image ndim {} in {}", ndim, path.display()),
    }
}

fn stack_to_array(values: Vec<f32>, shape: StackShape) -> Result<ArrayD<f32>> {
    if shape.frames <= 1 {
        Ok(Array2::from_shape_vec((shape.height, shape.width), values)?.into_dyn())
    } else {
        Ok(Array3::from_shape_vec((shape.frames, shape.height, shape.width), values)?.into_dyn())
    }
}

fn volume_to_array(values: Vec<f32>, shape: VolumeShape) -> Result<ArrayD<f32>> {
    match (shape.size_t > 1, shape.size_z > 1) {
        (true, true) => Ok(Array4::from_shape_vec(
            (shape.size_t, shape.size_z, shape.height, shape.width),
            values,
        )?
        .into_dyn()),
        (true, false) => Ok(Array3::from_shape_vec(
            (shape.size_t, shape.height, shape.width),
            values,
        )?
        .into_dyn()),
        (false, true) => Ok(Array3::from_shape_vec(
            (shape.size_z, shape.height, shape.width),
            values,
        )?
        .into_dyn()),
        (false, false) => {
            Ok(Array2::from_shape_vec((shape.height, shape.width), values)?.into_dyn())
        }
    }
}

fn max_project_3d(values: &Array3<f32>) -> Array2<f32> {
    let mut projected = values.index_axis(ndarray::Axis(0), 0).to_owned();
    for z in 1..values.shape()[0] {
        let plane = values.index_axis(ndarray::Axis(0), z);
        projected.zip_mut_with(&plane, |dst, src| {
            if *src > *dst {
                *dst = *src;
            }
        });
    }
    projected
}

fn estimate_xy_shift(reference: &Array2<f32>, moving: &Array2<f32>, max_shift: i32) -> (i32, i32) {
    let height = reference.shape()[0] as i32;
    let width = reference.shape()[1] as i32;
    let mut best = (0, 0);
    let mut best_score = f32::NEG_INFINITY;
    for dy in -max_shift..=max_shift {
        for dx in -max_shift..=max_shift {
            let mut score = 0.0f32;
            let mut count = 0usize;
            for y in 0..height {
                let my = y + dy;
                if my < 0 || my >= height {
                    continue;
                }
                for x in 0..width {
                    let mx = x + dx;
                    if mx < 0 || mx >= width {
                        continue;
                    }
                    score +=
                        reference[(y as usize, x as usize)] * moving[(my as usize, mx as usize)];
                    count += 1;
                }
            }
            if count > 0 {
                let normalized = score / count as f32;
                if normalized > best_score {
                    best_score = normalized;
                    best = (dy, dx);
                }
            }
        }
    }
    best
}

fn apply_shift_set_to_array(
    values: &ArrayD<f32>,
    shifts: &AlignmentShiftSet,
) -> Result<ArrayD<f32>> {
    match values.ndim() {
        2 => Ok(values.clone()),
        3 => {
            let arr = values.view().into_dimensionality::<Ix3>()?;
            let mut out = arr.to_owned();
            if shifts.shifts_xy.len() == arr.shape()[0] {
                for (frame_i, plane) in arr.outer_iter().enumerate() {
                    let [dx, dy] = shifts.shifts_xy[frame_i];
                    out.slice_mut(s![frame_i, .., ..]).assign(&shift_plane(
                        &plane.to_owned(),
                        dx,
                        dy,
                    ));
                }
            }
            Ok(out.into_dyn())
        }
        4 => {
            let arr = values.view().into_dimensionality::<Ix4>()?;
            let mut out = arr.to_owned();
            for (frame_i, stack) in arr.outer_iter().enumerate() {
                let [dx, dy] = shifts.shifts_xy.get(frame_i).copied().unwrap_or([0, 0]);
                for z in 0..stack.shape()[0] {
                    out.slice_mut(s![frame_i, z, .., ..]).assign(&shift_plane(
                        &stack.index_axis(ndarray::Axis(0), z).to_owned(),
                        dx,
                        dy,
                    ));
                }
            }
            Ok(out.into_dyn())
        }
        ndim => bail!("Unsupported image ndim {} for alignment", ndim),
    }
}

fn shift_plane(plane: &Array2<f32>, dx: i32, dy: i32) -> Array2<f32> {
    let height = plane.shape()[0] as i32;
    let width = plane.shape()[1] as i32;
    let mut shifted = Array2::<f32>::zeros((plane.shape()[0], plane.shape()[1]));
    for y in 0..height {
        for x in 0..width {
            let src_x = x + dx;
            let src_y = y + dy;
            if src_x >= 0 && src_x < width && src_y >= 0 && src_y < height {
                shifted[(y as usize, x as usize)] = plane[(src_y as usize, src_x as usize)];
            }
        }
    }
    shifted
}

fn crop_array(
    values: &ArrayD<f32>,
    roi: &CropRoiRect,
    frame_range: (usize, usize),
    z_range: (usize, usize),
) -> Result<ArrayD<f32>> {
    Ok(match values.ndim() {
        2 => values
            .slice(s![roi.y..roi.y + roi.height, roi.x..roi.x + roi.width])
            .to_owned()
            .into_dyn(),
        3 => {
            let arr = values.view().into_dimensionality::<Ix3>()?;
            if frame_range.1 - frame_range.0 == 1 && z_range.1 - z_range.0 > 1 {
                arr.slice(s![
                    z_range.0..z_range.1,
                    roi.y..roi.y + roi.height,
                    roi.x..roi.x + roi.width
                ])
                .to_owned()
                .into_dyn()
            } else {
                arr.slice(s![
                    frame_range.0..frame_range.1,
                    roi.y..roi.y + roi.height,
                    roi.x..roi.x + roi.width
                ])
                .to_owned()
                .into_dyn()
            }
        }
        4 => {
            let arr = values.view().into_dimensionality::<Ix4>()?;
            arr.slice(s![
                frame_range.0..frame_range.1,
                z_range.0..z_range.1,
                roi.y..roi.y + roi.height,
                roi.x..roi.x + roi.width
            ])
            .to_owned()
            .into_dyn()
        }
        ndim => bail!("Unsupported image ndim {} for crop", ndim),
    })
}

fn build_crop_targets(position_dir: &Path, roi_count: usize) -> Result<Vec<PathBuf>> {
    if roi_count <= 1 {
        return Ok(vec![position_dir.to_path_buf()]);
    }
    let parent = position_dir.parent().ok_or_else(|| {
        anyhow!(
            "Position directory has no parent: {}",
            position_dir.display()
        )
    })?;
    let mut max_index = 0usize;
    for entry in fs::read_dir(parent)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if let Some(suffix) = name.strip_prefix("Position_") {
            if let Ok(index) = suffix.parse::<usize>() {
                max_index = max_index.max(index);
            }
        }
    }
    Ok((0..roi_count)
        .map(|offset| parent.join(format!("Position_{}", max_index + offset + 1)))
        .collect())
}

fn translate_background_rois_for_crop(
    rois: &BackgroundRoiSet,
    crop: &CropRoiRect,
) -> BackgroundRoiSet {
    let items = rois
        .items
        .iter()
        .filter_map(|roi| {
            let x0 = roi.pos[0].round().max(0.0) as isize - crop.x as isize;
            let y0 = roi.pos[1].round().max(0.0) as isize - crop.y as isize;
            let width = roi.size[0].round().max(0.0) as isize;
            let height = roi.size[1].round().max(0.0) as isize;
            let x1 = (x0 + width).clamp(0, crop.width as isize);
            let y1 = (y0 + height).clamp(0, crop.height as isize);
            let x0 = x0.clamp(0, crop.width as isize);
            let y0 = y0.clamp(0, crop.height as isize);
            (x1 > x0 && y1 > y0).then_some(BackgroundRoiRect {
                pos: [x0 as f32, y0 as f32],
                size: [(x1 - x0) as f32, (y1 - y0) as f32],
            })
        })
        .collect();
    BackgroundRoiSet { items }
}

fn adjust_free_roi_for_crop(
    free_roi: Option<&FreehandRoiMask>,
    crop: &CropRoiRect,
) -> Option<FreehandRoiMask> {
    let free_roi = free_roi?;
    let (y0, x0, y1, x1) = free_roi.bbox_yxxy;
    let new_x0 = x0.max(crop.x).saturating_sub(crop.x);
    let new_y0 = y0.max(crop.y).saturating_sub(crop.y);
    let crop_x1 = crop.x + crop.width.saturating_sub(1);
    let crop_y1 = crop.y + crop.height.saturating_sub(1);
    let new_x1 = x1.min(crop_x1).saturating_sub(crop.x);
    let new_y1 = y1.min(crop_y1).saturating_sub(crop.y);
    if new_x1 < new_x0 || new_y1 < new_y0 {
        return None;
    }
    let local_x0 = new_x0 + crop.x - x0;
    let local_y0 = new_y0 + crop.y - y0;
    let local_x1 = new_x1 + crop.x - x0;
    let local_y1 = new_y1 + crop.y - y0;
    let local_mask = free_roi
        .local_mask
        .slice(s![local_y0..=local_y1, local_x0..=local_x1])
        .to_owned();
    Some(FreehandRoiMask {
        bbox_yxxy: (new_y0, new_x0, new_y1, new_x1),
        local_mask,
    })
}

fn write_metadata_map(
    path: &Path,
    mut values: BTreeMap<String, String>,
    basename: &str,
    size_t: usize,
    size_z: usize,
    size_y: usize,
    size_x: usize,
) -> Result<()> {
    values.insert("basename".into(), basename.to_string());
    values.insert("SizeT".into(), size_t.to_string());
    values.insert("SizeZ".into(), size_z.to_string());
    values.insert("SizeY".into(), size_y.to_string());
    values.insert("SizeX".into(), size_x.to_string());
    let tmp_path = temp_write_path(path);
    let mut writer = Writer::from_path(&tmp_path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record(["Description", "values"])?;
    let mut ordered = vec![
        "basename".to_string(),
        "SizeT".to_string(),
        "SizeZ".to_string(),
        "SizeY".to_string(),
        "SizeX".to_string(),
        "TimeIncrement".to_string(),
        "PhysicalSizeZ".to_string(),
        "PhysicalSizeY".to_string(),
        "PhysicalSizeX".to_string(),
        "channel_0_name".to_string(),
        "channel_1_name".to_string(),
    ];
    let extra_segm_keys = values
        .keys()
        .filter(|key| key.ends_with("_isSegm3D"))
        .cloned()
        .collect::<Vec<_>>();
    ordered.extend(extra_segm_keys);
    for key in ordered {
        if let Some(value) = values.remove(&key) {
            writer.write_record([key.as_str(), value.as_str()])?;
        }
    }
    for (key, value) in values {
        writer.write_record([key.as_str(), value.as_str()])?;
    }
    writer.flush()?;
    promote_temp_file(&tmp_path, path)
}

fn write_channel_array_atomic(
    target_path: &Path,
    values: &ArrayD<f32>,
    source_path: &Path,
) -> Result<()> {
    match target_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("npz") => write_array_npz_atomic(target_path, values),
        Some("tif") | Some("tiff") => {
            let scalar_type = detect_image_scalar_type(source_path)?;
            write_array_tiff_atomic(target_path, values, scalar_type)
        }
        Some("h5") => bail!(
            "Writing H5 channel outputs is not supported yet for {}. Use TIFF/NPZ-backed sessions for Data Prep saves.",
            target_path.display()
        ),
        other => bail!(
            "Unsupported output format {:?} for {}",
            other,
            target_path.display()
        ),
    }
}

fn detect_image_scalar_type(path: &Path) -> Result<ImageScalarType> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("tif") | Some("tiff") => {
            let file = File::open(path)
                .with_context(|| format!("Failed to open TIFF {}", path.display()))?;
            let mut decoder = tiff::decoder::Decoder::new(file)
                .with_context(|| format!("Failed to decode TIFF {}", path.display()))?;
            let result = decoder
                .read_image()
                .with_context(|| format!("Failed to inspect TIFF {}", path.display()))?;
            Ok(match result {
                tiff::decoder::DecodingResult::U8(_) => ImageScalarType::U8,
                tiff::decoder::DecodingResult::U16(_) => ImageScalarType::U16,
                tiff::decoder::DecodingResult::U32(_) => ImageScalarType::U32,
                _ => ImageScalarType::F32,
            })
        }
        _ => Ok(ImageScalarType::F32),
    }
}

fn write_array_npz_atomic(path: &Path, values: &ArrayD<f32>) -> Result<()> {
    let tmp_path = temp_write_path(path);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let file = File::create(&tmp_path)?;
    let mut writer = NpzWriter::new_compressed(file);
    writer.add_array("arr_0", values)?;
    writer.finish()?;
    promote_temp_file(&tmp_path, path)
}

fn write_array_tiff_atomic(
    path: &Path,
    values: &ArrayD<f32>,
    scalar_type: ImageScalarType,
) -> Result<()> {
    let tmp_path = temp_write_path(path);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let file = File::create(&tmp_path)?;
    let mut encoder = TiffEncoder::new(file)?;
    for plane in flatten_array_planes(values)? {
        match scalar_type {
            ImageScalarType::U8 => encoder.write_image::<colortype::Gray8>(
                plane.width as u32,
                plane.height as u32,
                &convert_plane_to_u8(&plane.pixels),
            )?,
            ImageScalarType::U16 => encoder.write_image::<colortype::Gray16>(
                plane.width as u32,
                plane.height as u32,
                &convert_plane_to_u16(&plane.pixels),
            )?,
            ImageScalarType::U32 => encoder.write_image::<colortype::Gray32>(
                plane.width as u32,
                plane.height as u32,
                &convert_plane_to_u32(&plane.pixels),
            )?,
            ImageScalarType::F32 => encoder.write_image::<colortype::Gray32Float>(
                plane.width as u32,
                plane.height as u32,
                &plane.pixels,
            )?,
        }
    }
    promote_temp_file(&tmp_path, path)
}

fn write_npy_atomic<T: ndarray_npy::WritableElement>(
    path: &Path,
    values: &Array2<T>,
) -> Result<()> {
    let tmp_path = temp_write_path(path);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    write_npy(&tmp_path, values)?;
    promote_temp_file(&tmp_path, path)
}

fn temp_write_path(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| OsString::from("temp"));
    file_name.push(".tmp");
    path.with_file_name(file_name)
}

fn promote_temp_file(temp_path: &Path, final_path: &Path) -> Result<()> {
    if final_path.exists() {
        fs::remove_file(final_path)
            .with_context(|| format!("Failed to replace {}", final_path.display()))?;
    }
    fs::rename(temp_path, final_path).with_context(|| {
        format!(
            "Failed to promote {} to {}",
            temp_path.display(),
            final_path.display()
        )
    })
}

#[derive(Debug)]
struct PlaneF32 {
    height: usize,
    width: usize,
    pixels: Vec<f32>,
}

fn flatten_array_planes(values: &ArrayD<f32>) -> Result<Vec<PlaneF32>> {
    match values.ndim() {
        2 => {
            let array = values.view().into_dimensionality::<Ix2>()?;
            Ok(vec![PlaneF32 {
                height: array.shape()[0],
                width: array.shape()[1],
                pixels: array.iter().copied().collect(),
            }])
        }
        3 => {
            let array = values.view().into_dimensionality::<Ix3>()?;
            Ok(array
                .outer_iter()
                .map(|plane| PlaneF32 {
                    height: plane.shape()[0],
                    width: plane.shape()[1],
                    pixels: plane.iter().copied().collect(),
                })
                .collect())
        }
        4 => {
            let array = values.view().into_dimensionality::<Ix4>()?;
            let mut planes = Vec::new();
            for stack in array.outer_iter() {
                for plane in stack.outer_iter() {
                    planes.push(PlaneF32 {
                        height: plane.shape()[0],
                        width: plane.shape()[1],
                        pixels: plane.iter().copied().collect(),
                    });
                }
            }
            Ok(planes)
        }
        ndim => bail!("Unsupported image ndim {} for TIFF output", ndim),
    }
}

fn convert_plane_to_integer(values: &[f32], max_value: f32) -> Vec<u32> {
    values
        .iter()
        .map(|value| value.round().clamp(0.0, max_value) as u32)
        .collect()
}

fn convert_plane_to_u8(values: &[f32]) -> Vec<u8> {
    convert_plane_to_integer(values, u8::MAX as f32)
        .into_iter()
        .map(|value| value as u8)
        .collect()
}

fn convert_plane_to_u16(values: &[f32]) -> Vec<u16> {
    convert_plane_to_integer(values, u16::MAX as f32)
        .into_iter()
        .map(|value| value as u16)
        .collect()
}

fn convert_plane_to_u32(values: &[f32]) -> Vec<u32> {
    convert_plane_to_integer(values, u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::resolve_measurement_position;
    use ndarray::Array;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

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
            "roi0_data".to_string(),
            Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn(),
        );
        write_background_roi_npz(&path, &archive)?;
        let loaded = read_background_roi_npz(&path)?;
        assert_eq!(loaded["roi0_data"].shape(), &[3]);
        Ok(())
    }

    #[test]
    fn roundtrips_crop_roi_coords_csv() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo_dataPrepROIs_coords.csv");
        let table = CropRoiCoordsTable {
            rois: vec![CropRoiRect {
                roi_id: 3,
                x: 5,
                y: 7,
                width: 11,
                height: 13,
            }],
            cropped_roi_ids: vec![3],
        };
        save_crop_roi_coords_csv(&path, &table)?;
        let loaded = load_crop_roi_coords_csv(&path)?;
        assert_eq!(loaded, table);
        Ok(())
    }

    #[test]
    fn roundtrips_free_roi_npz() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo_dataPrepFreeRoi.npz");
        let roi = FreehandRoiMask {
            bbox_yxxy: (2, 3, 5, 7),
            local_mask: Array2::from_shape_vec((4, 5), vec![true; 20])?,
        };
        write_freehand_roi_npz(&path, &roi)?;
        let loaded = read_freehand_roi_npz(&path)?.unwrap();
        assert_eq!(loaded.bbox_yxxy, roi.bbox_yxxy);
        assert_eq!(loaded.local_mask, roi.local_mask);
        Ok(())
    }

    #[test]
    fn computes_background_roi_archive_for_timelapse_2d() -> Result<()> {
        let dir = tempdir()?;
        let images = dir.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(
            &images.join("demo_phase.tif"),
            &[vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10, 11, 12]],
            2,
            3,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        let paths = compute_background_roi_archives(
            dir.path().join("Position_1"),
            &[String::from("phase")],
            &BackgroundRoiSet {
                items: vec![BackgroundRoiRect {
                    pos: [1.0, 0.0],
                    size: [2.0, 2.0],
                }],
            },
        )?;
        assert_eq!(paths.len(), 1);
        let archive = read_background_roi_npz(&paths[0])?;
        assert_eq!(archive["roi0_data"].shape(), &[2, 2, 2]);
        Ok(())
    }

    #[test]
    fn saves_aligned_outputs_and_shift_file() -> Result<()> {
        let dir = tempdir()?;
        let images = dir.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(
            &images.join("demo_phase.tif"),
            &[
                vec![0, 0, 1, 1, 0, 0, 0, 0, 0],
                vec![0, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
            3,
            3,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        let shifts = compute_alignment_shifts(&AlignmentRunConfig {
            position_dir: dir.path().join("Position_1"),
            reference_channel: "phase".to_string(),
            channels_to_align: vec!["phase".to_string()],
            frame_range: None,
            overwrite: true,
        })?;
        let result = apply_alignment(
            AlignmentRunConfig {
                position_dir: dir.path().join("Position_1"),
                reference_channel: "phase".to_string(),
                channels_to_align: vec!["phase".to_string()],
                frame_range: None,
                overwrite: true,
            },
            &shifts,
        )?;
        assert_eq!(result.aligned_outputs.len(), 1);
        assert!(result.shifts_path.exists());
        Ok(())
    }

    #[test]
    fn crop_save_single_roi_overwrites_position_with_metadata() -> Result<()> {
        let dir = tempdir()?;
        let position = dir.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_tiff(&images.join("demo_phase.tif"), &[vec![1, 2, 3, 4]], 2, 2)?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\nchannel_0_name,phase\n",
        )?;
        let result = save_cropped_data(CropSaveConfig {
            position_dir: position.clone(),
            channels: vec!["phase".to_string()],
            frame_range: None,
            z_range: None,
            crop_rois: vec![CropRoiRect {
                roi_id: 0,
                x: 0,
                y: 0,
                width: 1,
                height: 2,
            }],
            background_rois: BackgroundRoiSet::default(),
            free_roi: None,
            overwrite: true,
        })?;
        assert!(!result.written_files.is_empty());
        let spec = resolve_measurement_position(&position)?;
        assert_eq!(
            spec.channels[0]
                .image_path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("demo_phase.tif")
        );
        Ok(())
    }

    fn write_test_tiff(
        path: &Path,
        planes: &[Vec<u16>],
        height: usize,
        width: usize,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for plane in planes {
            encoder.write_image::<colortype::Gray16>(width as u32, height as u32, plane)?;
        }
        Ok(())
    }
}
