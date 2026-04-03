use crate::image_io::{
    load_image_stack_as_f32, load_mask_stack_as_u32, load_npz_archive_arrays_as_f32, NamedArrayF32,
    StackShape,
};
use crate::layout::{
    discover_measurement_experiment, resolve_measurement_position, ChannelSpec,
    MeasurementExperimentSpec, MeasurementPositionSpec, PositionSpec,
};
use crate::runner::OverwritePolicy;
use anyhow::{bail, Context, Result};
use csv::Writer;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementRunConfig {
    pub position_path: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementExperimentConfig {
    pub experiment_dir: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementOutputPaths {
    pub segm_npz_path: PathBuf,
    pub acdc_output_csv_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementRunResult {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub outputs: MeasurementOutputPaths,
    pub labels_found: u32,
    pub frames_processed: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedMeasurementPosition {
    pub spec: MeasurementPositionSpec,
    pub outputs: MeasurementOutputPaths,
    pub mask_values: Vec<u32>,
    pub mask_shape: StackShape,
}

pub(crate) fn measurement_position_from_position(position: &PositionSpec) -> MeasurementPositionSpec {
    MeasurementPositionSpec {
        position_dir: position.position_dir.clone(),
        images_dir: position.images_dir.clone(),
        basename: position.basename.clone(),
        channels: position.channels.clone(),
        metadata_path: position.metadata_path.clone(),
        data_prep_background_rois_path: position.data_prep_background_rois_path.clone(),
        size_t: position.size_t,
        time_increment: position.time_increment,
        physical_size_x: position.physical_size_x,
        physical_size_y: position.physical_size_y,
    }
}

#[derive(Debug, Clone)]
struct LoadedChannelData {
    spec: ChannelSpec,
    values: Vec<f32>,
    background_arrays: Vec<NamedArrayF32>,
}

#[derive(Debug, Clone)]
struct MeasurementRow {
    frame_i: usize,
    time_seconds: f64,
    time_minutes: f64,
    time_hours: f64,
    cell_id: u32,
    is_cell_dead: u8,
    is_cell_excluded: u8,
    was_manually_edited: u8,
    x_centroid: i32,
    y_centroid: i32,
    cell_area_pxl: usize,
    cell_area_um2: f64,
    cell_vol_vox: f64,
    cell_vol_fl: f64,
    velocity_pixel: f64,
    velocity_um: f64,
    disappears_before_end: u8,
    dynamic_values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
struct FrameRegion {
    label: u32,
    area: usize,
    pixels: Vec<(usize, usize)>,
    bbox: (usize, usize, usize, usize),
    centroid_y: f64,
    centroid_x: f64,
}

#[derive(Debug, Clone)]
struct RegionMeasurements {
    major_axis_length: f64,
    minor_axis_length: f64,
    eccentricity: f64,
    aspect_ratio: f64,
    circularity: f64,
    roundness: f64,
    equivalent_diameter: f64,
    area: f64,
    solidity: f64,
    extent: f64,
    feret_diameter_max: f64,
    filled_area: f64,
    convex_area: f64,
    euler_number: f64,
    bbox_area: f64,
    centroid_y: f64,
    centroid_x: f64,
    local_centroid_y: f64,
    local_centroid_x: f64,
    bbox_min_y: f64,
    bbox_min_x: f64,
    bbox_max_y: f64,
    bbox_max_x: f64,
    inertia_eig0: f64,
    inertia_eig1: f64,
    orientation: f64,
}

const CHANNEL_METRIC_SUFFIXES: &[&str] = &[
    "mean",
    "sum",
    "amount_autoBkgr",
    "amount_dataPrepBkgr",
    "concentration_autoBkgr_from_vol_vox",
    "concentration_dataPrepBkgr_from_vol_vox",
    "concentration_autoBkgr_from_vol_fl",
    "concentration_dataPrepBkgr_from_vol_fl",
    "median",
    "min",
    "max",
    "q25",
    "q75",
    "q05",
    "q95",
    "autoBkgr_bkgrVal_median",
    "autoBkgr_bkgrVal_mean",
    "autoBkgr_bkgrVal_q75",
    "autoBkgr_bkgrVal_q25",
    "autoBkgr_bkgrVal_q95",
    "autoBkgr_bkgrVal_q05",
    "dataPrepBkgr_bkgrVal_median",
    "dataPrepBkgr_bkgrVal_mean",
    "dataPrepBkgr_bkgrVal_q75",
    "dataPrepBkgr_bkgrVal_q25",
    "dataPrepBkgr_bkgrVal_q95",
    "dataPrepBkgr_bkgrVal_q05",
    "CV",
];

const REGIONPROP_HEADERS: &[&str] = &[
    "inertia_tensor_eigvals-0",
    "inertia_tensor_eigvals-1",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "circularity",
    "roundness",
    "aspect_ratio",
    "equivalent_diameter",
    "area",
    "solidity",
    "feret_diameter_max",
    "extent",
    "filled_area",
    "convex_area",
    "euler_number",
    "bbox_area",
    "centroid-0",
    "centroid-1",
    "local_centroid-0",
    "local_centroid-1",
    "bbox-0",
    "bbox-1",
    "bbox-2",
    "bbox-3",
];

pub fn measure_position(config: MeasurementRunConfig) -> Result<MeasurementRunResult> {
    let spec = resolve_measurement_position(&config.position_path)?;
    let loaded = load_measurement_inputs(spec, config.segm_endname.as_deref())?;
    if config.overwrite_policy == OverwritePolicy::Refuse
        && loaded.outputs.acdc_output_csv_path.exists()
    {
        bail!(
            "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
            loaded.outputs.acdc_output_csv_path.display()
        );
    }
    write_measurements(&loaded)
}

pub fn measure_experiment(config: MeasurementExperimentConfig) -> Result<Vec<MeasurementRunResult>> {
    let experiment = discover_measurement_experiment(&config.experiment_dir)?;
    measure_experiment_from_spec(config, experiment)
}

pub(crate) fn measure_experiment_from_spec(
    config: MeasurementExperimentConfig,
    experiment: MeasurementExperimentSpec,
) -> Result<Vec<MeasurementRunResult>> {
    let mut results = Vec::with_capacity(experiment.positions.len());
    for position in experiment.positions {
        let loaded = load_measurement_inputs(position, config.segm_endname.as_deref())?;
        if config.overwrite_policy == OverwritePolicy::Refuse
            && loaded.outputs.acdc_output_csv_path.exists()
        {
            bail!(
                "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
                loaded.outputs.acdc_output_csv_path.display()
            );
        }
        results.push(write_measurements(&loaded)?);
    }
    Ok(results)
}

pub(crate) fn load_measurement_inputs(
    spec: MeasurementPositionSpec,
    segm_endname: Option<&str>,
) -> Result<LoadedMeasurementPosition> {
    let outputs = measurement_output_paths(&spec.images_dir, &spec.basename, segm_endname);
    let (mask_values, mask_shape) = load_mask_stack_as_u32(&outputs.segm_npz_path).with_context(|| {
        format!(
            "Failed to load segmentation masks from {}",
            outputs.segm_npz_path.display()
        )
    })?;

    if mask_shape.frames != spec.size_t {
        bail!(
            "Segmentation frame count ({}) does not match metadata SizeT ({}) in {}",
            mask_shape.frames,
            spec.size_t,
            outputs.segm_npz_path.display()
        );
    }

    Ok(LoadedMeasurementPosition {
        spec,
        outputs,
        mask_values,
        mask_shape,
    })
}

pub(crate) fn write_measurements(loaded: &LoadedMeasurementPosition) -> Result<MeasurementRunResult> {
    let channels = load_channels(&loaded.spec, loaded.mask_shape)?;
    let roi_mask = load_data_prep_roi_mask(
        loaded
            .spec
            .data_prep_background_rois_path
            .as_deref(),
        loaded.mask_shape.height,
        loaded.mask_shape.width,
    )?;
    let frame_len = loaded.mask_shape.height * loaded.mask_shape.width;
    let pixel_area_um2 = loaded.spec.physical_size_x * loaded.spec.physical_size_y;
    let mut rows = Vec::new();
    let mut row_indices = HashMap::<(usize, u32), usize>::new();
    let mut previous_centroids = HashMap::<u32, (f64, f64)>::new();

    for frame_i in 0..loaded.mask_shape.frames {
        let mask_frame = &loaded.mask_values[frame_i * frame_len..(frame_i + 1) * frame_len];
        let regions = extract_regions(mask_frame, loaded.mask_shape.height, loaded.mask_shape.width);
        let current_centroids = regions
            .iter()
            .map(|region| (region.label, (region.centroid_x, region.centroid_y)))
            .collect::<HashMap<_, _>>();

        let mut auto_background_masks = HashMap::<String, Vec<bool>>::new();
        let background_mask = build_background_mask(mask_frame);
        for channel in &channels {
            auto_background_masks.insert(channel.spec.name.clone(), background_mask.clone());
        }

        for region in &regions {
            let region_measurements = compute_region_measurements(region);
            let (cell_vol_vox, cell_vol_fl) = rotational_volume(
                region,
                region_measurements.orientation,
                loaded.spec.physical_size_y,
                loaded.spec.physical_size_x,
            );
            let mut dynamic_values = BTreeMap::new();
            insert_regionprop_values(&mut dynamic_values, &region_measurements);

            for channel in &channels {
                let channel_frame =
                    &channel.values[frame_i * frame_len..(frame_i + 1) * frame_len];
                let object_values = collect_object_values(channel_frame, &region.pixels, loaded.mask_shape.width);
                let auto_background = collect_masked_values(
                    channel_frame,
                    auto_background_masks
                        .get(&channel.spec.name)
                        .expect("missing auto background mask"),
                );
                let data_prep_background = collect_data_prep_background(
                    channel,
                    frame_i,
                    roi_mask.as_deref(),
                    channel_frame,
                    mask_frame,
                )?;
                insert_channel_measurements(
                    &mut dynamic_values,
                    &channel.spec.name,
                    &object_values,
                    &auto_background,
                    data_prep_background.as_deref(),
                    region.area,
                    cell_vol_vox,
                    cell_vol_fl,
                );
            }

            let previous_centroid = previous_centroids.get(&region.label).copied();
            let velocity_pixel = previous_centroid
                .map(|(prev_x, prev_y)| distance(prev_x, prev_y, region.centroid_x, region.centroid_y))
                .unwrap_or(f64::NAN);
            let velocity_um = previous_centroid
                .map(|(prev_x, prev_y)| {
                    let dx = (region.centroid_x - prev_x) * loaded.spec.physical_size_x;
                    let dy = (region.centroid_y - prev_y) * loaded.spec.physical_size_y;
                    (dx * dx + dy * dy).sqrt()
                })
                .unwrap_or(f64::NAN);

            let row = MeasurementRow {
                frame_i,
                time_seconds: loaded.spec.time_increment * frame_i as f64,
                time_minutes: loaded.spec.time_increment * frame_i as f64 / 60.0,
                time_hours: loaded.spec.time_increment * frame_i as f64 / 3600.0,
                cell_id: region.label,
                is_cell_dead: 0,
                is_cell_excluded: 0,
                was_manually_edited: 0,
                x_centroid: region.centroid_x.floor() as i32,
                y_centroid: region.centroid_y.floor() as i32,
                cell_area_pxl: region.area,
                cell_area_um2: region.area as f64 * pixel_area_um2,
                cell_vol_vox,
                cell_vol_fl,
                velocity_pixel,
                velocity_um,
                disappears_before_end: 0,
                dynamic_values,
            };
            row_indices.insert((frame_i, region.label), rows.len());
            rows.push(row);
        }

        previous_centroids = current_centroids;
    }

    mark_disappearances(
        &loaded.mask_values,
        loaded.mask_shape,
        &row_indices,
        &mut rows,
    );

    let headers = build_headers(&loaded.spec.channels);
    write_measurement_csv(&loaded.outputs.acdc_output_csv_path, &headers, &rows)?;

    Ok(MeasurementRunResult {
        position_dir: loaded.spec.position_dir.clone(),
        images_dir: loaded.spec.images_dir.clone(),
        outputs: loaded.outputs.clone(),
        labels_found: loaded.mask_values.iter().copied().max().unwrap_or(0),
        frames_processed: loaded.mask_shape.frames,
    })
}

fn measurement_output_paths(
    images_dir: &Path,
    basename: &str,
    segm_endname: Option<&str>,
) -> MeasurementOutputPaths {
    let suffix = match segm_endname {
        Some(value) if !value.trim().is_empty() => format!("_{value}"),
        _ => String::new(),
    };
    MeasurementOutputPaths {
        segm_npz_path: images_dir.join(format!("{basename}segm{suffix}.npz")),
        acdc_output_csv_path: images_dir.join(format!("{basename}acdc_output{suffix}.csv")),
    }
}

fn load_channels(spec: &MeasurementPositionSpec, mask_shape: StackShape) -> Result<Vec<LoadedChannelData>> {
    let mut channels = Vec::with_capacity(spec.channels.len());
    for channel in &spec.channels {
        let (values, shape) = load_image_stack_as_f32(&channel.image_path).with_context(|| {
            format!("Failed to load channel image {}", channel.image_path.display())
        })?;
        if shape != mask_shape {
            bail!(
                "Image stack {} has shape {}x{}x{}, but segmentation has {}x{}x{}",
                channel.image_path.display(),
                shape.frames,
                shape.height,
                shape.width,
                mask_shape.frames,
                mask_shape.height,
                mask_shape.width
            );
        }
        let background_arrays = match &channel.background_data_path {
            Some(path) => load_npz_archive_arrays_as_f32(path)
                .with_context(|| format!("Failed to load background data {}", path.display()))?,
            None => Vec::new(),
        };
        channels.push(LoadedChannelData {
            spec: channel.clone(),
            values,
            background_arrays,
        });
    }
    Ok(channels)
}

fn load_data_prep_roi_mask(
    path: Option<&Path>,
    height: usize,
    width: usize,
) -> Result<Option<Vec<bool>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let text = fs::read_to_string(path)
        .with_context(|| format!("Failed to read data-prep ROI file {}", path.display()))?;
    let json: Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse ROI JSON {}", path.display()))?;
    let mut mask = vec![false; height * width];
    let mut any = false;

    let Some(items) = json.as_array() else {
        return Ok(None);
    };
    for item in items {
        let Some((x, y)) = parse_pair(item.get("pos").or_else(|| item.get("state").and_then(|v| v.get("pos")))) else {
            continue;
        };
        let Some((w, h)) = parse_pair(item.get("size").or_else(|| item.get("state").and_then(|v| v.get("size")))) else {
            continue;
        };
        let x0 = x.round().max(0.0) as usize;
        let y0 = y.round().max(0.0) as usize;
        let x1 = (x + w).round().max(0.0) as usize;
        let y1 = (y + h).round().max(0.0) as usize;
        let x_end = x1.min(width);
        let y_end = y1.min(height);
        for yy in y0.min(height)..y_end {
            for xx in x0.min(width)..x_end {
                mask[yy * width + xx] = true;
                any = true;
            }
        }
    }

    if any {
        Ok(Some(mask))
    } else {
        Ok(None)
    }
}

fn parse_pair(value: Option<&Value>) -> Option<(f64, f64)> {
    let array = value?.as_array()?;
    if array.len() < 2 {
        return None;
    }
    Some((array[0].as_f64()?, array[1].as_f64()?))
}

fn build_background_mask(mask_frame: &[u32]) -> Vec<bool> {
    mask_frame.iter().map(|label| *label == 0).collect()
}

fn collect_object_values(
    channel_frame: &[f32],
    pixels: &[(usize, usize)],
    width: usize,
) -> Vec<f32> {
    pixels
        .iter()
        .map(|(y, x)| channel_frame[y * width + x])
        .collect()
}

fn collect_masked_values(values: &[f32], mask: &[bool]) -> Vec<f32> {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, keep)| keep.then_some(*value))
        .collect()
}

fn collect_data_prep_background(
    channel: &LoadedChannelData,
    frame_i: usize,
    roi_mask: Option<&[bool]>,
    channel_frame: &[f32],
    mask_frame: &[u32],
) -> Result<Option<Vec<f32>>> {
    if !channel.background_arrays.is_empty() {
        let mut values = Vec::new();
        for array in &channel.background_arrays {
            values.extend(frame_slice(&array.values, array.shape, frame_i)?);
        }
        return Ok(Some(values));
    }

    let Some(roi_mask) = roi_mask else {
        return Ok(None);
    };

    let mut values = Vec::new();
    for (idx, value) in channel_frame.iter().enumerate() {
        if roi_mask[idx] && mask_frame[idx] == 0 {
            values.push(*value);
        }
    }
    if values.is_empty() {
        return Ok(None);
    }
    Ok(Some(values))
}

fn frame_slice<'a>(values: &'a [f32], shape: StackShape, frame_i: usize) -> Result<&'a [f32]> {
    let frame_len = shape.height * shape.width;
    match shape.frames {
        1 => Ok(values),
        frames if frame_i < frames => {
            let start = frame_i * frame_len;
            Ok(&values[start..start + frame_len])
        }
        _ => bail!(
            "Background archive frame index {} exceeds available frames {}",
            frame_i,
            shape.frames
        ),
    }
}

fn insert_channel_measurements(
    out: &mut BTreeMap<String, f64>,
    channel_name: &str,
    object_values: &[f32],
    auto_background: &[f32],
    data_prep_background: Option<&[f32]>,
    area: usize,
    cell_vol_vox: f64,
    cell_vol_fl: f64,
) {
    let prefix = format!("{channel_name}_");
    let mean = mean_f32(object_values);
    let sum = sum_f32(object_values);
    let median = quantile_f32(object_values, 0.5);
    let min = min_f32(object_values);
    let max = max_f32(object_values);
    let q25 = quantile_f32(object_values, 0.25);
    let q75 = quantile_f32(object_values, 0.75);
    let q05 = quantile_f32(object_values, 0.05);
    let q95 = quantile_f32(object_values, 0.95);
    let cv = coefficient_of_variation(object_values);

    insert_value(out, &format!("{prefix}mean"), mean);
    insert_value(out, &format!("{prefix}sum"), sum);
    insert_value(out, &format!("{prefix}median"), median);
    insert_value(out, &format!("{prefix}min"), min);
    insert_value(out, &format!("{prefix}max"), max);
    insert_value(out, &format!("{prefix}q25"), q25);
    insert_value(out, &format!("{prefix}q75"), q75);
    insert_value(out, &format!("{prefix}q05"), q05);
    insert_value(out, &format!("{prefix}q95"), q95);
    insert_value(out, &format!("{prefix}CV"), cv);

    let auto_median = quantile_f32(auto_background, 0.5);
    let auto_mean = mean_f32(auto_background);
    let auto_q75 = quantile_f32(auto_background, 0.75);
    let auto_q25 = quantile_f32(auto_background, 0.25);
    let auto_q95 = quantile_f32(auto_background, 0.95);
    let auto_q05 = quantile_f32(auto_background, 0.05);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_median"), auto_median);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_mean"), auto_mean);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_q75"), auto_q75);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_q25"), auto_q25);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_q95"), auto_q95);
    insert_value(out, &format!("{prefix}autoBkgr_bkgrVal_q05"), auto_q05);
    let auto_amount = if auto_median.is_nan() || mean.is_nan() {
        f64::NAN
    } else {
        (mean - auto_median) * area as f64
    };
    insert_value(out, &format!("{prefix}amount_autoBkgr"), auto_amount);
    insert_value(
        out,
        &format!("{prefix}concentration_autoBkgr_from_vol_vox"),
        auto_amount / cell_vol_vox,
    );
    insert_value(
        out,
        &format!("{prefix}concentration_autoBkgr_from_vol_fl"),
        auto_amount / cell_vol_fl,
    );

    let data_values = data_prep_background.unwrap_or(&[]);
    let data_median = quantile_f32(data_values, 0.5);
    let data_mean = mean_f32(data_values);
    let data_q75 = quantile_f32(data_values, 0.75);
    let data_q25 = quantile_f32(data_values, 0.25);
    let data_q95 = quantile_f32(data_values, 0.95);
    let data_q05 = quantile_f32(data_values, 0.05);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_median"), data_median);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_mean"), data_mean);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_q75"), data_q75);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_q25"), data_q25);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_q95"), data_q95);
    insert_value(out, &format!("{prefix}dataPrepBkgr_bkgrVal_q05"), data_q05);
    let data_amount = if data_median.is_nan() || mean.is_nan() {
        f64::NAN
    } else {
        (mean - data_median) * area as f64
    };
    insert_value(out, &format!("{prefix}amount_dataPrepBkgr"), data_amount);
    insert_value(
        out,
        &format!("{prefix}concentration_dataPrepBkgr_from_vol_vox"),
        data_amount / cell_vol_vox,
    );
    insert_value(
        out,
        &format!("{prefix}concentration_dataPrepBkgr_from_vol_fl"),
        data_amount / cell_vol_fl,
    );
}

fn insert_regionprop_values(out: &mut BTreeMap<String, f64>, region: &RegionMeasurements) {
    insert_value(out, "inertia_tensor_eigvals-0", region.inertia_eig0);
    insert_value(out, "inertia_tensor_eigvals-1", region.inertia_eig1);
    insert_value(out, "major_axis_length", region.major_axis_length);
    insert_value(out, "minor_axis_length", region.minor_axis_length);
    insert_value(out, "eccentricity", region.eccentricity);
    insert_value(out, "circularity", region.circularity);
    insert_value(out, "roundness", region.roundness);
    insert_value(out, "aspect_ratio", region.aspect_ratio);
    insert_value(out, "equivalent_diameter", region.equivalent_diameter);
    insert_value(out, "area", region.area);
    insert_value(out, "solidity", region.solidity);
    insert_value(out, "feret_diameter_max", region.feret_diameter_max);
    insert_value(out, "extent", region.extent);
    insert_value(out, "filled_area", region.filled_area);
    insert_value(out, "convex_area", region.convex_area);
    insert_value(out, "euler_number", region.euler_number);
    insert_value(out, "bbox_area", region.bbox_area);
    insert_value(out, "centroid-0", region.centroid_y);
    insert_value(out, "centroid-1", region.centroid_x);
    insert_value(out, "local_centroid-0", region.local_centroid_y);
    insert_value(out, "local_centroid-1", region.local_centroid_x);
    insert_value(out, "bbox-0", region.bbox_min_y);
    insert_value(out, "bbox-1", region.bbox_min_x);
    insert_value(out, "bbox-2", region.bbox_max_y);
    insert_value(out, "bbox-3", region.bbox_max_x);
}

fn insert_value(out: &mut BTreeMap<String, f64>, key: &str, value: f64) {
    out.insert(key.to_string(), value);
}

fn extract_regions(mask_frame: &[u32], height: usize, width: usize) -> Vec<FrameRegion> {
    #[derive(Default)]
    struct Accumulator {
        area: usize,
        sum_y: f64,
        sum_x: f64,
        min_y: usize,
        min_x: usize,
        max_y: usize,
        max_x: usize,
        pixels: Vec<(usize, usize)>,
    }

    let mut map = BTreeMap::<u32, Accumulator>::new();
    for y in 0..height {
        for x in 0..width {
            let label = mask_frame[y * width + x];
            if label == 0 {
                continue;
            }
            let entry = map.entry(label).or_insert_with(|| Accumulator {
                min_y: y,
                min_x: x,
                max_y: y + 1,
                max_x: x + 1,
                ..Accumulator::default()
            });
            entry.area += 1;
            entry.sum_y += y as f64;
            entry.sum_x += x as f64;
            entry.min_y = entry.min_y.min(y);
            entry.min_x = entry.min_x.min(x);
            entry.max_y = entry.max_y.max(y + 1);
            entry.max_x = entry.max_x.max(x + 1);
            entry.pixels.push((y, x));
        }
    }

    map.into_iter()
        .map(|(label, acc)| FrameRegion {
            label,
            area: acc.area,
            bbox: (acc.min_y, acc.min_x, acc.max_y, acc.max_x),
            centroid_y: acc.sum_y / acc.area as f64,
            centroid_x: acc.sum_x / acc.area as f64,
            pixels: acc.pixels,
        })
        .collect()
}

fn compute_region_measurements(region: &FrameRegion) -> RegionMeasurements {
    let (min_y, min_x, max_y, max_x) = region.bbox;
    let bbox_height = max_y - min_y;
    let bbox_width = max_x - min_x;
    let bbox_area = (bbox_height * bbox_width) as f64;

    let mut local_mask = vec![false; bbox_height * bbox_width];
    for &(y, x) in &region.pixels {
        local_mask[(y - min_y) * bbox_width + (x - min_x)] = true;
    }

    let perimeter = perimeter_length(&local_mask, bbox_height, bbox_width);
    let (components, holes, hole_area) = component_and_hole_stats(&local_mask, bbox_height, bbox_width);
    let filled_area = (region.area + hole_area) as f64;

    let points = pixel_square_corners(&region.pixels);
    let hull = convex_hull(points);
    let convex_area = polygon_area(&hull);
    let feret_diameter_max = max_pair_distance(&hull);
    let solidity = if convex_area > 0.0 {
        region.area as f64 / convex_area
    } else {
        f64::NAN
    };
    let extent = if bbox_area > 0.0 {
        region.area as f64 / bbox_area
    } else {
        f64::NAN
    };

    let (eig0, eig1, major_axis_length, minor_axis_length, orientation) = inertia_metrics(region);
    let eccentricity = if eig0 > 0.0 {
        (1.0 - (eig1.max(0.0) / eig0)).max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let aspect_ratio = if minor_axis_length > 0.0 {
        major_axis_length / minor_axis_length
    } else {
        f64::NAN
    };
    let circularity = if perimeter > 0.0 {
        4.0 * PI * region.area as f64 / (perimeter * perimeter)
    } else {
        f64::NAN
    };
    let roundness = if major_axis_length > 0.0 {
        4.0 * region.area as f64 / PI / (major_axis_length * major_axis_length)
    } else {
        f64::NAN
    };

    RegionMeasurements {
        major_axis_length,
        minor_axis_length,
        eccentricity,
        aspect_ratio,
        circularity,
        roundness,
        equivalent_diameter: (4.0 * region.area as f64 / PI).sqrt(),
        area: region.area as f64,
        solidity,
        extent,
        feret_diameter_max,
        filled_area,
        convex_area,
        euler_number: (components as isize - holes as isize) as f64,
        bbox_area,
        centroid_y: region.centroid_y,
        centroid_x: region.centroid_x,
        local_centroid_y: region.centroid_y - min_y as f64,
        local_centroid_x: region.centroid_x - min_x as f64,
        bbox_min_y: min_y as f64,
        bbox_min_x: min_x as f64,
        bbox_max_y: max_y as f64,
        bbox_max_x: max_x as f64,
        inertia_eig0: eig0,
        inertia_eig1: eig1,
        orientation,
    }
}

fn rotational_volume(
    region: &FrameRegion,
    orientation: f64,
    physical_size_y: f64,
    physical_size_x: f64,
) -> (f64, f64) {
    if region.pixels.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let ux = orientation.cos();
    let uy = orientation.sin();
    let vx = -uy;
    let vy = ux;
    let cx = region.centroid_x;
    let cy = region.centroid_y;
    let mut min_u = f64::INFINITY;
    for &(y, x) in &region.pixels {
        let px = x as f64 - cx;
        let py = y as f64 - cy;
        min_u = min_u.min(px * ux + py * uy);
    }

    let mut bins = BTreeMap::<i64, (f64, f64)>::new();
    for &(y, x) in &region.pixels {
        let px = x as f64 - cx;
        let py = y as f64 - cy;
        let u = px * ux + py * uy;
        let v = px * vx + py * vy;
        let bin = (u - min_u).floor() as i64;
        let entry = bins.entry(bin).or_insert((v, v));
        entry.0 = entry.0.min(v);
        entry.1 = entry.1.max(v);
    }

    let mut vol_vox = 0.0;
    for (_, (min_v, max_v)) in bins {
        let width = (max_v - min_v + 1.0).max(1.0);
        let radius = width / 2.0;
        vol_vox += PI * radius * radius;
    }

    let vox_to_fl = physical_size_y * physical_size_x * physical_size_x;
    (vol_vox, vol_vox * vox_to_fl)
}

fn inertia_metrics(region: &FrameRegion) -> (f64, f64, f64, f64, f64) {
    let cx = region.centroid_x;
    let cy = region.centroid_y;
    let mut mu20 = 0.0;
    let mut mu02 = 0.0;
    let mut mu11 = 0.0;

    for &(y, x) in &region.pixels {
        let dx = x as f64 - cx;
        let dy = y as f64 - cy;
        mu20 += dx * dx;
        mu02 += dy * dy;
        mu11 += dx * dy;
    }

    let area = region.area as f64;
    let cov_xx = mu20 / area;
    let cov_yy = mu02 / area;
    let cov_xy = mu11 / area;
    let trace = cov_xx + cov_yy;
    let delta = ((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy).sqrt();
    let eig0 = ((trace + delta) / 2.0).max(0.0);
    let eig1 = ((trace - delta) / 2.0).max(0.0);
    let major_axis_length = 4.0 * eig0.sqrt();
    let minor_axis_length = 4.0 * eig1.sqrt();
    let orientation = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
    (eig0, eig1, major_axis_length, minor_axis_length, orientation)
}

fn perimeter_length(mask: &[bool], height: usize, width: usize) -> f64 {
    let mut perimeter = 0.0;
    for y in 0..height {
        for x in 0..width {
            if !mask[y * width + x] {
                continue;
            }
            for (dy, dx) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let ny = y as isize + dy;
                let nx = x as isize + dx;
                if ny < 0
                    || nx < 0
                    || ny >= height as isize
                    || nx >= width as isize
                    || !mask[ny as usize * width + nx as usize]
                {
                    perimeter += 1.0;
                }
            }
        }
    }
    perimeter
}

fn component_and_hole_stats(mask: &[bool], height: usize, width: usize) -> (usize, usize, usize) {
    let components = count_components(mask, height, width).0;
    let exterior = flood_background(mask, height, width);
    let mut hole_mask = vec![false; mask.len()];
    for idx in 0..mask.len() {
        if !mask[idx] && !exterior[idx] {
            hole_mask[idx] = true;
        }
    }
    let (holes, hole_area) = count_components(&hole_mask, height, width);
    (components, holes, hole_area)
}

fn count_components(mask: &[bool], height: usize, width: usize) -> (usize, usize) {
    let mut visited = vec![false; mask.len()];
    let mut components = 0usize;
    let mut total_area = 0usize;
    for idx in 0..mask.len() {
        if visited[idx] || !mask[idx] {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([idx]);
        visited[idx] = true;
        while let Some(current) = queue.pop_front() {
            total_area += 1;
            let y = current / width;
            let x = current % width;
            for (dy, dx) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let ny = y as isize + dy;
                let nx = x as isize + dx;
                if ny < 0 || nx < 0 || ny >= height as isize || nx >= width as isize {
                    continue;
                }
                let next = ny as usize * width + nx as usize;
                if visited[next] || !mask[next] {
                    continue;
                }
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }
    (components, total_area)
}

fn flood_background(mask: &[bool], height: usize, width: usize) -> Vec<bool> {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();

    for x in 0..width {
        for y in [0usize, height.saturating_sub(1)] {
            let idx = y * width + x;
            if !mask[idx] && !visited[idx] {
                visited[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    for y in 0..height {
        for x in [0usize, width.saturating_sub(1)] {
            let idx = y * width + x;
            if !mask[idx] && !visited[idx] {
                visited[idx] = true;
                queue.push_back(idx);
            }
        }
    }

    while let Some(current) = queue.pop_front() {
        let y = current / width;
        let x = current % width;
        for (dy, dx) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let ny = y as isize + dy;
            let nx = x as isize + dx;
            if ny < 0 || nx < 0 || ny >= height as isize || nx >= width as isize {
                continue;
            }
            let next = ny as usize * width + nx as usize;
            if mask[next] || visited[next] {
                continue;
            }
            visited[next] = true;
            queue.push_back(next);
        }
    }

    visited
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

fn max_pair_distance(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return f64::NAN;
    }
    let mut max_distance: f64 = 0.0;
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            max_distance = max_distance.max(distance(points[i].0, points[i].1, points[j].0, points[j].1));
        }
    }
    max_distance
}

fn distance(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    (dx * dx + dy * dy).sqrt()
}

fn mark_disappearances(
    masks: &[u32],
    shape: StackShape,
    row_indices: &HashMap<(usize, u32), usize>,
    rows: &mut [MeasurementRow],
) {
    let frame_len = shape.height * shape.width;
    for frame_i in 0..shape.frames.saturating_sub(1) {
        let current = &masks[frame_i * frame_len..(frame_i + 1) * frame_len];
        let next = &masks[(frame_i + 1) * frame_len..(frame_i + 2) * frame_len];
        let current_labels = unique_labels(current);
        let next_labels = unique_labels(next);
        for label in current_labels.difference(&next_labels) {
            if let Some(row_idx) = row_indices.get(&(frame_i, *label)).copied() {
                rows[row_idx].disappears_before_end = 1;
            }
        }
    }
}

fn unique_labels(frame: &[u32]) -> BTreeSet<u32> {
    frame.iter().copied().filter(|label| *label != 0).collect()
}

fn build_headers(channels: &[ChannelSpec]) -> Vec<String> {
    let mut headers = vec![
        "frame_i".to_string(),
        "time_seconds".to_string(),
        "time_minutes".to_string(),
        "time_hours".to_string(),
        "Cell_ID".to_string(),
        "is_cell_dead".to_string(),
        "is_cell_excluded".to_string(),
        "was_manually_edited".to_string(),
        "x_centroid".to_string(),
        "y_centroid".to_string(),
        "cell_area_pxl".to_string(),
        "cell_area_um2".to_string(),
        "cell_vol_vox".to_string(),
        "cell_vol_fl".to_string(),
        "velocity_pixel".to_string(),
        "velocity_um".to_string(),
        "disappears_before_end".to_string(),
    ];

    for channel in channels {
        for suffix in CHANNEL_METRIC_SUFFIXES {
            headers.push(format!("{}_{}", channel.name, suffix));
        }
    }
    headers.extend(REGIONPROP_HEADERS.iter().map(|header| header.to_string()));
    headers
}

fn write_measurement_csv(path: &Path, headers: &[String], rows: &[MeasurementRow]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;

    let mut writer =
        Writer::from_path(path).with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record(headers)?;

    for row in rows {
        let mut record = Vec::with_capacity(headers.len());
        for header in headers {
            let value = match header.as_str() {
                "frame_i" => row.frame_i.to_string(),
                "time_seconds" => format_f64(row.time_seconds),
                "time_minutes" => format_f64(row.time_minutes),
                "time_hours" => format_f64(row.time_hours),
                "Cell_ID" => row.cell_id.to_string(),
                "is_cell_dead" => row.is_cell_dead.to_string(),
                "is_cell_excluded" => row.is_cell_excluded.to_string(),
                "was_manually_edited" => row.was_manually_edited.to_string(),
                "x_centroid" => row.x_centroid.to_string(),
                "y_centroid" => row.y_centroid.to_string(),
                "cell_area_pxl" => row.cell_area_pxl.to_string(),
                "cell_area_um2" => format_f64(row.cell_area_um2),
                "cell_vol_vox" => format_f64(row.cell_vol_vox),
                "cell_vol_fl" => format_f64(row.cell_vol_fl),
                "velocity_pixel" => format_f64(row.velocity_pixel),
                "velocity_um" => format_f64(row.velocity_um),
                "disappears_before_end" => row.disappears_before_end.to_string(),
                other => format_f64(*row.dynamic_values.get(other).unwrap_or(&f64::NAN)),
            };
            record.push(value);
        }
        writer.write_record(record)?;
    }

    writer.flush()?;
    Ok(())
}

fn format_f64(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else {
        value.to_string()
    }
}

fn mean_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
}

fn sum_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().map(|value| *value as f64).sum()
}

fn min_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| *value as f64)
        .reduce(f64::min)
        .unwrap_or(f64::NAN)
}

fn max_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| *value as f64)
        .reduce(f64::max)
        .unwrap_or(f64::NAN)
}

fn quantile_f32(values: &[f32], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let last = sorted.len().saturating_sub(1) as f64;
    let position = q.clamp(0.0, 1.0) * last;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn coefficient_of_variation(values: &[f32]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = mean_f32(values);
    if !mean.is_finite() || mean == 0.0 {
        return f64::NAN;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io::write_mask_npz;
    use ndarray::Array3;
    use ndarray_npy::NpzWriter;
    use std::fs::File;
    use tiff::encoder::{colortype, TiffEncoder};
    use tempfile::tempdir;

    #[test]
    fn measures_position_from_existing_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 20])?;
        write_test_stack(&images.join("demo_gfp.tif"), &[30, 40])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\nTimeIncrement,15\nPhysicalSizeX,0.5\nPhysicalSizeY,0.25\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0,
                1, 1, 0, 0,
                0, 0, 2, 2,
                0, 0, 2, 2,
                1, 1, 0, 0,
                1, 1, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            2,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
        })?;
        let csv = fs::read_to_string(result.outputs.acdc_output_csv_path)?;
        assert!(csv.contains("demo_acdc_output.csv") == false);
        assert!(csv.contains("cell_vol_vox"));
        assert!(csv.contains("phase_mean"));
        assert!(csv.contains("gfp_mean"));
        assert!(csv.contains("disappears_before_end"));
        Ok(())
    }

    #[test]
    fn loads_data_prep_background_archive() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10])?;
        fs::write(images.join("demo_metadata.csv"), "Description,values\nbasename,demo_\n")?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;
        let file = File::create(images.join("demo_phase_bkgrRoiData.npz"))?;
        let mut writer = NpzWriter::new(file);
        let array = Array3::from_shape_vec((1, 1, 2), vec![7u16, 8])?;
        writer.add_array("roi0_data", &array)?;
        writer.finish()?;

        let spec = resolve_measurement_position(temp.path().join("Position_1"))?;
        let loaded = load_channels(
            &spec,
            StackShape {
                frames: 1,
                height: 4,
                width: 4,
            },
        )?;
        assert_eq!(loaded[0].background_arrays.len(), 1);
        Ok(())
    }

    #[test]
    fn parses_data_prep_roi_json() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("rois.json");
        fs::write(&path, r#"[{"pos":[1,1],"size":[2,2]}]"#)?;
        let mask = load_data_prep_roi_mask(Some(&path), 5, 5)?.unwrap();
        assert!(mask[1 * 5 + 1]);
        assert!(mask[2 * 5 + 2]);
        assert!(!mask[0]);
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
}
