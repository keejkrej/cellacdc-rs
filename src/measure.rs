use crate::image_io::{
    load_image_stack_as_f32, load_image_volume_as_f32, load_npz_archive_arrays_as_f32,
    NamedArrayF32, StackShape, VolumeShape,
};
use crate::layout::{
    discover_measurement_experiment, resolve_measurement_position, ChannelSpec,
    MeasurementExperimentSpec, MeasurementPositionSpec, PositionSpec,
};
use crate::mask_io::{load_mask_data, MaskData, MaskPathResolution, SegmentationLayout};
use crate::runner::OverwritePolicy;
use crate::segm_info::{clamp_segm_info_z_slices, load_segm_info, SegmInfoRecord, SegmInfoTable};
use crate::tabular::{write_table, Table, TableValue};
use crate::utilities::objects_count_summary;
use crate::zstack::{count_mask_volume_labels, project_frame_f32, project_mask_volume_max};
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
    pub stop_frame: Option<usize>,
    pub channel_names: Option<Vec<String>>,
    pub metric_options: Option<MeasurementMetricOptions>,
    pub save_object_counts_table: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementExperimentConfig {
    pub experiment_dir: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub stop_frame: Option<usize>,
    pub channel_names: Option<Vec<String>>,
    pub metric_options: Option<MeasurementMetricOptions>,
    pub save_object_counts_table: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeasurementMetricOptions {
    pub channel_metrics: Option<BTreeMap<String, Vec<String>>>,
    pub channel_metrics_to_skip: BTreeMap<String, Vec<String>>,
    pub calc_for_each_zslice_channels: BTreeMap<String, bool>,
    pub calc_size_for_each_zslice: bool,
    pub size_metrics: Option<Vec<String>>,
    pub regionprops: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementOutputPaths {
    pub segm_npz_path: PathBuf,
    pub acdc_output_csv_path: PathBuf,
    pub objects_count_csv_path: PathBuf,
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
    pub mask_data: MaskData,
    pub manual_background_mask_data: Option<MaskData>,
    pub segm_info: Option<SegmInfoTable>,
    pub is_segm_3d: bool,
    pub stop_frame: Option<usize>,
}

pub(crate) fn measurement_position_from_position(
    position: &PositionSpec,
) -> MeasurementPositionSpec {
    MeasurementPositionSpec {
        position_dir: position.position_dir.clone(),
        images_dir: position.images_dir.clone(),
        basename: position.basename.clone(),
        channels: position.channels.clone(),
        metadata_path: position.metadata_path.clone(),
        data_prep_background_rois_path: position.data_prep_background_rois_path.clone(),
        data_prep_roi_coords_path: position.data_prep_roi_coords_path.clone(),
        data_prep_free_roi_path: position.data_prep_free_roi_path.clone(),
        segm_info_path: position.segm_info_path.clone(),
        size_t: position.size_t,
        size_z: position.size_z,
        time_increment: position.time_increment,
        physical_size_z: position.physical_size_z,
        physical_size_x: position.physical_size_x,
        physical_size_y: position.physical_size_y,
        segm_is_3d: position.segm_is_3d.clone(),
    }
}

#[derive(Debug, Clone)]
struct LoadedChannelData {
    spec: ChannelSpec,
    values: Vec<f32>,
    shape: LoadedChannelShape,
    background_arrays: Vec<NamedArrayF32>,
}

#[derive(Debug, Clone)]
enum LoadedChannelShape {
    Stack(StackShape),
    Volume(VolumeShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ChannelVariant {
    Base,
    MaxProj,
    MeanProj,
    ZSlice,
    ZSliceIndex(usize),
    ThreeD,
}

impl ChannelVariant {
    fn column_suffix(self) -> String {
        match self {
            Self::Base => String::new(),
            Self::MaxProj => "_maxProj".to_string(),
            Self::MeanProj => "_meanProj".to_string(),
            Self::ZSlice => "_zSlice".to_string(),
            Self::ZSliceIndex(z) => format!("_zSlice{z}"),
            Self::ThreeD => "_3D".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct MeasurementMaskContext {
    projected_frames: Vec<Vec<u32>>,
    frame_height: usize,
    frame_width: usize,
    voxel_counts_per_frame: Option<Vec<BTreeMap<u32, usize>>>,
    volume_frames: Option<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone)]
struct ChannelFrameContext {
    projected_foregrounds: BTreeMap<ChannelVariant, Vec<f32>>,
    auto_backgrounds: BTreeMap<ChannelVariant, Vec<f32>>,
    data_prep_backgrounds: BTreeMap<ChannelVariant, Vec<f32>>,
    volume_foreground: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct MeasurementRow {
    frame_i: usize,
    time_seconds: f64,
    time_minutes: f64,
    time_hours: f64,
    z_slice_used: Option<usize>,
    which_z_proj: Option<String>,
    cell_id: u32,
    cell_cycle_stage: String,
    generation_num: i32,
    relative_id: i32,
    relationship: String,
    emerg_frame_i: i32,
    division_frame_i: i32,
    is_history_known: bool,
    corrected_on_frame_i: i32,
    will_divide: u8,
    daughter_disappears_before_division: u8,
    disappears_before_division: u8,
    is_cell_dead: u8,
    is_cell_excluded: u8,
    was_manually_edited: u8,
    x_centroid: i32,
    y_centroid: i32,
    cell_area_pxl: usize,
    cell_area_um2: f64,
    cell_vol_vox: f64,
    cell_vol_fl: f64,
    cell_vol_vox_3d: f64,
    cell_vol_fl_3d: f64,
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
const MANUAL_BACKGROUND_METRIC_SUFFIXES: &[&str] = &[
    "amount_manualBkgr",
    "mean_manualBkgr",
    "manualBkgr_bkgrVal_median",
    "manualBkgr_bkgrVal_mean",
    "manualBkgr_bkgrVal_q75",
    "manualBkgr_bkgrVal_q25",
    "manualBkgr_bkgrVal_q95",
    "manualBkgr_bkgrVal_q05",
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

const CORE_MEASUREMENT_HEADERS: &[&str] = &[
    "frame_i",
    "time_seconds",
    "time_minutes",
    "time_hours",
    "z_slice_used",
    "which_z_proj",
    "Cell_ID",
    "cell_cycle_stage",
    "generation_num",
    "relative_ID",
    "relationship",
    "emerg_frame_i",
    "division_frame_i",
    "is_history_known",
    "corrected_on_frame_i",
    "will_divide",
    "daughter_disappears_before_division",
    "disappears_before_division",
    "is_cell_dead",
    "is_cell_excluded",
    "was_manually_edited",
    "x_centroid",
    "y_centroid",
    "disappears_before_end",
];

const SIZE_METRIC_HEADERS: &[&str] = &[
    "cell_area_pxl",
    "cell_area_um2",
    "cell_vol_vox",
    "cell_vol_fl",
    "cell_vol_vox_3D",
    "cell_vol_fl_3D",
    "velocity_pixel",
    "velocity_um",
];

const DEFAULT_CELL_CYCLE_STAGE: &str = "G1";
const DEFAULT_RELATIONSHIP: &str = "mother";
const DEFAULT_GENERATION_NUM: i32 = 2;
const DEFAULT_UNASSIGNED_FRAME: i32 = -1;
const DEFAULT_RELATIVE_ID: i32 = -1;

pub fn measure_position(config: MeasurementRunConfig) -> Result<MeasurementRunResult> {
    let spec = resolve_measurement_position(&config.position_path)?;
    let loaded = load_measurement_inputs(
        spec,
        config.segm_endname.as_deref(),
        config.stop_frame,
        config.channel_names.as_deref(),
    )?;
    if config.overwrite_policy == OverwritePolicy::Refuse
        && loaded.outputs.acdc_output_csv_path.exists()
    {
        bail!(
            "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
            loaded.outputs.acdc_output_csv_path.display()
        );
    }
    guard_object_counts_output(
        &loaded,
        config.overwrite_policy,
        config.save_object_counts_table,
    )?;
    write_measurements(
        &loaded,
        config.metric_options.as_ref(),
        config.save_object_counts_table,
    )
}

pub fn measure_experiment(
    config: MeasurementExperimentConfig,
) -> Result<Vec<MeasurementRunResult>> {
    let experiment = discover_measurement_experiment(&config.experiment_dir)?;
    measure_experiment_from_spec(config, experiment)
}

pub(crate) fn measure_experiment_from_spec(
    config: MeasurementExperimentConfig,
    experiment: MeasurementExperimentSpec,
) -> Result<Vec<MeasurementRunResult>> {
    let mut results = Vec::with_capacity(experiment.positions.len());
    for position in experiment.positions {
        let loaded = load_measurement_inputs(
            position,
            config.segm_endname.as_deref(),
            config.stop_frame,
            config.channel_names.as_deref(),
        )?;
        if config.overwrite_policy == OverwritePolicy::Refuse
            && loaded.outputs.acdc_output_csv_path.exists()
        {
            bail!(
                "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
                loaded.outputs.acdc_output_csv_path.display()
            );
        }
        guard_object_counts_output(
            &loaded,
            config.overwrite_policy,
            config.save_object_counts_table,
        )?;
        results.push(write_measurements(
            &loaded,
            config.metric_options.as_ref(),
            config.save_object_counts_table,
        )?);
    }
    Ok(results)
}

pub(crate) fn load_measurement_inputs(
    mut spec: MeasurementPositionSpec,
    segm_endname: Option<&str>,
    stop_frame: Option<usize>,
    channel_names: Option<&[String]>,
) -> Result<LoadedMeasurementPosition> {
    apply_measurement_channel_filter(&mut spec, channel_names)?;
    let outputs = measurement_output_paths(&spec.images_dir, &spec.basename, segm_endname);
    let segm_name = measurement_segmentation_name(segm_endname);
    let is_segm_3d = spec.segm_is_3d.get(&segm_name).copied().unwrap_or(false);
    let mask_resolution = MaskPathResolution {
        size_t: Some(spec.size_t),
        size_z: Some(if is_segm_3d { spec.size_z } else { 1 }),
        layout: None,
    };
    let mask_data =
        load_mask_data(&outputs.segm_npz_path, Some(&mask_resolution)).with_context(|| {
            format!(
                "Failed to load segmentation masks from {}",
                outputs.segm_npz_path.display()
            )
        })?;
    let manual_background_mask_data = manual_background_mask_path(&outputs.segm_npz_path)
        .filter(|path| path.exists())
        .map(|path| {
            load_mask_data(&path, Some(&mask_resolution)).with_context(|| {
                format!(
                    "Failed to load manual background masks from {}",
                    path.display()
                )
            })
        })
        .transpose()?;

    let mask_size_t = match mask_data.layout {
        SegmentationLayout::YX | SegmentationLayout::ZYX => 1,
        SegmentationLayout::TYX | SegmentationLayout::TZYX => mask_data.values.shape()[0],
    };
    if mask_size_t != spec.size_t {
        bail!(
            "Segmentation frame count ({}) does not match metadata SizeT ({}) in {}",
            mask_size_t,
            spec.size_t,
            outputs.segm_npz_path.display()
        );
    }
    let segm_info = if spec.size_z > 1 && !is_segm_3d {
        let path = spec.segm_info_path.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "Position {} is a z-stack but no _segmInfo.csv was found. Run prepare-zstack-segm-info first.",
                spec.position_dir.display()
            )
        })?;
        let mut table = load_segm_info(path)?;
        clamp_segm_info_z_slices(&mut table, spec.size_z);
        Some(table)
    } else {
        None
    };

    Ok(LoadedMeasurementPosition {
        spec,
        outputs,
        mask_data,
        manual_background_mask_data,
        segm_info,
        is_segm_3d,
        stop_frame,
    })
}

fn apply_measurement_channel_filter(
    spec: &mut MeasurementPositionSpec,
    channel_names: Option<&[String]>,
) -> Result<()> {
    let Some(channel_names) = channel_names else {
        return Ok(());
    };
    let requested = channel_names
        .iter()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if requested.is_empty() {
        spec.channels.clear();
        return Ok(());
    }
    let requested_set = requested.iter().collect::<BTreeSet<_>>();
    let available = spec
        .channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect::<BTreeSet<_>>();
    let missing = requested
        .iter()
        .filter(|name| !available.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "Measurement workflow requested missing channel(s) {} in {}",
            missing.join(", "),
            spec.position_dir.display()
        );
    }
    spec.channels
        .retain(|channel| requested_set.contains(&channel.name));
    Ok(())
}

fn guard_object_counts_output(
    loaded: &LoadedMeasurementPosition,
    overwrite_policy: OverwritePolicy,
    save_object_counts_table: bool,
) -> Result<()> {
    if !save_object_counts_table || overwrite_policy == OverwritePolicy::Overwrite {
        return Ok(());
    }
    if loaded.outputs.objects_count_csv_path.exists() {
        bail!(
            "Refusing to overwrite existing output {}. Re-run with --overwrite to replace it.",
            loaded.outputs.objects_count_csv_path.display()
        );
    }
    Ok(())
}

pub(crate) fn write_measurements(
    loaded: &LoadedMeasurementPosition,
    metric_options: Option<&MeasurementMetricOptions>,
    save_object_counts_table: bool,
) -> Result<MeasurementRunResult> {
    let mut mask_context = measurement_mask_context(&loaded.mask_data);
    let frame_limit = resolve_measurement_stop_frame(loaded.spec.size_t, loaded.stop_frame)?;
    mask_context.projected_frames.truncate(frame_limit);
    if let Some(counts) = &mut mask_context.voxel_counts_per_frame {
        counts.truncate(frame_limit);
    }
    if let Some(volumes) = &mut mask_context.volume_frames {
        volumes.truncate(frame_limit);
    }
    let mut manual_background_context = loaded
        .manual_background_mask_data
        .as_ref()
        .map(measurement_mask_context);
    if let Some(context) = &mut manual_background_context {
        context.projected_frames.truncate(frame_limit);
        if let Some(volumes) = &mut context.volume_frames {
            volumes.truncate(frame_limit);
        }
    }
    let channels = load_channels(
        &loaded.spec,
        mask_context.frame_height,
        mask_context.frame_width,
        loaded.is_segm_3d,
    )?;
    let variants_by_channel = channels
        .iter()
        .map(|channel| {
            (
                channel.spec.name.clone(),
                measurement_channel_variants(
                    loaded.spec.size_z,
                    loaded.is_segm_3d,
                    &channel.spec.name,
                    metric_options,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let roi_mask = load_data_prep_roi_mask(
        loaded.spec.data_prep_background_rois_path.as_deref(),
        mask_context.frame_height,
        mask_context.frame_width,
    )?;
    let pixel_area_um2 = loaded.spec.physical_size_x * loaded.spec.physical_size_y;
    let mut rows = Vec::new();
    let mut row_indices = HashMap::<(usize, u32), usize>::new();
    let mut previous_centroids = HashMap::<u32, (f64, f64)>::new();

    for (frame_i, mask_frame) in mask_context.projected_frames.iter().enumerate() {
        let regions = extract_regions(
            mask_frame,
            mask_context.frame_height,
            mask_context.frame_width,
        );
        let current_centroids = regions
            .iter()
            .map(|region| (region.label, (region.centroid_x, region.centroid_y)))
            .collect::<HashMap<_, _>>();
        let mask_volume = mask_context
            .volume_frames
            .as_ref()
            .and_then(|frames| frames.get(frame_i))
            .map(Vec::as_slice);
        let manual_background_frame = manual_background_context
            .as_ref()
            .and_then(|context| context.projected_frames.get(frame_i))
            .map(Vec::as_slice);
        let manual_background_volume = manual_background_context
            .as_ref()
            .and_then(|context| context.volume_frames.as_ref())
            .and_then(|frames| frames.get(frame_i))
            .map(Vec::as_slice);
        let mut channel_contexts = HashMap::<String, ChannelFrameContext>::new();
        for channel in &channels {
            let segm_info = segm_info_for_channel_from_table(
                loaded.segm_info.as_ref(),
                &channel.spec,
                frame_i,
            )?;
            let variants = variants_by_channel
                .get(&channel.spec.name)
                .expect("missing channel variants");
            let context = build_channel_frame_context(
                channel,
                frame_i,
                segm_info,
                variants,
                mask_frame,
                mask_volume,
                roi_mask.as_deref(),
            )?;
            channel_contexts.insert(channel.spec.name.clone(), context);
        }

        for region in &regions {
            let region_measurements = compute_region_measurements(region);
            let (cell_vol_vox, cell_vol_fl) = rotational_volume(
                region,
                region_measurements.orientation,
                loaded.spec.physical_size_y,
                loaded.spec.physical_size_x,
            );
            let (cell_vol_vox_3d, cell_vol_fl_3d) = mask_context
                .voxel_counts_per_frame
                .as_ref()
                .and_then(|counts| counts.get(frame_i))
                .and_then(|counts| counts.get(&region.label).copied())
                .map(|voxels| {
                    let voxels = voxels as f64;
                    (
                        voxels,
                        voxels
                            * loaded.spec.physical_size_z
                            * loaded.spec.physical_size_y
                            * loaded.spec.physical_size_x,
                    )
                })
                .unwrap_or((f64::NAN, f64::NAN));
            let mut dynamic_values = BTreeMap::new();
            insert_regionprop_values(&mut dynamic_values, &region_measurements);
            insert_zslice_size_values(
                &mut dynamic_values,
                metric_options,
                mask_volume,
                loaded.spec.size_z,
                mask_context.frame_height,
                mask_context.frame_width,
                region.label,
                loaded.spec.physical_size_x * loaded.spec.physical_size_y,
            );
            let segm_info_record = loaded
                .segm_info
                .as_ref()
                .map(|table| segm_info_for_channel(table, &loaded.spec.channels[0], frame_i))
                .transpose()?;

            for channel in &channels {
                let channel_context = channel_contexts
                    .get(&channel.spec.name)
                    .expect("missing channel frame context");
                let variants = variants_by_channel
                    .get(&channel.spec.name)
                    .expect("missing channel variants");
                for variant in variants {
                    let object_values = match variant {
                        ChannelVariant::ThreeD => channel_context
                            .volume_foreground
                            .as_ref()
                            .and_then(|volume| {
                                mask_volume.map(|mask| {
                                    collect_object_volume_values(volume, mask, region.label)
                                })
                            })
                            .unwrap_or_default(),
                        _ => channel_context
                            .projected_foregrounds
                            .get(variant)
                            .map(|frame| {
                                collect_object_values(
                                    frame,
                                    &region.pixels,
                                    mask_context.frame_width,
                                )
                            })
                            .unwrap_or_default(),
                    };
                    let auto_background = channel_context
                        .auto_backgrounds
                        .get(variant)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let data_prep_background = channel_context
                        .data_prep_backgrounds
                        .get(variant)
                        .map(Vec::as_slice);
                    let manual_background = match variant {
                        ChannelVariant::ThreeD => channel_context
                            .volume_foreground
                            .as_ref()
                            .and_then(|volume| {
                                manual_background_volume.map(|mask| {
                                    collect_object_volume_values(volume, mask, region.label)
                                })
                            }),
                        _ => channel_context
                            .projected_foregrounds
                            .get(variant)
                            .and_then(|frame| {
                                manual_background_frame
                                    .map(|mask| collect_labeled_values(frame, mask, region.label))
                            }),
                    };
                    let (variant_cell_vol_vox, variant_cell_vol_fl) =
                        if *variant == ChannelVariant::ThreeD {
                            (cell_vol_vox_3d, cell_vol_fl_3d)
                        } else {
                            (cell_vol_vox, cell_vol_fl)
                        };
                    let variant_area = if *variant == ChannelVariant::ThreeD {
                        object_values.len()
                    } else {
                        region.area
                    };
                    insert_channel_measurements(
                        &mut dynamic_values,
                        &channel.spec.name,
                        *variant,
                        &object_values,
                        auto_background,
                        data_prep_background,
                        manual_background.as_deref(),
                        variant_area,
                        variant_cell_vol_vox,
                        variant_cell_vol_fl,
                    );
                }
            }

            let previous_centroid = previous_centroids.get(&region.label).copied();
            let velocity_pixel = previous_centroid
                .map(|(prev_x, prev_y)| {
                    distance(prev_x, prev_y, region.centroid_x, region.centroid_y)
                })
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
                z_slice_used: segm_info_record
                    .as_ref()
                    .map(|record| record.z_slice_used_data_prep),
                which_z_proj: segm_info_record
                    .as_ref()
                    .map(|record| record.which_z_proj.as_str().to_string()),
                cell_id: region.label,
                cell_cycle_stage: DEFAULT_CELL_CYCLE_STAGE.to_string(),
                generation_num: DEFAULT_GENERATION_NUM,
                relative_id: DEFAULT_RELATIVE_ID,
                relationship: DEFAULT_RELATIONSHIP.to_string(),
                emerg_frame_i: DEFAULT_UNASSIGNED_FRAME,
                division_frame_i: DEFAULT_UNASSIGNED_FRAME,
                is_history_known: false,
                corrected_on_frame_i: DEFAULT_UNASSIGNED_FRAME,
                will_divide: 0,
                daughter_disappears_before_division: 0,
                disappears_before_division: 0,
                is_cell_dead: 0,
                is_cell_excluded: 0,
                was_manually_edited: 0,
                x_centroid: region.centroid_x.floor() as i32,
                y_centroid: region.centroid_y.floor() as i32,
                cell_area_pxl: region.area,
                cell_area_um2: region.area as f64 * pixel_area_um2,
                cell_vol_vox,
                cell_vol_fl,
                cell_vol_vox_3d,
                cell_vol_fl_3d,
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

    mark_disappearances_for_frames(
        &mask_context.projected_frames,
        mask_context.frame_width,
        &row_indices,
        &mut rows,
    );

    let headers = build_headers(
        &loaded.spec.channels,
        &variants_by_channel,
        loaded.spec.size_z,
        metric_options,
        loaded.manual_background_mask_data.is_some(),
    );
    write_measurement_csv(&loaded.outputs.acdc_output_csv_path, &headers, &rows)?;
    if save_object_counts_table {
        write_object_counts_csv(&loaded.outputs.objects_count_csv_path, &loaded.mask_data)?;
    }

    Ok(MeasurementRunResult {
        position_dir: loaded.spec.position_dir.clone(),
        images_dir: loaded.spec.images_dir.clone(),
        outputs: loaded.outputs.clone(),
        labels_found: loaded.mask_data.values.iter().copied().max().unwrap_or(0),
        frames_processed: frame_limit,
    })
}

fn resolve_measurement_stop_frame(total_frames: usize, stop_frame: Option<usize>) -> Result<usize> {
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

fn measurement_output_paths(
    images_dir: &Path,
    basename: &str,
    segm_endname: Option<&str>,
) -> MeasurementOutputPaths {
    let segm_name = measurement_segmentation_name(segm_endname);
    let acdc_output_name = segm_name.replacen("segm", "acdc_output", 1);
    let objects_count_name = segm_name.replacen("segm", "acdc_objects_count", 1);
    let segm_npz_path = resolve_measurement_segmentation_path(images_dir, basename, &segm_name);
    MeasurementOutputPaths {
        segm_npz_path,
        acdc_output_csv_path: images_dir.join(format!("{basename}{acdc_output_name}.csv")),
        objects_count_csv_path: images_dir.join(format!("{basename}{objects_count_name}.csv")),
    }
}

fn resolve_measurement_segmentation_path(
    images_dir: &Path,
    basename: &str,
    segm_name: &str,
) -> PathBuf {
    let canonical_npz = images_dir.join(format!("{basename}{segm_name}.npz"));
    if canonical_npz.exists() {
        return canonical_npz;
    }
    if let Some(path) = find_visible_file_by_endname(images_dir, &format!("{segm_name}.npz")) {
        return path;
    }

    let canonical_npy = images_dir.join(format!("{basename}{segm_name}.npy"));
    if canonical_npy.exists() {
        return canonical_npy;
    }
    find_visible_file_by_endname(images_dir, &format!("{segm_name}.npy")).unwrap_or(canonical_npz)
}

fn measurement_segmentation_name(endname: Option<&str>) -> String {
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

fn manual_background_mask_path(segm_npz_path: &Path) -> Option<PathBuf> {
    let file_name = segm_npz_path.file_name()?.to_str()?;
    let manual_name = file_name.replacen("segm", "manualBackground", 1);
    if manual_name == file_name {
        return None;
    }
    Some(segm_npz_path.with_file_name(manual_name))
}

fn find_visible_file_by_endname(dir: &Path, endname: &str) -> Option<PathBuf> {
    let mut matches = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if !path.is_file() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            if should_skip_python_listdir_entry(name) || !name.ends_with(endname) {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        let left_name = left
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let right_name = right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        left_name
            .len()
            .cmp(&right_name.len())
            .then_with(|| left_name.cmp(right_name))
            .then_with(|| left.cmp(right))
    });
    matches.into_iter().next()
}

fn should_skip_python_listdir_entry(name: &str) -> bool {
    name.starts_with('.')
        || name == "desktop.ini"
        || name == "recovery"
        || name.ends_with(".new.npz")
}

fn measurement_mask_context(mask_data: &MaskData) -> MeasurementMaskContext {
    match mask_data.layout {
        SegmentationLayout::YX => {
            let shape = mask_data.values.shape();
            MeasurementMaskContext {
                projected_frames: vec![mask_data.values.iter().copied().collect()],
                frame_height: shape[0],
                frame_width: shape[1],
                voxel_counts_per_frame: None,
                volume_frames: None,
            }
        }
        SegmentationLayout::TYX => {
            let shape = mask_data.values.shape();
            let plane_len = shape[1] * shape[2];
            let mut frames = Vec::with_capacity(shape[0]);
            for frame_i in 0..shape[0] {
                let start = frame_i * plane_len;
                frames.push(
                    mask_data
                        .values
                        .iter()
                        .copied()
                        .skip(start)
                        .take(plane_len)
                        .collect(),
                );
            }
            MeasurementMaskContext {
                projected_frames: frames,
                frame_height: shape[1],
                frame_width: shape[2],
                voxel_counts_per_frame: None,
                volume_frames: None,
            }
        }
        SegmentationLayout::ZYX => {
            let shape = mask_data.values.shape();
            let values = mask_data.values.iter().copied().collect::<Vec<_>>();
            MeasurementMaskContext {
                projected_frames: vec![project_mask_volume_max(
                    &values, shape[0], shape[1], shape[2],
                )],
                frame_height: shape[1],
                frame_width: shape[2],
                voxel_counts_per_frame: Some(vec![count_mask_volume_labels(&values)]),
                volume_frames: Some(vec![values]),
            }
        }
        SegmentationLayout::TZYX => {
            let shape = mask_data.values.shape();
            let frame_len = shape[1] * shape[2] * shape[3];
            let mut projected_frames = Vec::with_capacity(shape[0]);
            let mut voxel_counts = Vec::with_capacity(shape[0]);
            let mut volume_frames = Vec::with_capacity(shape[0]);
            let values = mask_data.values.iter().copied().collect::<Vec<_>>();
            for frame_i in 0..shape[0] {
                let start = frame_i * frame_len;
                let frame = &values[start..start + frame_len];
                projected_frames.push(project_mask_volume_max(frame, shape[1], shape[2], shape[3]));
                voxel_counts.push(count_mask_volume_labels(frame));
                volume_frames.push(frame.to_vec());
            }
            MeasurementMaskContext {
                projected_frames,
                frame_height: shape[2],
                frame_width: shape[3],
                voxel_counts_per_frame: Some(voxel_counts),
                volume_frames: Some(volume_frames),
            }
        }
    }
}

fn segm_info_for_channel<'a>(
    table: &'a SegmInfoTable,
    channel: &ChannelSpec,
    frame_i: usize,
) -> Result<&'a SegmInfoRecord> {
    let filename = channel
        .image_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!("Invalid image filename in {}", channel.image_path.display())
        })?;
    table.get(filename, frame_i).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing _segmInfo entry for file {:?} frame {}",
            filename,
            frame_i
        )
    })
}

fn segm_info_for_channel_from_table<'a>(
    table: Option<&'a SegmInfoTable>,
    channel: &ChannelSpec,
    frame_i: usize,
) -> Result<Option<&'a SegmInfoRecord>> {
    match table {
        Some(table) => Ok(Some(segm_info_for_channel(table, channel, frame_i)?)),
        None => Ok(None),
    }
}

fn measurement_channel_variants(
    size_z: usize,
    is_segm_3d: bool,
    channel_name: &str,
    metric_options: Option<&MeasurementMetricOptions>,
) -> Vec<ChannelVariant> {
    if size_z <= 1 {
        return vec![ChannelVariant::Base];
    }

    let mut variants = vec![
        ChannelVariant::MaxProj,
        ChannelVariant::MeanProj,
        ChannelVariant::ZSlice,
    ];
    if metric_options
        .and_then(|options| {
            options
                .calc_for_each_zslice_channels
                .get(&normalize_metric_key(channel_name))
        })
        .copied()
        .unwrap_or(false)
    {
        variants.extend((0..size_z).map(ChannelVariant::ZSliceIndex));
    }
    if is_segm_3d {
        variants.push(ChannelVariant::ThreeD);
    }
    variants
}

fn z_slice_for_variant(segm_info: Option<&SegmInfoRecord>, shape: VolumeShape) -> usize {
    segm_info
        .map(|record| record.z_slice_used_data_prep)
        .unwrap_or(shape.size_z / 2)
}

fn volume_frame_slice<'a>(
    values: &'a [f32],
    shape: VolumeShape,
    frame_i: usize,
) -> Result<&'a [f32]> {
    let frame_len = shape.size_z * shape.height * shape.width;
    match shape.size_t {
        1 => Ok(values),
        frames if frame_i < frames => {
            let start = frame_i * frame_len;
            Ok(&values[start..start + frame_len])
        }
        _ => bail!(
            "Image volume frame index {} exceeds available frames {}",
            frame_i,
            shape.size_t
        ),
    }
}

fn project_volume_frame_variant(
    frame: &[f32],
    shape: VolumeShape,
    z_slice: usize,
    variant: ChannelVariant,
) -> Result<Vec<f32>> {
    match variant {
        ChannelVariant::MaxProj => project_frame_f32(
            frame,
            VolumeShape { size_t: 1, ..shape },
            0,
            z_slice,
            crate::segm_info::ZProjectionMode::MaxZProjection,
        ),
        ChannelVariant::MeanProj => project_frame_f32(
            frame,
            VolumeShape { size_t: 1, ..shape },
            0,
            z_slice,
            crate::segm_info::ZProjectionMode::MeanZProjection,
        ),
        ChannelVariant::ZSlice => project_frame_f32(
            frame,
            VolumeShape { size_t: 1, ..shape },
            0,
            z_slice,
            crate::segm_info::ZProjectionMode::SingleZSlice,
        ),
        ChannelVariant::ZSliceIndex(index) => project_frame_f32(
            frame,
            VolumeShape { size_t: 1, ..shape },
            0,
            index,
            crate::segm_info::ZProjectionMode::SingleZSlice,
        ),
        ChannelVariant::Base | ChannelVariant::ThreeD => {
            bail!("Variant {:?} is not a projected 2D variant", variant)
        }
    }
}

fn metric_column_name(channel_name: &str, metric_suffix: &str, variant: ChannelVariant) -> String {
    format!("{channel_name}_{metric_suffix}{}", variant.column_suffix())
}

fn load_channels(
    spec: &MeasurementPositionSpec,
    frame_height: usize,
    frame_width: usize,
    is_segm_3d: bool,
) -> Result<Vec<LoadedChannelData>> {
    let mut channels = Vec::with_capacity(spec.channels.len());
    for channel in &spec.channels {
        let (values, shape) = if spec.size_z > 1 {
            let (values, shape) =
                load_image_volume_as_f32(&channel.image_path, Some(spec.size_t), Some(spec.size_z))
                    .with_context(|| {
                        format!(
                            "Failed to load channel image {}",
                            channel.image_path.display()
                        )
                    })?;
            if shape.height != frame_height
                || shape.width != frame_width
                || shape.size_t != spec.size_t
                || (!is_segm_3d && shape.size_z != spec.size_z)
            {
                bail!(
                    "Image stack {} has shape {}x{}x{}x{}, expected {}x{}x{}x{}",
                    channel.image_path.display(),
                    shape.size_t,
                    shape.size_z,
                    shape.height,
                    shape.width,
                    spec.size_t,
                    spec.size_z,
                    frame_height,
                    frame_width
                );
            }
            (values, LoadedChannelShape::Volume(shape))
        } else {
            let (values, shape) =
                load_image_stack_as_f32(&channel.image_path).with_context(|| {
                    format!(
                        "Failed to load channel image {}",
                        channel.image_path.display()
                    )
                })?;
            if shape.height != frame_height
                || shape.width != frame_width
                || shape.frames != spec.size_t
            {
                bail!(
                    "Image stack {} has shape {}x{}x{}, expected {}x{}x{}",
                    channel.image_path.display(),
                    shape.frames,
                    shape.height,
                    shape.width,
                    spec.size_t,
                    frame_height,
                    frame_width
                );
            }
            (values, LoadedChannelShape::Stack(shape))
        };
        let background_arrays = match &channel.background_data_path {
            Some(path) => load_npz_archive_arrays_as_f32(path)
                .with_context(|| format!("Failed to load background data {}", path.display()))?,
            None => Vec::new(),
        };
        channels.push(LoadedChannelData {
            spec: channel.clone(),
            values,
            shape,
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
        let Some((x, y)) = parse_pair(
            item.get("pos")
                .or_else(|| item.get("state").and_then(|v| v.get("pos"))),
        ) else {
            continue;
        };
        let Some((w, h)) = parse_pair(
            item.get("size")
                .or_else(|| item.get("state").and_then(|v| v.get("size"))),
        ) else {
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

fn build_channel_frame_context(
    channel: &LoadedChannelData,
    frame_i: usize,
    segm_info: Option<&SegmInfoRecord>,
    variants: &[ChannelVariant],
    mask_frame: &[u32],
    mask_volume: Option<&[u32]>,
    roi_mask: Option<&[bool]>,
) -> Result<ChannelFrameContext> {
    let mut projected_foregrounds = BTreeMap::new();
    let mut auto_backgrounds = BTreeMap::new();
    let mut data_prep_backgrounds = BTreeMap::new();
    let volume_foreground = match channel.shape {
        LoadedChannelShape::Stack(shape) => {
            let frame = frame_slice(&channel.values, shape, frame_i)?.to_vec();
            let auto_background = collect_masked_values(&frame, &build_background_mask(mask_frame));
            projected_foregrounds.insert(ChannelVariant::Base, frame.clone());
            auto_backgrounds.insert(ChannelVariant::Base, auto_background);
            if let Some(values) =
                collect_archive_background(channel, frame_i, ChannelVariant::Base)?
            {
                data_prep_backgrounds.insert(ChannelVariant::Base, values);
            } else if let Some(roi_mask) = roi_mask {
                if let Some(values) =
                    collect_projected_data_prep_background(&frame, roi_mask, mask_frame)
                {
                    data_prep_backgrounds.insert(ChannelVariant::Base, values);
                }
            }
            None
        }
        LoadedChannelShape::Volume(shape) => {
            let z_slice = z_slice_for_variant(segm_info, shape);
            let volume_frame = volume_frame_slice(&channel.values, shape, frame_i)?.to_vec();
            let projected_background_mask = build_background_mask(mask_frame);
            for variant in variants
                .iter()
                .copied()
                .filter(|variant| *variant != ChannelVariant::ThreeD)
            {
                let projected =
                    project_volume_frame_variant(&volume_frame, shape, z_slice, variant)?;
                let auto_background = collect_masked_values(&projected, &projected_background_mask);
                projected_foregrounds.insert(variant, projected.clone());
                auto_backgrounds.insert(variant, auto_background);

                if let Some(values) = collect_archive_background(channel, frame_i, variant)? {
                    data_prep_backgrounds.insert(variant, values);
                } else if let Some(roi_mask) = roi_mask {
                    if let Some(values) =
                        collect_projected_data_prep_background(&projected, roi_mask, mask_frame)
                    {
                        data_prep_backgrounds.insert(variant, values);
                    }
                }
            }

            if variants.contains(&ChannelVariant::ThreeD) {
                if let Some(mask_volume) = mask_volume {
                    auto_backgrounds.insert(
                        ChannelVariant::ThreeD,
                        collect_object_volume_values(&volume_frame, mask_volume, 0),
                    );
                    if let Some(values) =
                        collect_archive_background(channel, frame_i, ChannelVariant::ThreeD)?
                    {
                        data_prep_backgrounds.insert(ChannelVariant::ThreeD, values);
                    } else if let Some(roi_mask) = roi_mask {
                        let plane_len = shape.height * shape.width;
                        if let Some(values) = collect_volume_data_prep_background(
                            &volume_frame,
                            roi_mask,
                            mask_volume,
                            plane_len,
                        ) {
                            data_prep_backgrounds.insert(ChannelVariant::ThreeD, values);
                        }
                    }
                }
            }

            Some(volume_frame)
        }
    };

    Ok(ChannelFrameContext {
        projected_foregrounds,
        auto_backgrounds,
        data_prep_backgrounds,
        volume_foreground,
    })
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

fn collect_object_volume_values(volume_frame: &[f32], mask_volume: &[u32], label: u32) -> Vec<f32> {
    volume_frame
        .iter()
        .zip(mask_volume.iter())
        .filter_map(|(value, current_label)| (*current_label == label).then_some(*value))
        .collect()
}

fn collect_labeled_values(values: &[f32], labels: &[u32], label: u32) -> Vec<f32> {
    values
        .iter()
        .zip(labels.iter())
        .filter_map(|(value, current_label)| (*current_label == label).then_some(*value))
        .collect()
}

fn collect_masked_values(values: &[f32], mask: &[bool]) -> Vec<f32> {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, keep)| keep.then_some(*value))
        .collect()
}

fn collect_projected_data_prep_background(
    channel_frame: &[f32],
    roi_mask: &[bool],
    mask_frame: &[u32],
) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    for (idx, value) in channel_frame.iter().enumerate() {
        if roi_mask[idx] && mask_frame[idx] == 0 {
            values.push(*value);
        }
    }
    (!values.is_empty()).then_some(values)
}

fn collect_volume_data_prep_background(
    volume_frame: &[f32],
    roi_mask: &[bool],
    mask_volume: &[u32],
    plane_len: usize,
) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    for (idx, value) in volume_frame.iter().enumerate() {
        let yx_idx = idx % plane_len;
        if roi_mask[yx_idx] && mask_volume[idx] == 0 {
            values.push(*value);
        }
    }
    (!values.is_empty()).then_some(values)
}

fn collect_archive_background(
    channel: &LoadedChannelData,
    frame_i: usize,
    variant: ChannelVariant,
) -> Result<Option<Vec<f32>>> {
    if channel.background_arrays.is_empty() {
        return Ok(None);
    }
    if variant == ChannelVariant::ThreeD {
        return Ok(None);
    }
    let mut values = Vec::new();
    for array in &channel.background_arrays {
        values.extend(frame_slice(&array.values, array.shape, frame_i)?);
    }
    Ok((!values.is_empty()).then_some(values))
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
    variant: ChannelVariant,
    object_values: &[f32],
    auto_background: &[f32],
    data_prep_background: Option<&[f32]>,
    manual_background: Option<&[f32]>,
    area: usize,
    cell_vol_vox: f64,
    cell_vol_fl: f64,
) {
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

    insert_value(
        out,
        &metric_column_name(channel_name, "mean", variant),
        mean,
    );
    insert_value(out, &metric_column_name(channel_name, "sum", variant), sum);
    insert_value(
        out,
        &metric_column_name(channel_name, "median", variant),
        median,
    );
    insert_value(out, &metric_column_name(channel_name, "min", variant), min);
    insert_value(out, &metric_column_name(channel_name, "max", variant), max);
    insert_value(out, &metric_column_name(channel_name, "q25", variant), q25);
    insert_value(out, &metric_column_name(channel_name, "q75", variant), q75);
    insert_value(out, &metric_column_name(channel_name, "q05", variant), q05);
    insert_value(out, &metric_column_name(channel_name, "q95", variant), q95);
    insert_value(out, &metric_column_name(channel_name, "CV", variant), cv);

    let auto_median = quantile_f32(auto_background, 0.5);
    let auto_mean = mean_f32(auto_background);
    let auto_q75 = quantile_f32(auto_background, 0.75);
    let auto_q25 = quantile_f32(auto_background, 0.25);
    let auto_q95 = quantile_f32(auto_background, 0.95);
    let auto_q05 = quantile_f32(auto_background, 0.05);
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_median", variant),
        auto_median,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_mean", variant),
        auto_mean,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_q75", variant),
        auto_q75,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_q25", variant),
        auto_q25,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_q95", variant),
        auto_q95,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "autoBkgr_bkgrVal_q05", variant),
        auto_q05,
    );
    let auto_amount = if auto_median.is_nan() || mean.is_nan() {
        f64::NAN
    } else {
        (mean - auto_median) * area as f64
    };
    insert_value(
        out,
        &metric_column_name(channel_name, "amount_autoBkgr", variant),
        auto_amount,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "concentration_autoBkgr_from_vol_vox", variant),
        auto_amount / cell_vol_vox,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "concentration_autoBkgr_from_vol_fl", variant),
        auto_amount / cell_vol_fl,
    );

    let data_values = data_prep_background.unwrap_or(&[]);
    let data_median = quantile_f32(data_values, 0.5);
    let data_mean = mean_f32(data_values);
    let data_q75 = quantile_f32(data_values, 0.75);
    let data_q25 = quantile_f32(data_values, 0.25);
    let data_q95 = quantile_f32(data_values, 0.95);
    let data_q05 = quantile_f32(data_values, 0.05);
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_median", variant),
        data_median,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_mean", variant),
        data_mean,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_q75", variant),
        data_q75,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_q25", variant),
        data_q25,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_q95", variant),
        data_q95,
    );
    insert_value(
        out,
        &metric_column_name(channel_name, "dataPrepBkgr_bkgrVal_q05", variant),
        data_q05,
    );
    let data_amount = if data_median.is_nan() || mean.is_nan() {
        f64::NAN
    } else {
        (mean - data_median) * area as f64
    };
    insert_value(
        out,
        &metric_column_name(channel_name, "amount_dataPrepBkgr", variant),
        data_amount,
    );
    insert_value(
        out,
        &metric_column_name(
            channel_name,
            "concentration_dataPrepBkgr_from_vol_vox",
            variant,
        ),
        data_amount / cell_vol_vox,
    );
    insert_value(
        out,
        &metric_column_name(
            channel_name,
            "concentration_dataPrepBkgr_from_vol_fl",
            variant,
        ),
        data_amount / cell_vol_fl,
    );

    if let Some(manual_values) = manual_background {
        let manual_median = quantile_f32(manual_values, 0.5);
        let manual_mean = mean_f32(manual_values);
        let manual_q75 = quantile_f32(manual_values, 0.75);
        let manual_q25 = quantile_f32(manual_values, 0.25);
        let manual_q95 = quantile_f32(manual_values, 0.95);
        let manual_q05 = quantile_f32(manual_values, 0.05);
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_median", variant),
            manual_median,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_mean", variant),
            manual_mean,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_q75", variant),
            manual_q75,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_q25", variant),
            manual_q25,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_q95", variant),
            manual_q95,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "manualBkgr_bkgrVal_q05", variant),
            manual_q05,
        );
        let manual_amount = if manual_mean.is_nan() || mean.is_nan() {
            f64::NAN
        } else {
            (mean - manual_mean) * area as f64
        };
        insert_value(
            out,
            &metric_column_name(channel_name, "amount_manualBkgr", variant),
            manual_amount,
        );
        insert_value(
            out,
            &metric_column_name(channel_name, "mean_manualBkgr", variant),
            mean - manual_mean,
        );
    }
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

fn insert_zslice_size_values(
    out: &mut BTreeMap<String, f64>,
    metric_options: Option<&MeasurementMetricOptions>,
    mask_volume: Option<&[u32]>,
    size_z: usize,
    height: usize,
    width: usize,
    label: u32,
    pixel_area_um2: f64,
) {
    if !metric_options
        .map(|options| options.calc_size_for_each_zslice)
        .unwrap_or(false)
        || size_z <= 1
    {
        return;
    }

    let plane_len = height * width;
    for z in 0..size_z {
        let area_pxl = mask_volume
            .and_then(|volume| volume.get(z * plane_len..(z + 1) * plane_len))
            .map(|plane| plane.iter().filter(|value| **value == label).count() as f64)
            .unwrap_or(f64::NAN);
        insert_value(out, &format!("cell_area_pxl_zslice{z}"), area_pxl);
        insert_value(
            out,
            &format!("cell_area_um2_zslice{z}"),
            area_pxl * pixel_area_um2,
        );
    }
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
    let (components, holes, hole_area) =
        component_and_hole_stats(&local_mask, bbox_height, bbox_width);
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
    (
        eig0,
        eig1,
        major_axis_length,
        minor_axis_length,
        orientation,
    )
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
            max_distance =
                max_distance.max(distance(points[i].0, points[i].1, points[j].0, points[j].1));
        }
    }
    max_distance
}

fn distance(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    (dx * dx + dy * dy).sqrt()
}

fn mark_disappearances_for_frames(
    frames: &[Vec<u32>],
    _width: usize,
    row_indices: &HashMap<(usize, u32), usize>,
    rows: &mut [MeasurementRow],
) {
    for frame_i in 0..frames.len().saturating_sub(1) {
        let current = &frames[frame_i];
        let next = &frames[frame_i + 1];
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

fn build_headers(
    channels: &[ChannelSpec],
    variants_by_channel: &BTreeMap<String, Vec<ChannelVariant>>,
    size_z: usize,
    metric_options: Option<&MeasurementMetricOptions>,
    has_manual_background: bool,
) -> Vec<String> {
    let mut headers = CORE_MEASUREMENT_HEADERS
        .iter()
        .map(|header| header.to_string())
        .collect::<Vec<_>>();

    headers.extend(
        SIZE_METRIC_HEADERS
            .iter()
            .filter(|header| {
                metric_name_allowed(
                    metric_options.and_then(|opts| opts.size_metrics.as_ref()),
                    header,
                )
            })
            .map(|header| header.to_string()),
    );
    if include_zslice_size_headers(metric_options, size_z) {
        for z in 0..size_z {
            headers.push(format!("cell_area_pxl_zslice{z}"));
            headers.push(format!("cell_area_um2_zslice{z}"));
        }
    }

    for channel in channels {
        let variants = variants_by_channel
            .get(&channel.name)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for variant in variants {
            for suffix in CHANNEL_METRIC_SUFFIXES {
                if channel_metric_allowed(metric_options, &channel.name, suffix, *variant) {
                    headers.push(metric_column_name(&channel.name, suffix, *variant));
                }
            }
            if has_manual_background {
                for suffix in MANUAL_BACKGROUND_METRIC_SUFFIXES {
                    if channel_metric_allowed(metric_options, &channel.name, suffix, *variant) {
                        headers.push(metric_column_name(&channel.name, suffix, *variant));
                    }
                }
            }
        }
    }
    headers.extend(
        REGIONPROP_HEADERS
            .iter()
            .filter(|header| {
                regionprop_name_allowed(
                    metric_options.and_then(|opts| opts.regionprops.as_ref()),
                    header,
                )
            })
            .map(|header| header.to_string()),
    );
    headers
}

fn include_zslice_size_headers(
    metric_options: Option<&MeasurementMetricOptions>,
    size_z: usize,
) -> bool {
    let Some(metric_options) = metric_options else {
        return false;
    };
    if !metric_options.calc_size_for_each_zslice || size_z <= 1 {
        return false;
    }
    metric_options
        .size_metrics
        .as_ref()
        .map(|metrics| !metrics.is_empty())
        .unwrap_or(true)
}

fn metric_name_allowed(selected: Option<&Vec<String>>, header: &str) -> bool {
    let Some(selected) = selected else {
        return true;
    };
    selected
        .iter()
        .any(|metric| metric.trim().eq_ignore_ascii_case(header))
}

fn regionprop_name_allowed(selected: Option<&Vec<String>>, header: &str) -> bool {
    let Some(selected) = selected else {
        return true;
    };
    let header_lower = header.to_ascii_lowercase();
    selected.iter().any(|metric| {
        let metric_lower = metric.trim().to_ascii_lowercase();
        metric_lower == header_lower
            || header_lower
                .strip_prefix(&metric_lower)
                .is_some_and(|suffix| suffix.starts_with('-'))
    })
}

fn channel_metric_allowed(
    metric_options: Option<&MeasurementMetricOptions>,
    channel_name: &str,
    suffix: &str,
    variant: ChannelVariant,
) -> bool {
    let Some(metric_options) = metric_options else {
        return true;
    };
    let channel_key = normalize_metric_key(channel_name);
    if metric_options
        .channel_metrics_to_skip
        .get(&channel_key)
        .is_some_and(|metrics| channel_metric_list_matches(metrics, channel_name, suffix, variant))
    {
        return false;
    }
    let Some(channel_metrics) = &metric_options.channel_metrics else {
        return true;
    };
    channel_metrics
        .get(&channel_key)
        .is_some_and(|metrics| channel_metric_list_matches(metrics, channel_name, suffix, variant))
}

fn channel_metric_list_matches(
    metrics: &[String],
    channel_name: &str,
    suffix: &str,
    variant: ChannelVariant,
) -> bool {
    let header = metric_column_name(channel_name, suffix, variant);
    let variant_suffix = variant.column_suffix();
    let suffix_with_variant = format!("{suffix}{variant_suffix}");
    metrics.iter().any(|metric| {
        let metric = metric.trim();
        metric.eq_ignore_ascii_case(suffix)
            || metric.eq_ignore_ascii_case(&suffix_with_variant)
            || metric.eq_ignore_ascii_case(&header)
    })
}

pub(crate) fn normalize_metric_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
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
                "z_slice_used" => row
                    .z_slice_used
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NaN".to_string()),
                "which_z_proj" => row
                    .which_z_proj
                    .clone()
                    .unwrap_or_else(|| "NaN".to_string()),
                "Cell_ID" => row.cell_id.to_string(),
                "cell_cycle_stage" => row.cell_cycle_stage.clone(),
                "generation_num" => row.generation_num.to_string(),
                "relative_ID" => row.relative_id.to_string(),
                "relationship" => row.relationship.clone(),
                "emerg_frame_i" => row.emerg_frame_i.to_string(),
                "division_frame_i" => row.division_frame_i.to_string(),
                "is_history_known" => row.is_history_known.to_string(),
                "corrected_on_frame_i" => row.corrected_on_frame_i.to_string(),
                "will_divide" => row.will_divide.to_string(),
                "daughter_disappears_before_division" => {
                    row.daughter_disappears_before_division.to_string()
                }
                "disappears_before_division" => row.disappears_before_division.to_string(),
                "is_cell_dead" => row.is_cell_dead.to_string(),
                "is_cell_excluded" => row.is_cell_excluded.to_string(),
                "was_manually_edited" => row.was_manually_edited.to_string(),
                "x_centroid" => row.x_centroid.to_string(),
                "y_centroid" => row.y_centroid.to_string(),
                "cell_area_pxl" => row.cell_area_pxl.to_string(),
                "cell_area_um2" => format_f64(row.cell_area_um2),
                "cell_vol_vox" => format_f64(row.cell_vol_vox),
                "cell_vol_fl" => format_f64(row.cell_vol_fl),
                "cell_vol_vox_3D" => format_f64(row.cell_vol_vox_3d),
                "cell_vol_fl_3D" => format_f64(row.cell_vol_fl_3d),
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

fn write_object_counts_csv(path: &Path, masks: &MaskData) -> Result<()> {
    let counts = objects_count_summary(masks);
    let mut table = Table::new(counts.keys().cloned().collect());
    table.push_row(
        counts
            .values()
            .map(|value| TableValue::Number(*value as f64))
            .collect(),
    )?;
    write_table(path, &table)
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
    use crate::tabular::read_table;
    use crate::utilities::{add_lineage_tree, LineageTreeConfig};
    use ndarray::Array3;
    use ndarray::Array4;
    use ndarray_npy::{write_npy, NpzWriter};
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

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
                1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
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
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(!headers
            .iter()
            .any(|header| header == "demo_acdc_output.csv"));
        for required in [
            "cell_vol_vox",
            "phase_mean",
            "gfp_mean",
            "cell_cycle_stage",
            "generation_num",
            "relative_ID",
            "relationship",
            "emerg_frame_i",
            "division_frame_i",
            "is_history_known",
            "corrected_on_frame_i",
            "will_divide",
            "daughter_disappears_before_division",
            "disappears_before_division",
        ] {
            assert!(
                headers.iter().any(|header| header == required),
                "missing required header {required}"
            );
        }
        let first_row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");
        let get = |name: &str| -> Result<&str> {
            let idx = headers
                .iter()
                .position(|header| header == name)
                .ok_or_else(|| anyhow::anyhow!("missing header {name}"))?;
            Ok(first_row.get(idx).expect("value"))
        };
        assert_eq!(get("cell_cycle_stage")?, "G1");
        assert_eq!(get("generation_num")?, "2");
        assert_eq!(get("relative_ID")?, "-1");
        assert_eq!(get("relationship")?, "mother");
        assert_eq!(get("is_history_known")?, "false");
        assert_eq!(get("corrected_on_frame_i")?, "-1");
        Ok(())
    }

    #[test]
    fn measures_plain_npy_channels_from_existing_position() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_npy_stack(&images.join("demo_phase.npy"), &[10, 20])?;
        write_test_npy_stack(&images.join("demo_gfp.npy"), &[30, 40])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            2,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;

        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let rows = reader.records().collect::<csv::Result<Vec<_>>>()?;
        assert_eq!(rows.len(), 2);
        assert!(headers.iter().any(|header| header == "phase_mean"));
        assert!(headers.iter().any(|header| header == "gfp_mean"));
        assert_eq!(csv_f64(&headers, &rows[0], "phase_mean")?, 10.0);
        assert_eq!(csv_f64(&headers, &rows[0], "gfp_mean")?, 30.0);
        assert_eq!(csv_f64(&headers, &rows[1], "phase_mean")?, 20.0);
        assert_eq!(csv_f64(&headers, &rows[1], "gfp_mean")?, 40.0);
        Ok(())
    }

    #[test]
    fn normalizes_python_segmentation_end_filename() -> Result<()> {
        let images = Path::new("/tmp/images");
        let from_full_name = measurement_output_paths(images, "demo_", Some("segm_rust.npz"));
        assert_eq!(
            from_full_name.segm_npz_path,
            images.join("demo_segm_rust.npz")
        );
        assert_eq!(
            from_full_name.acdc_output_csv_path,
            images.join("demo_acdc_output_rust.csv")
        );
        assert_eq!(
            from_full_name.objects_count_csv_path,
            images.join("demo_acdc_objects_count_rust.csv")
        );

        let from_suffix = measurement_output_paths(images, "demo_", Some("rust"));
        assert_eq!(from_suffix, from_full_name);
        Ok(())
    }

    #[test]
    fn measures_legacy_position_token_segmentation_without_copying() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_s01_phase.tif"), &[10])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_s01_segm.npz"),
            &[
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: Some(vec!["phase".to_string()]),
            metric_options: None,
            save_object_counts_table: false,
        })?;

        assert_eq!(
            result.outputs.segm_npz_path,
            images.join("demo_s01_segm.npz")
        );
        assert_eq!(
            result.outputs.acdc_output_csv_path,
            images.join("demo_acdc_output.csv")
        );
        assert!(result.outputs.acdc_output_csv_path.exists());
        assert!(!images.join("demo_segm.npz").exists());
        Ok(())
    }

    #[test]
    fn measures_old_python_npy_segmentation_fallback() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 12])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        let labels = Array3::from_shape_vec(
            (2, 4, 4),
            vec![
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
        )?;
        write_npy(images.join("demo_segm.npy"), &labels)?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: Some(vec!["phase".to_string()]),
            metric_options: None,
            save_object_counts_table: false,
        })?;

        assert_eq!(result.outputs.segm_npz_path, images.join("demo_segm.npy"));
        assert!(result.outputs.acdc_output_csv_path.exists());
        assert!(!images.join("demo_segm.npz").exists());
        Ok(())
    }

    #[test]
    fn measures_only_selected_channels_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10])?;
        write_test_stack(&images.join("demo_gfp.tif"), &[30])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
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

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: Some(vec!["phase".to_string()]),
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean"));
        assert!(!headers.iter().any(|header| header == "gfp_mean"));
        Ok(())
    }

    #[test]
    fn saves_object_counts_table_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10, 20])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 2, 2, //
                0, 0, 2, 2, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            2,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: true,
        })?;
        let table = read_table(&result.outputs.objects_count_csv_path)?;
        assert!(result.outputs.objects_count_csv_path.exists());
        assert_eq!(
            table
                .headers
                .iter()
                .position(|header| header == "In entire video")
                .and_then(|idx| table.rows[0][idx].as_i64()),
            Some(2)
        );
        assert_eq!(
            table
                .headers
                .iter()
                .position(|header| header == "Unique objects in entire video")
                .and_then(|idx| table.rows[0][idx].as_i64()),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn writes_manual_background_metrics_when_mask_exists() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack_pixels(
            &images.join("demo_phase.tif"),
            &[
                10, 14, 0, 0, //
                0, 0, 0, 0, //
                2, 4, 0, 0, //
                0, 0, 0, 0, //
            ],
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;
        write_mask_npz(
            &images.join("demo_manualBackground.npz"),
            &[
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: Some(vec!["phase".to_string()]),
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let row = reader.records().next().transpose()?.expect("row");
        assert_eq!(
            csv_f64(&headers, &row, "phase_manualBkgr_bkgrVal_mean")?,
            3.0
        );
        assert_eq!(csv_f64(&headers, &row, "phase_mean_manualBkgr")?, 9.0);
        assert_eq!(csv_f64(&headers, &row, "phase_amount_manualBkgr")?, 18.0);
        Ok(())
    }

    #[test]
    fn filters_manual_background_metrics_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack_pixels(
            &images.join("demo_phase.tif"),
            &[
                10, 14, 0, 0, //
                0, 0, 0, 0, //
                2, 4, 0, 0, //
                0, 0, 0, 0, //
            ],
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;
        write_mask_npz(
            &images.join("demo_manualBackground.npz"),
            &[
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
            ],
            1,
            4,
            4,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: Some(vec!["phase".to_string()]),
            metric_options: Some(MeasurementMetricOptions {
                channel_metrics: Some(BTreeMap::from([(
                    "phase".to_string(),
                    vec!["mean_manualBkgr".to_string()],
                )])),
                channel_metrics_to_skip: BTreeMap::new(),
                calc_for_each_zslice_channels: BTreeMap::new(),
                calc_size_for_each_zslice: false,
                size_metrics: Some(Vec::new()),
                regionprops: Some(Vec::new()),
            }),
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers
            .iter()
            .any(|header| header == "phase_mean_manualBkgr"));
        assert!(!headers.iter().any(|header| header == "phase_mean"));
        assert!(!headers
            .iter()
            .any(|header| header == "phase_manualBkgr_bkgrVal_mean"));
        Ok(())
    }

    #[test]
    fn writes_only_selected_measurement_metrics_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10])?;
        write_test_stack(&images.join("demo_gfp.tif"), &[30])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
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

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: Some(MeasurementMetricOptions {
                channel_metrics: Some(BTreeMap::from([(
                    "phase".to_string(),
                    vec!["mean".to_string()],
                )])),
                channel_metrics_to_skip: BTreeMap::new(),
                calc_for_each_zslice_channels: BTreeMap::new(),
                calc_size_for_each_zslice: false,
                size_metrics: Some(vec!["cell_area_pxl".to_string()]),
                regionprops: Some(vec!["centroid".to_string()]),
            }),
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean"));
        assert!(!headers.iter().any(|header| header == "phase_sum"));
        assert!(!headers.iter().any(|header| header == "gfp_mean"));
        assert!(headers.iter().any(|header| header == "cell_area_pxl"));
        assert!(!headers.iter().any(|header| header == "cell_vol_vox"));
        assert!(headers.iter().any(|header| header == "centroid-0"));
        assert!(headers.iter().any(|header| header == "centroid-1"));
        assert!(!headers.iter().any(|header| header == "area"));
        Ok(())
    }

    #[test]
    fn measured_output_can_flow_into_lineage_tree_utility() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
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

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let lineage_output = images.join("demo_acdc_output_lineage.csv");
        add_lineage_tree(LineageTreeConfig {
            input_path: result.outputs.acdc_output_csv_path,
            output_path: lineage_output.clone(),
        })?;
        let table = read_table(&lineage_output)?;
        for required in [
            "Cell_ID_tree",
            "generation_num_tree",
            "parent_ID_tree",
            "root_ID_tree",
            "sister_ID_tree",
        ] {
            assert!(
                table.headers.iter().any(|header| header == required),
                "missing lineage header {required}"
            );
        }
        Ok(())
    }

    #[test]
    fn loads_data_prep_background_archive() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[10])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\n",
        )?;
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
        let loaded = load_channels(&spec, 4, 4, false)?;
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

    #[test]
    fn emits_zstack_projection_variant_columns_for_2d_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(
            &images.join("demo_phase_aligned.npz"),
            &[1.0, 5.0, 0.0, 10.0, 3.0, 7.0, 2.0, 12.0],
            1,
            2,
            2,
            2,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\nPhysicalSizeX,1\nPhysicalSizeY,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, //
                0, 0, //
            ],
            1,
            2,
            2,
        )?;
        write_test_segm_info(
            &images.join("demo_segmInfo.csv"),
            "demo_phase_aligned.npz",
            0,
            0,
        )?;
        fs::write(
            images.join("demo_dataPrep_bkgrROIs.json"),
            r#"[{"pos":[0,1],"size":[2,1]}]"#,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean_maxProj"));
        assert!(headers.iter().any(|header| header == "phase_mean_meanProj"));
        assert!(headers.iter().any(|header| header == "phase_mean_zSlice"));
        assert!(headers
            .iter()
            .any(|header| header == "phase_autoBkgr_bkgrVal_median_maxProj"));
        assert!(headers
            .iter()
            .any(|header| header == "phase_dataPrepBkgr_bkgrVal_median_zSlice"));
        assert!(!headers.iter().any(|header| header == "phase_mean"));

        let row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");
        assert_eq!(csv_f64(&headers, &row, "phase_mean_maxProj")?, 5.0);
        assert_eq!(csv_f64(&headers, &row, "phase_mean_meanProj")?, 4.0);
        assert_eq!(csv_f64(&headers, &row, "phase_mean_zSlice")?, 3.0);
        assert_eq!(
            csv_f64(&headers, &row, "phase_autoBkgr_bkgrVal_median_maxProj")?,
            7.0
        );
        assert_eq!(
            csv_f64(&headers, &row, "phase_dataPrepBkgr_bkgrVal_median_meanProj")?,
            6.0
        );
        Ok(())
    }

    #[test]
    fn measures_zstack_with_stale_segm_info_z_slice_like_python() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(
            &images.join("demo_phase_aligned.npz"),
            &[1.0, 5.0, 0.0, 10.0, 3.0, 7.0, 2.0, 12.0],
            1,
            2,
            2,
            2,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\nPhysicalSizeX,1\nPhysicalSizeY,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, //
                0, 0, //
            ],
            1,
            2,
            2,
        )?;
        write_test_segm_info(
            &images.join("demo_segmInfo.csv"),
            "demo_phase_aligned.npz",
            0,
            9,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");

        assert_eq!(csv_f64(&headers, &row, "phase_mean_zSlice")?, 5.0);
        assert_eq!(csv_f64(&headers, &row, "z_slice_used")?, 1.0);
        Ok(())
    }

    #[test]
    fn emits_each_zslice_columns_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(
            &images.join("demo_phase_aligned.npz"),
            &[1.0, 5.0, 0.0, 10.0, 3.0, 7.0, 2.0, 12.0],
            1,
            2,
            2,
            2,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\nPhysicalSizeX,1\nPhysicalSizeY,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, //
                0, 0, //
            ],
            1,
            2,
            2,
        )?;
        write_test_segm_info(
            &images.join("demo_segmInfo.csv"),
            "demo_phase_aligned.npz",
            0,
            0,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: Some(MeasurementMetricOptions {
                channel_metrics: Some(BTreeMap::from([(
                    "phase".to_string(),
                    vec!["mean".to_string()],
                )])),
                channel_metrics_to_skip: BTreeMap::new(),
                calc_for_each_zslice_channels: BTreeMap::from([("phase".to_string(), true)]),
                calc_size_for_each_zslice: false,
                size_metrics: None,
                regionprops: None,
            }),
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean_zSlice"));
        assert!(headers.iter().any(|header| header == "phase_mean_zSlice0"));
        assert!(headers.iter().any(|header| header == "phase_mean_zSlice1"));

        let row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");
        assert_eq!(csv_f64(&headers, &row, "phase_mean_zSlice")?, 3.0);
        assert_eq!(csv_f64(&headers, &row, "phase_mean_zSlice0")?, 3.0);
        assert_eq!(csv_f64(&headers, &row, "phase_mean_zSlice1")?, 5.0);
        Ok(())
    }

    #[test]
    fn emits_zstack_3d_variant_columns_for_3d_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(
            &images.join("demo_phase_aligned.npz"),
            &[1.0, 5.0, 0.0, 10.0, 3.0, 7.0, 2.0, 12.0],
            1,
            2,
            2,
            2,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\nsegm_isSegm3D,True\nPhysicalSizeZ,1\nPhysicalSizeX,1\nPhysicalSizeY,1\n",
        )?;
        write_test_mask_volume_npz(
            &images.join("demo_segm.npz"),
            &[1, 1, 0, 0, 1, 0, 0, 0],
            1,
            2,
            2,
            2,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: None,
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean_maxProj"));
        assert!(headers.iter().any(|header| header == "phase_mean_meanProj"));
        assert!(headers.iter().any(|header| header == "phase_mean_zSlice"));
        assert!(headers.iter().any(|header| header == "phase_mean_3D"));
        assert!(headers
            .iter()
            .any(|header| header == "phase_amount_autoBkgr_3D"));
        assert!(headers
            .iter()
            .any(|header| header == "phase_concentration_autoBkgr_from_vol_vox_3D"));
        assert!(!headers.iter().any(|header| header == "phase_mean"));

        let row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");
        assert_eq!(csv_f64(&headers, &row, "phase_mean_maxProj")?, 5.0);
        assert_eq!(csv_f64(&headers, &row, "phase_mean_3D")?, 3.0);
        assert_eq!(csv_f64(&headers, &row, "cell_vol_vox_3D")?, 3.0);
        assert_eq!(
            csv_f64(
                &headers,
                &row,
                "phase_concentration_autoBkgr_from_vol_vox_3D"
            )?,
            -4.0
        );
        Ok(())
    }

    #[test]
    fn emits_zslice_size_columns_when_configured() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        write_test_volume_npz(
            &images.join("demo_phase_aligned.npz"),
            &[1.0, 5.0, 0.0, 10.0, 3.0, 7.0, 2.0, 12.0],
            1,
            2,
            2,
            2,
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\nsegm_isSegm3D,True\nPhysicalSizeZ,1\nPhysicalSizeX,2\nPhysicalSizeY,3\n",
        )?;
        write_test_mask_volume_npz(
            &images.join("demo_segm.npz"),
            &[1, 1, 0, 0, 1, 0, 0, 0],
            1,
            2,
            2,
            2,
        )?;

        let result = measure_position(MeasurementRunConfig {
            position_path: temp.path().join("Position_1"),
            segm_endname: None,
            overwrite_policy: OverwritePolicy::Overwrite,
            stop_frame: None,
            channel_names: None,
            metric_options: Some(MeasurementMetricOptions {
                channel_metrics: None,
                channel_metrics_to_skip: BTreeMap::new(),
                calc_for_each_zslice_channels: BTreeMap::new(),
                calc_size_for_each_zslice: true,
                size_metrics: Some(vec!["cell_area_pxl".to_string()]),
                regionprops: None,
            }),
            save_object_counts_table: false,
        })?;
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "cell_area_pxl"));
        assert!(!headers.iter().any(|header| header == "cell_area_um2"));
        assert!(headers
            .iter()
            .any(|header| header == "cell_area_pxl_zslice0"));
        assert!(headers
            .iter()
            .any(|header| header == "cell_area_um2_zslice0"));
        assert!(headers
            .iter()
            .any(|header| header == "cell_area_pxl_zslice1"));
        assert!(headers
            .iter()
            .any(|header| header == "cell_area_um2_zslice1"));

        let row = reader
            .records()
            .next()
            .transpose()?
            .expect("first output row");
        assert_eq!(csv_f64(&headers, &row, "cell_area_pxl_zslice0")?, 2.0);
        assert_eq!(csv_f64(&headers, &row, "cell_area_um2_zslice0")?, 12.0);
        assert_eq!(csv_f64(&headers, &row, "cell_area_pxl_zslice1")?, 1.0);
        assert_eq!(csv_f64(&headers, &row, "cell_area_um2_zslice1")?, 6.0);
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

    fn write_test_stack_pixels(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        encoder.write_image::<colortype::Gray16>(4, 4, frame_values)?;
        Ok(())
    }

    fn write_test_npy_stack(path: &Path, frame_values: &[u16]) -> Result<()> {
        let mut values = Vec::with_capacity(frame_values.len() * 16);
        for value in frame_values {
            values.extend(std::iter::repeat(*value).take(16));
        }
        let array = Array3::from_shape_vec((frame_values.len(), 4, 4), values)?;
        write_npy(path, &array)?;
        Ok(())
    }

    fn write_test_volume_npz(
        path: &Path,
        values: &[f32],
        size_t: usize,
        size_z: usize,
        height: usize,
        width: usize,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = NpzWriter::new(file);
        let array = Array4::from_shape_vec((size_t, size_z, height, width), values.to_vec())?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;
        Ok(())
    }

    fn write_test_mask_volume_npz(
        path: &Path,
        values: &[u32],
        size_t: usize,
        size_z: usize,
        height: usize,
        width: usize,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = NpzWriter::new(file);
        let array = Array4::from_shape_vec((size_t, size_z, height, width), values.to_vec())?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;
        Ok(())
    }

    fn write_test_segm_info(
        path: &Path,
        filename: &str,
        frame_i: usize,
        z_slice: usize,
    ) -> Result<()> {
        fs::write(
            path,
            format!(
                "filename,frame_i,z_slice_used_dataPrep,which_z_proj,is_from_dataPrep,z_slice_used_gui,which_z_proj_gui,resegmented_in_gui\n{filename},{frame_i},{z_slice},single z-slice,1,{z_slice},single z-slice,0\n"
            ),
        )?;
        Ok(())
    }

    fn csv_f64(headers: &[String], row: &csv::StringRecord, name: &str) -> Result<f64> {
        let idx = headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| anyhow::anyhow!("missing header {name}"))?;
        row.get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing value for {name}"))?
            .parse::<f64>()
            .with_context(|| format!("failed to parse {name} as f64"))
    }
}
