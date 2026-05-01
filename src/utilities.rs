use crate::build_lineage_state;
use crate::layout::{discover_measurement_experiment, resolve_measurement_position};
use crate::mask_io::{
    load_mask_data, save_mask_data, MaskData, MaskPathResolution, SegmentationLayout,
};
use crate::tabular::{read_table, write_table, Table, TableFormat, TableValue};
use crate::zstack::{connect_3d_lab_z_boundaries, stack_2d_lab_to_3d};
use anyhow::{anyhow, bail, Context, Result};
use evalexpr::{ContextWithMutableVariables, HashMapContext, Value as EvalValue};
use hdf5_reader::Hdf5File;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::{read_npy, write_npy, NpzReader, NpzWriter};
use roxmltree::Document;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tiff::encoder::{colortype, TiffEncoder};

const DEFAULT_INDEX_COLS: &[&str] = &[
    "experiment_folderpath",
    "experiment_foldername",
    "Position_n",
    "frame_i",
    "Cell_ID",
];
const REQUIRED_LINEAGE_COLS: &[&str] = &[
    "frame_i",
    "Cell_ID",
    "cell_cycle_stage",
    "generation_num",
    "relative_ID",
    "relationship",
    "is_history_known",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityOutputPaths {
    pub primary_path: PathBuf,
    pub secondary_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatConfig {
    pub experiment_dirs: Vec<PathBuf>,
    pub table_endname: String,
    pub output_format: TableFormat,
    pub selected_columns: Option<Vec<String>>,
    pub output_name: Option<String>,
    pub multi_experiment_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatResult {
    pub all_position_outputs: Vec<PathBuf>,
    pub all_position_count_outputs: Vec<PathBuf>,
    pub multi_experiment_output: Option<PathBuf>,
    pub multi_experiment_count_output: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombineMetricsConfig {
    pub source_paths: Vec<PathBuf>,
    pub formulas: BTreeMap<String, String>,
    pub output_path: PathBuf,
    pub equations_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombineMetricsResult {
    pub output_path: PathBuf,
    pub equations_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeMultiChannelConfig {
    pub position_dir: Option<PathBuf>,
    pub experiment_dir: Option<PathBuf>,
    pub source_endnames: Vec<String>,
    pub formulas: BTreeMap<String, String>,
    pub append_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeMultiChannelResult {
    pub outputs: Vec<CombineMetricsResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombineChannelsConfig {
    pub position_dir: Option<PathBuf>,
    pub experiment_dir: Option<PathBuf>,
    pub recipe_path: PathBuf,
    pub append_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombineChannelsResult {
    pub output_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountObjectsConfig {
    pub segmentation_path: PathBuf,
    pub output_path: PathBuf,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCoordinatesConfig {
    pub segmentation_path: PathBuf,
    pub output_path: PathBuf,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectsCountSummary {
    pub counts: BTreeMap<String, usize>,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountObjectsResult {
    pub summary: ObjectsCountSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillHolesConfig {
    pub segmentation_path: PathBuf,
    pub output_path: PathBuf,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connect3DSegmConfig {
    pub segmentation_path: PathBuf,
    pub output_path: PathBuf,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stack2DSegmTo3DConfig {
    pub segmentation_path: PathBuf,
    pub output_path: PathBuf,
    pub size_z: usize,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinateFilterConfig {
    pub segmentation_path: PathBuf,
    pub coords_table_path: PathBuf,
    pub output_path: PathBuf,
    pub x_col: String,
    pub y_col: String,
    pub z_col: Option<String>,
    pub frame_col: Option<String>,
    pub position_col: Option<String>,
    pub position_value: Option<String>,
    pub resolution: Option<MaskPathResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackingColumnMap {
    pub frame_index_col: String,
    pub is_first_frame_one: bool,
    pub track_ids_col: String,
    pub mask_ids_col: Option<String>,
    pub x_centroid_col: Option<String>,
    pub y_centroid_col: Option<String>,
    pub z_centroid_col: Option<String>,
    pub delete_untracked_ids: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyTrackingConfig {
    pub segmentation_path: PathBuf,
    pub tracking_table_path: PathBuf,
    pub output_path: PathBuf,
    pub columns: TrackingColumnMap,
    pub resolution: Option<MaskPathResolution>,
    pub source_acdc_output_path: Option<PathBuf>,
    pub output_acdc_output_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyTrackingFromTrackMateXmlConfig {
    pub position_dir: PathBuf,
    pub segm_endname: String,
    pub xml_path: PathBuf,
    pub output_segmentation_path: Option<PathBuf>,
    pub source_acdc_output_path: Option<PathBuf>,
    pub output_acdc_output_path: Option<PathBuf>,
    pub delete_untracked_ids: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageTreeConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageTreeBatchConfig {
    pub position_dir: Option<PathBuf>,
    pub experiment_dir: Option<PathBuf>,
    pub table_endname: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateMotherBudTotalConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub column_operation_mapper: BTreeMap<String, String>,
    pub copy_all_nonselected_columns: bool,
    pub grouping_columns: Vec<String>,
    pub entity_colname: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertFileFormatConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub cast_segm_uint32: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameFilesConfig {
    pub file_paths: Vec<PathBuf>,
    pub append_text: String,
}

pub fn concat_acdc_outputs(config: ConcatConfig) -> Result<ConcatResult> {
    if config.experiment_dirs.is_empty() {
        bail!("concat_acdc_outputs requires at least one experiment directory");
    }

    let mut all_position_outputs = Vec::new();
    let mut all_position_count_outputs = Vec::new();
    let mut multi_tables = Vec::new();
    let mut multi_count_tables = Vec::new();
    let output_ext = table_extension(config.output_format);

    for experiment_dir in &config.experiment_dirs {
        let experiment_name = experiment_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("experiment")
            .to_string();
        let mut position_rows = Vec::new();
        let mut position_count_rows = Vec::new();
        let mut ordered_headers = Vec::new();
        let mut ordered_count_headers = Vec::new();

        for position_dir in list_position_dirs(experiment_dir)? {
            let images_dir =
                if position_dir.file_name().and_then(|value| value.to_str()) == Some("Images") {
                    position_dir.clone()
                } else {
                    position_dir.join("Images")
                };
            let table_path = find_table_by_endname(&images_dir, &config.table_endname)?;
            let Some(table_path) = table_path else {
                continue;
            };
            let table = read_table(&table_path)?;
            for row in table_to_rows(&table) {
                let mut row = row;
                row.insert(
                    "experiment_folderpath".into(),
                    TableValue::Text(experiment_dir.display().to_string()),
                );
                row.insert(
                    "experiment_foldername".into(),
                    TableValue::Text(experiment_name.clone()),
                );
                row.insert(
                    "Position_n".into(),
                    TableValue::Text(
                        position_dir
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or_default()
                            .to_string(),
                    ),
                );
                extend_header_order(&mut ordered_headers, row.keys().cloned());
                position_rows.push(row);
            }

            let count_endname = config
                .table_endname
                .replace("acdc_output", "acdc_objects_count");
            if let Some(count_path) = find_table_by_endname(&images_dir, &count_endname)? {
                let count_table = read_table(&count_path)?;
                for row in table_to_rows(&count_table) {
                    let mut row = row;
                    row.insert(
                        "experiment_folderpath".into(),
                        TableValue::Text(experiment_dir.display().to_string()),
                    );
                    row.insert(
                        "experiment_foldername".into(),
                        TableValue::Text(experiment_name.clone()),
                    );
                    row.insert(
                        "Position_n".into(),
                        TableValue::Text(
                            position_dir
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or_default()
                                .to_string(),
                        ),
                    );
                    extend_header_order(&mut ordered_count_headers, row.keys().cloned());
                    position_count_rows.push(row);
                }
            }
        }

        let selected_headers =
            selected_or_all_headers(config.selected_columns.as_ref(), &ordered_headers);
        let allpos_dir = experiment_dir.join("AllPos_acdc_output");
        fs::create_dir_all(&allpos_dir)?;
        let output_name = config
            .output_name
            .clone()
            .unwrap_or_else(|| format!("AllPos_{}.{}", config.table_endname, output_ext));
        let output_path = allpos_dir.join(output_name);
        let table = rows_to_table(&selected_headers, &position_rows);
        write_table(&output_path, &table)?;
        all_position_outputs.push(output_path.clone());
        multi_tables.extend(table_to_rows(&table));

        if !position_count_rows.is_empty() {
            let count_filename = replace_output_name_metric(
                output_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default(),
                "acdc_output",
                "acdc_objects_count",
            );
            let count_path = allpos_dir.join(count_filename);
            let count_headers = selected_or_all_headers(None, &ordered_count_headers);
            let count_table = rows_to_table(&count_headers, &position_count_rows);
            write_table(&count_path, &count_table)?;
            all_position_count_outputs.push(count_path.clone());
            multi_count_tables.extend(table_to_rows(&count_table));
        }
    }

    let multi_experiment_output = if config.experiment_dirs.len() > 1 {
        let target_dir = config
            .multi_experiment_dir
            .clone()
            .unwrap_or_else(|| config.experiment_dirs[0].join("AllPos_acdc_output"));
        fs::create_dir_all(&target_dir)?;
        let filename = config
            .output_name
            .clone()
            .map(|name| format!("multiExp_{name}"))
            .unwrap_or_else(|| format!("multiExp_{}.{}", config.table_endname, output_ext));
        let path = target_dir.join(filename);
        let headers = infer_headers_from_rows(&multi_tables);
        write_table(&path, &rows_to_table(&headers, &multi_tables))?;
        Some(path)
    } else {
        None
    };

    let multi_experiment_count_output =
        if config.experiment_dirs.len() > 1 && !multi_count_tables.is_empty() {
            let target_dir = config
                .multi_experiment_dir
                .clone()
                .unwrap_or_else(|| config.experiment_dirs[0].join("AllPos_acdc_output"));
            let filename = format!(
                "multiExp_{}.{}",
                config
                    .table_endname
                    .replace("acdc_output", "acdc_objects_count"),
                output_ext
            );
            let path = target_dir.join(filename);
            let headers = infer_headers_from_rows(&multi_count_tables);
            write_table(&path, &rows_to_table(&headers, &multi_count_tables))?;
            Some(path)
        } else {
            None
        };

    Ok(ConcatResult {
        all_position_outputs,
        all_position_count_outputs,
        multi_experiment_output,
        multi_experiment_count_output,
    })
}

pub fn combine_metrics(config: CombineMetricsConfig) -> Result<CombineMetricsResult> {
    if config.source_paths.len() < 2 {
        bail!("combine_metrics requires at least two source tables");
    }
    if config.formulas.is_empty() {
        bail!("combine_metrics requires at least one formula");
    }
    let tables = config
        .source_paths
        .iter()
        .map(|path| read_table(path))
        .collect::<Result<Vec<_>>>()?;

    let key_columns = shared_key_columns(&tables);
    let key_columns = if key_columns.is_empty() {
        vec!["__row_index".to_string()]
    } else {
        key_columns
    };
    let source_rows = tables.iter().map(table_to_rows).collect::<Vec<_>>();
    let key_sets = source_rows
        .iter()
        .enumerate()
        .flat_map(|(table_idx, rows)| {
            let key_columns = key_columns.clone();
            rows.iter()
                .enumerate()
                .map(move |(row_idx, row)| build_key(row, &key_columns, table_idx, row_idx))
        })
        .collect::<BTreeSet<_>>();

    let mut output_rows = Vec::new();
    for key in key_sets {
        let mut context = HashMapContext::new();
        let mut output_row: Row = BTreeMap::new();
        for (name, value) in key_columns.iter().zip(key.0.iter()) {
            output_row.insert(name.clone(), TableValue::Text(value.clone()));
        }

        for (table_idx, rows) in source_rows.iter().enumerate() {
            let row = rows
                .iter()
                .enumerate()
                .find(|(row_idx, row)| build_key(row, &key_columns, table_idx, *row_idx) == key)
                .map(|(_, row)| row);
            let headers = &tables[table_idx].headers;
            for header in headers {
                if key_columns.iter().any(|col| col == header) {
                    continue;
                }
                let value = row
                    .and_then(|row| row.get(header))
                    .and_then(TableValue::as_f64)
                    .unwrap_or(f64::NAN);
                for alias in metric_aliases(table_idx + 1, header) {
                    context.set_value(alias, EvalValue::Float(value))?;
                }
            }
        }

        for (column, expression) in &config.formulas {
            let value = evalexpr::eval_number_with_context(expression, &context)
                .or_else(|_| evalexpr::eval_float_with_context(expression, &context))
                .unwrap_or(f64::NAN);
            output_row.insert(column.clone(), TableValue::Number(value));
        }
        output_rows.push(output_row);
    }

    let mut headers = key_columns.clone();
    headers.extend(config.formulas.keys().cloned());
    write_table(&config.output_path, &rows_to_table(&headers, &output_rows))?;

    let equations_path = config.equations_path.clone().unwrap_or_else(|| {
        config
            .output_path
            .with_file_name(replace_file_stem_suffix(&config.output_path, "equations"))
    });
    write_equations_ini(&equations_path, &config.source_paths, &config.formulas)?;

    Ok(CombineMetricsResult {
        output_path: config.output_path,
        equations_path,
    })
}

pub fn compute_multi_channel(
    config: ComputeMultiChannelConfig,
) -> Result<ComputeMultiChannelResult> {
    if config.source_endnames.len() < 2 {
        bail!("compute_multi_channel requires at least two --source-endname values");
    }
    if config.formulas.is_empty() {
        bail!("compute_multi_channel requires at least one --formula value");
    }
    let images_dirs = collect_images_dirs_from_scope(
        config.position_dir.as_deref(),
        config.experiment_dir.as_deref(),
    )?;

    let mut outputs = Vec::new();
    for images_dir in images_dirs {
        let mut source_paths = Vec::with_capacity(config.source_endnames.len());
        for endname in &config.source_endnames {
            let path = find_table_by_endname(&images_dir, endname)?.ok_or_else(|| {
                anyhow!(
                    "No table ending with {:?} found in {}",
                    endname,
                    images_dir.display()
                )
            })?;
            source_paths.push(path);
        }

        let basename = infer_table_basename(&source_paths[0], &config.source_endnames[0])?;
        let output_path =
            images_dir.join(format!("{basename}acdc_output_{}.csv", config.append_name));
        let equations_path =
            images_dir.join(format!("{basename}equations_{}.ini", config.append_name));
        outputs.push(combine_metrics(CombineMetricsConfig {
            source_paths,
            formulas: config.formulas.clone(),
            output_path,
            equations_path: Some(equations_path),
        })?);
    }

    Ok(ComputeMultiChannelResult { outputs })
}

pub fn count_objects(config: CountObjectsConfig) -> Result<CountObjectsResult> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    let counts = objects_count_summary(&masks);
    let mut table = Table::new(counts.keys().cloned().collect());
    table.push_row(
        counts
            .values()
            .map(|value| TableValue::Number(*value as f64))
            .collect(),
    )?;
    write_table(&config.output_path, &table)?;
    Ok(CountObjectsResult {
        summary: ObjectsCountSummary {
            counts,
            output_path: config.output_path,
        },
    })
}

pub fn segmentation_to_object_coords(
    config: ObjectCoordinatesConfig,
) -> Result<UtilityOutputPaths> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    let table = object_coordinates_table(&masks)?;
    write_table(&config.output_path, &table)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn fill_holes(config: FillHolesConfig) -> Result<UtilityOutputPaths> {
    let mut masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    fill_holes_in_mask_data(&mut masks)?;
    save_mask_data(&config.output_path, &masks)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn connect_3d_segm(config: Connect3DSegmConfig) -> Result<UtilityOutputPaths> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    let connected = match masks.layout {
        SegmentationLayout::ZYX => {
            let shape = masks.values.shape();
            let values = masks.values.iter().copied().collect::<Vec<_>>();
            let connected = connect_3d_lab_z_boundaries(&values, shape[0], shape[1], shape[2]);
            MaskData {
                values: ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), connected)?,
                layout: masks.layout,
                source_path: masks.source_path.clone(),
            }
        }
        SegmentationLayout::TZYX => {
            let shape = masks.values.shape();
            let frame_len = shape[1] * shape[2] * shape[3];
            let mut connected = Vec::with_capacity(masks.values.len());
            let values = masks.values.iter().copied().collect::<Vec<_>>();
            for frame_i in 0..shape[0] {
                let start = frame_i * frame_len;
                connected.extend(connect_3d_lab_z_boundaries(
                    &values[start..start + frame_len],
                    shape[1],
                    shape[2],
                    shape[3],
                ));
            }
            MaskData {
                values: ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), connected)?,
                layout: masks.layout,
                source_path: masks.source_path.clone(),
            }
        }
        other => bail!(
            "connect_3d_segm requires ZYX or TZYX masks, got {:?}",
            other
        ),
    };
    save_mask_data(&config.output_path, &connected)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn stack_2d_segm_to_3d(config: Stack2DSegmTo3DConfig) -> Result<UtilityOutputPaths> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    if config.size_z == 0 {
        bail!("stack_2d_segm_to_3d requires size_z > 0");
    }
    let stacked = match masks.layout {
        SegmentationLayout::YX => {
            let shape = masks.values.shape();
            let values = masks.values.iter().copied().collect::<Vec<_>>();
            let stacked = stack_2d_lab_to_3d(&values, config.size_z);
            MaskData {
                values: ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[config.size_z, shape[0], shape[1]]),
                    stacked,
                )?,
                layout: SegmentationLayout::ZYX,
                source_path: masks.source_path.clone(),
            }
        }
        SegmentationLayout::TYX => {
            let shape = masks.values.shape();
            let plane_len = shape[1] * shape[2];
            let mut stacked = Vec::with_capacity(masks.values.len() * config.size_z);
            let values = masks.values.iter().copied().collect::<Vec<_>>();
            for frame_i in 0..shape[0] {
                let start = frame_i * plane_len;
                stacked.extend(stack_2d_lab_to_3d(
                    &values[start..start + plane_len],
                    config.size_z,
                ));
            }
            MaskData {
                values: ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[shape[0], config.size_z, shape[1], shape[2]]),
                    stacked,
                )?,
                layout: SegmentationLayout::TZYX,
                source_path: masks.source_path.clone(),
            }
        }
        other => bail!(
            "stack_2d_segm_to_3d requires YX or TYX masks, got {:?}",
            other
        ),
    };
    save_mask_data(&config.output_path, &stacked)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn filter_segm_from_table(config: CoordinateFilterConfig) -> Result<UtilityOutputPaths> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    let coords_table = read_table(&config.coords_table_path)?;
    let mut filtered = masks.clone();
    filter_mask_data(
        &mut filtered,
        &coords_table,
        &config.x_col,
        &config.y_col,
        config.z_col.as_deref(),
        config.frame_col.as_deref(),
        config.position_col.as_deref(),
        config.position_value.as_deref(),
    )?;
    save_mask_data(&config.output_path, &filtered)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn apply_tracking_from_table(config: ApplyTrackingConfig) -> Result<UtilityOutputPaths> {
    let masks = load_mask_data(&config.segmentation_path, config.resolution.as_ref())?;
    if !matches!(
        masks.layout,
        SegmentationLayout::TYX | SegmentationLayout::TZYX
    ) {
        bail!(
            "apply_tracking_from_table requires a segmentation mask with a time axis, got {:?}",
            masks.layout
        );
    }
    let tracking_table = read_table(&config.tracking_table_path)?;
    apply_tracking_with_loaded_table(
        masks,
        tracking_table,
        config.output_path,
        &config.columns,
        config.source_acdc_output_path.as_deref(),
        config.output_acdc_output_path.as_deref(),
    )
}

pub fn combine_channels(config: CombineChannelsConfig) -> Result<CombineChannelsResult> {
    let recipe = load_combine_channels_recipe(&config.recipe_path)?;
    let positions = collect_measurement_positions_from_scope(
        config.position_dir.as_deref(),
        config.experiment_dir.as_deref(),
    )?;
    let mut output_paths = Vec::with_capacity(positions.len());

    for position in positions {
        let output = combine_channels_for_position(&position, &recipe, &config.append_name)?;
        output_paths.push(output);
    }

    Ok(CombineChannelsResult { output_paths })
}

pub fn apply_tracking_from_trackmate_xml(
    config: ApplyTrackingFromTrackMateXmlConfig,
) -> Result<UtilityOutputPaths> {
    let images_dir = normalize_images_dir(&config.position_dir)?;
    let segmentation_path = find_file_by_endname(
        &images_dir,
        &config.segm_endname,
        &["npz", "tif", "tiff", "h5"],
    )?
    .ok_or_else(|| {
        anyhow!(
            "No segmentation ending with {:?} found in {}",
            config.segm_endname,
            images_dir.display()
        )
    })?;
    let masks = load_mask_data(&segmentation_path, None).or_else(|err| {
        if err.to_string().contains("Ambiguous 3D segmentation layout") {
            load_mask_data(
                &segmentation_path,
                Some(&MaskPathResolution {
                    size_t: None,
                    size_z: Some(1),
                    layout: Some(SegmentationLayout::TYX),
                }),
            )
        } else {
            Err(err)
        }
    })?;
    let tracking_table = trackmate_xml_to_table(&config.xml_path)?;
    let output_path = config
        .output_segmentation_path
        .clone()
        .unwrap_or_else(|| append_to_file_stem(&segmentation_path, "_tracked"));
    let source_acdc_output_path = config
        .source_acdc_output_path
        .clone()
        .or_else(|| infer_tracking_source_acdc_output(&images_dir, &config.segm_endname));

    apply_tracking_with_loaded_table(
        masks,
        tracking_table,
        output_path,
        &TrackingColumnMap {
            frame_index_col: "frame_i".into(),
            is_first_frame_one: false,
            track_ids_col: "ID".into(),
            mask_ids_col: None,
            x_centroid_col: Some("x".into()),
            y_centroid_col: Some("y".into()),
            z_centroid_col: Some("z".into()),
            delete_untracked_ids: config.delete_untracked_ids,
        },
        source_acdc_output_path.as_deref(),
        config.output_acdc_output_path.as_deref(),
    )
}

pub fn add_lineage_tree(config: LineageTreeConfig) -> Result<UtilityOutputPaths> {
    let table = read_table(&config.input_path)?;
    ensure_required_columns(&table, REQUIRED_LINEAGE_COLS)?;
    let state = build_lineage_state(&table)?;
    write_table(&config.output_path, &state.to_table())?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn add_lineage_tree_to_tables(config: LineageTreeBatchConfig) -> Result<UtilityOutputPaths> {
    let images_dirs = collect_images_dirs_from_scope(
        config.position_dir.as_deref(),
        config.experiment_dir.as_deref(),
    )?;
    let mut output_paths = Vec::new();
    for images_dir in images_dirs {
        let Some(table_path) = find_table_by_endname(&images_dir, &config.table_endname)? else {
            continue;
        };
        add_lineage_tree(LineageTreeConfig {
            input_path: table_path.clone(),
            output_path: table_path.clone(),
        })?;
        output_paths.push(table_path);
    }
    if output_paths.is_empty() {
        bail!(
            "No tables ending with {}.csv or {}.xlsx were found in the selected scope",
            config.table_endname,
            config.table_endname
        );
    }
    Ok(UtilityOutputPaths {
        primary_path: output_paths[0].clone(),
        secondary_paths: output_paths.into_iter().skip(1).collect(),
    })
}

pub fn generate_mother_bud_total(
    config: GenerateMotherBudTotalConfig,
) -> Result<UtilityOutputPaths> {
    let table = read_table(&config.input_path)?;
    ensure_required_columns(
        &table,
        &[
            "frame_i",
            "Cell_ID",
            "cell_cycle_stage",
            "relationship",
            "relative_ID",
        ],
    )?;
    let rows = table_to_rows(&table);
    let output_rows = generate_mother_bud_total_rows(
        &rows,
        &config.column_operation_mapper,
        config.copy_all_nonselected_columns,
        &config.grouping_columns,
        &config.entity_colname,
    )?;
    let headers = infer_headers_from_rows(&output_rows);
    write_table(&config.output_path, &rows_to_table(&headers, &output_rows))?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn convert_file_format(config: ConvertFileFormatConfig) -> Result<UtilityOutputPaths> {
    let mut array = load_convertible_array(&config.input_path)?;
    if config.cast_segm_uint32 {
        array.scalar_type = ImageScalarType::U32;
        array.values.mapv_inplace(|value| value.max(0.0).round());
    }
    save_convertible_array(&config.output_path, &array)?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
    })
}

pub fn rename_files(config: RenameFilesConfig) -> Result<UtilityOutputPaths> {
    if config.file_paths.is_empty() {
        bail!("rename_files requires at least one file path");
    }
    let append_text = config.append_text.trim();
    if append_text.is_empty() {
        bail!("rename_files requires non-empty append text");
    }

    let mut renamed_paths = Vec::new();
    for path in config.file_paths {
        if !path.is_file() {
            bail!("Cannot rename missing file {}", path.display());
        }
        let target_path = append_text_to_filename(&path, append_text)?;
        if target_path.exists() {
            bail!(
                "Cannot rename {} because target already exists: {}",
                path.display(),
                target_path.display()
            );
        }
        fs::rename(&path, &target_path).with_context(|| {
            format!(
                "Failed to rename {} to {}",
                path.display(),
                target_path.display()
            )
        })?;
        renamed_paths.push(target_path);
    }

    let primary_path = renamed_paths
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("rename_files produced no output paths"))?;
    Ok(UtilityOutputPaths {
        primary_path,
        secondary_paths: renamed_paths.into_iter().skip(1).collect(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayLayout {
    YX,
    TYX,
    ZYX,
    TZYX,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageScalarType {
    U8,
    U16,
    U32,
    F32,
}

#[derive(Debug, Clone)]
struct LoadedChannelArray {
    values: ArrayD<f32>,
    layout: ArrayLayout,
    scalar_type: ImageScalarType,
}

#[derive(Debug, Clone)]
struct ConvertibleArray {
    values: ArrayD<f32>,
    scalar_type: ImageScalarType,
}

#[derive(Debug, Clone)]
struct CombineChannelStep {
    key: String,
    name: String,
    channel: String,
    binarize: String,
    min_val: f32,
    max_val: f32,
}

#[derive(Debug, Clone)]
struct CombineChannelsRecipe {
    steps: Vec<CombineChannelStep>,
    formula: String,
    keep_input_data_type: bool,
    save_as_segm: bool,
}

fn apply_tracking_with_loaded_table(
    masks: MaskData,
    tracking_table: Table,
    output_path: PathBuf,
    columns: &TrackingColumnMap,
    source_acdc_output_path: Option<&Path>,
    output_acdc_output_path: Option<&Path>,
) -> Result<UtilityOutputPaths> {
    let mut tracked = masks.clone();
    let (tracked_ids_mapper, deleted_ids_mapper) =
        apply_tracking_to_mask_data(&mut tracked, &tracking_table, columns)?;
    save_mask_data(&output_path, &tracked)?;

    let mapper_base = output_path.with_extension("");
    let deleted_path = mapper_base.with_file_name(format!(
        "{}_deletedIDs_mapper.json",
        mapper_base
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("segm")
    ));
    let replaced_path = mapper_base.with_file_name(format!(
        "{}_replacedIDs_mapper.json",
        mapper_base
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("segm")
    ));
    fs::write(
        &deleted_path,
        serde_json::to_vec_pretty(&deleted_ids_mapper)?,
    )?;
    fs::write(
        &replaced_path,
        serde_json::to_vec_pretty(&tracked_ids_mapper)?,
    )?;

    let mut secondary_paths = vec![deleted_path, replaced_path];
    if let Some(source_path) = source_acdc_output_path {
        if source_path.exists() {
            let table = read_table(source_path)?;
            let remapped = apply_tracked_ids_mapper_to_table(
                &table,
                &tracked_ids_mapper,
                &deleted_ids_mapper,
            )?;
            let acdc_output_path = output_acdc_output_path
                .map(Path::to_path_buf)
                .unwrap_or_else(|| derive_acdc_output_path(source_path, &output_path));
            write_table(&acdc_output_path, &remapped)?;
            secondary_paths.push(acdc_output_path);
        }
    }

    Ok(UtilityOutputPaths {
        primary_path: output_path,
        secondary_paths,
    })
}

fn combine_channels_for_position(
    position: &crate::layout::MeasurementPositionSpec,
    recipe: &CombineChannelsRecipe,
    append_name: &str,
) -> Result<PathBuf> {
    let mut loaded_steps = Vec::with_capacity(recipe.steps.len());
    let mut output_layout = None;
    let mut output_shape = None;
    let mut original_scalar_type = None;

    for step in &recipe.steps {
        if step.channel == "current segm." {
            bail!("combine_channels does not support the GUI-only channel \"current segm.\"");
        }
        let loaded = load_recipe_channel(position, &step.channel)?;
        if original_scalar_type.is_none() {
            original_scalar_type = Some(loaded.scalar_type);
        }
        output_layout = Some(match output_layout {
            Some(layout) => merge_layout(layout, loaded.layout)?,
            None => loaded.layout,
        });
        output_shape = Some(match output_shape {
            Some(shape) => merge_layout_shape(shape, loaded.values.shape())?,
            None => loaded.values.shape().to_vec(),
        });
        loaded_steps.push((step.clone(), loaded));
    }

    let output_layout = output_layout.ok_or_else(|| anyhow!("combine_channels recipe is empty"))?;
    let output_shape = output_shape.expect("shape set with non-empty recipe");
    let mut variables = BTreeMap::new();
    let mut first_output = None;

    for (step, loaded) in loaded_steps {
        let mut values = align_channel_to_target_layout(
            loaded.values,
            loaded.layout,
            output_layout,
            &output_shape,
        )?;
        apply_binarize(&mut values, &step.binarize)?;
        if !(step.min_val == 0.0 && step.max_val == 1.0) {
            values = rescale_array(&values, step.min_val, step.max_val);
        }
        if first_output.is_none() {
            first_output = Some(values.clone());
        }
        variables.insert(step.name.clone(), values);
    }

    let mut output = if recipe.formula.trim().is_empty() {
        first_output.ok_or_else(|| anyhow!("combine_channels recipe does not contain any steps"))?
    } else {
        let expression = parse_array_expression(&recipe.formula)?;
        expression.evaluate(&variables)?
    };

    if !recipe.save_as_segm {
        output = rescale_array(&output, 0.0, 1.0);
    }

    let output_path = if recipe.save_as_segm {
        position
            .images_dir
            .join(format!("{}segm_{}.npz", position.basename, append_name))
    } else {
        position
            .images_dir
            .join(format!("{}{}.tif", position.basename, append_name))
    };

    if recipe.save_as_segm {
        let values = output
            .mapv(|value| if value < 0.0 { 0 } else { value as u32 })
            .into_dyn();
        let layout = segmentation_layout_from_array_layout(output_layout);
        save_mask_data(
            &output_path,
            &MaskData {
                values,
                layout,
                source_path: output_path.clone(),
            },
        )?;
    } else {
        let scalar_type = if recipe.keep_input_data_type {
            original_scalar_type.unwrap_or(ImageScalarType::F32)
        } else {
            ImageScalarType::F32
        };
        save_image_tiff(&output_path, &output, scalar_type)?;
    }

    Ok(output_path)
}

fn load_combine_channels_recipe(path: &Path) -> Result<CombineChannelsRecipe> {
    let raw: serde_json::Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?,
    )
    .with_context(|| format!("Failed to parse JSON recipe {}", path.display()))?;
    let object = raw
        .as_object()
        .ok_or_else(|| anyhow!("combine_channels recipe must be a JSON object"))?;

    let mut step_keys = object
        .keys()
        .filter(|key| key.chars().all(|ch| ch.is_ascii_digit()))
        .cloned()
        .collect::<Vec<_>>();
    step_keys.sort_by_key(|key| key.parse::<usize>().unwrap_or(usize::MAX));
    if step_keys.is_empty() {
        bail!("combine_channels recipe does not contain any numeric step entries");
    }

    let mut steps = Vec::with_capacity(step_keys.len());
    for key in step_keys {
        let step = object
            .get(&key)
            .and_then(|value| value.as_object())
            .ok_or_else(|| anyhow!("Recipe step {key:?} must be a JSON object"))?;
        steps.push(CombineChannelStep {
            key,
            name: json_required_string(step, "name")?.to_string(),
            channel: json_required_string(step, "channel")?.to_string(),
            binarize: json_required_string(step, "binarize")?.to_string(),
            min_val: json_required_f32(step, "min_val")?,
            max_val: json_required_f32(step, "max_val")?,
        });
    }

    Ok(CombineChannelsRecipe {
        steps,
        formula: object
            .get("formula")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        keep_input_data_type: object
            .get("keep_input_data_type")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        save_as_segm: object
            .get("save_as_segm")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    })
}

fn json_required_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str> {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("Recipe field {key:?} must be a string"))
}

fn json_required_f32(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<f32> {
    object
        .get(key)
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .ok_or_else(|| anyhow!("Recipe field {key:?} must be numeric"))
}

fn load_recipe_channel(
    position: &crate::layout::MeasurementPositionSpec,
    channel_name: &str,
) -> Result<LoadedChannelArray> {
    if let Some(channel) = position
        .channels
        .iter()
        .find(|channel| channel.name == channel_name)
    {
        let (pixels, shape) = crate::image_io::load_image_volume_as_f32(
            &channel.image_path,
            Some(position.size_t),
            Some(position.size_z),
        )
        .with_context(|| {
            format!(
                "Failed to load channel {:?} from {}",
                channel_name,
                channel.image_path.display()
            )
        })?;
        let scalar_type = detect_image_scalar_type(&channel.image_path)?;
        let values = normalize_image_array(shape_volume_pixels(pixels, shape)?, scalar_type)?;
        let layout = volume_shape_layout(shape);
        return Ok(LoadedChannelArray {
            values,
            layout,
            scalar_type,
        });
    }

    let path = find_file_by_endname(
        &position.images_dir,
        channel_name,
        &["npz", "tif", "tiff", "h5"],
    )?
    .ok_or_else(|| {
        anyhow!(
            "No raw or segmentation channel ending with {:?} found in {}",
            channel_name,
            position.images_dir.display()
        )
    })?;
    let masks = load_mask_data(&path, None)
        .with_context(|| format!("Failed to load segmentation {}", path.display()))?;
    Ok(LoadedChannelArray {
        values: masks.values.mapv(|value| value as f32),
        layout: array_layout_from_segm_layout(masks.layout),
        scalar_type: ImageScalarType::U32,
    })
}

fn normalize_image_array(
    mut values: ArrayD<f32>,
    scalar_type: ImageScalarType,
) -> Result<ArrayD<f32>> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    match scalar_type {
        ImageScalarType::U8 => values.mapv_inplace(|value| value / u8::MAX as f32),
        ImageScalarType::U16 => values.mapv_inplace(|value| value / u16::MAX as f32),
        ImageScalarType::U32 => values.mapv_inplace(|value| value / u32::MAX as f32),
        ImageScalarType::F32 => {
            if max.is_finite() && max > 1.0 {
                let divisor = if max <= u8::MAX as f32 {
                    u8::MAX as f32
                } else if max <= u16::MAX as f32 {
                    u16::MAX as f32
                } else if max <= u32::MAX as f32 {
                    u32::MAX as f32
                } else {
                    bail!("Float image contains values above the supported 32-bit range");
                };
                values.mapv_inplace(|value| value / divisor);
            }
        }
    }
    Ok(values)
}

fn shape_volume_pixels(
    pixels: Vec<f32>,
    shape: crate::image_io::VolumeShape,
) -> Result<ArrayD<f32>> {
    let dims = if shape.size_t > 1 && shape.size_z > 1 {
        vec![shape.size_t, shape.size_z, shape.height, shape.width]
    } else if shape.size_t > 1 {
        vec![shape.size_t, shape.height, shape.width]
    } else if shape.size_z > 1 {
        vec![shape.size_z, shape.height, shape.width]
    } else {
        vec![shape.height, shape.width]
    };
    ArrayD::from_shape_vec(IxDyn(&dims), pixels)
        .with_context(|| format!("Failed to shape image array with dims {:?}", dims))
}

fn volume_shape_layout(shape: crate::image_io::VolumeShape) -> ArrayLayout {
    match (shape.size_t > 1, shape.size_z > 1) {
        (false, false) => ArrayLayout::YX,
        (true, false) => ArrayLayout::TYX,
        (false, true) => ArrayLayout::ZYX,
        (true, true) => ArrayLayout::TZYX,
    }
}

fn array_layout_from_segm_layout(layout: SegmentationLayout) -> ArrayLayout {
    match layout {
        SegmentationLayout::YX => ArrayLayout::YX,
        SegmentationLayout::TYX => ArrayLayout::TYX,
        SegmentationLayout::ZYX => ArrayLayout::ZYX,
        SegmentationLayout::TZYX => ArrayLayout::TZYX,
    }
}

fn segmentation_layout_from_array_layout(layout: ArrayLayout) -> SegmentationLayout {
    match layout {
        ArrayLayout::YX => SegmentationLayout::YX,
        ArrayLayout::TYX => SegmentationLayout::TYX,
        ArrayLayout::ZYX => SegmentationLayout::ZYX,
        ArrayLayout::TZYX => SegmentationLayout::TZYX,
    }
}

fn merge_layout(left: ArrayLayout, right: ArrayLayout) -> Result<ArrayLayout> {
    let has_t = layout_has_time(left) || layout_has_time(right);
    let has_z = layout_has_depth(left) || layout_has_depth(right);
    Ok(match (has_t, has_z) {
        (false, false) => ArrayLayout::YX,
        (true, false) => ArrayLayout::TYX,
        (false, true) => ArrayLayout::ZYX,
        (true, true) => ArrayLayout::TZYX,
    })
}

fn merge_layout_shape(left: Vec<usize>, right: &[usize]) -> Result<Vec<usize>> {
    if left.len() != right.len() {
        return Ok(if left.len() > right.len() {
            left
        } else {
            right.to_vec()
        });
    }
    Ok(left
        .into_iter()
        .zip(right.iter().copied())
        .map(|(l, r)| l.max(r))
        .collect())
}

fn layout_has_time(layout: ArrayLayout) -> bool {
    matches!(layout, ArrayLayout::TYX | ArrayLayout::TZYX)
}

fn layout_has_depth(layout: ArrayLayout) -> bool {
    matches!(layout, ArrayLayout::ZYX | ArrayLayout::TZYX)
}

fn align_channel_to_target_layout(
    values: ArrayD<f32>,
    source_layout: ArrayLayout,
    target_layout: ArrayLayout,
    target_shape: &[usize],
) -> Result<ArrayD<f32>> {
    if source_layout == target_layout {
        if values.shape() != target_shape {
            bail!(
                "Shape mismatch for combine_channels: got {:?}, expected {:?}",
                values.shape(),
                target_shape
            );
        }
        return Ok(values);
    }
    match (source_layout, target_layout) {
        (ArrayLayout::YX, ArrayLayout::ZYX) => {
            let depth = *target_shape
                .first()
                .ok_or_else(|| anyhow!("Missing target depth for ZYX broadcast"))?;
            let plane = values.into_dimensionality::<ndarray::Ix2>()?;
            let mut out = Vec::with_capacity(depth * plane.len());
            let base = plane.iter().copied().collect::<Vec<_>>();
            for _ in 0..depth {
                out.extend(base.iter().copied());
            }
            ArrayD::from_shape_vec(IxDyn(target_shape), out)
                .context("Failed to broadcast 2D segmentation across Z")
        }
        (ArrayLayout::TYX, ArrayLayout::TZYX) => {
            let depth = target_shape
                .get(1)
                .copied()
                .ok_or_else(|| anyhow!("Missing target Z axis for TZYX broadcast"))?;
            let array = values.into_dimensionality::<ndarray::Ix3>()?;
            let shape = array.shape().to_vec();
            let mut out = Vec::with_capacity(shape[0] * depth * shape[1] * shape[2]);
            for frame in array.outer_iter() {
                let base = frame.iter().copied().collect::<Vec<_>>();
                for _ in 0..depth {
                    out.extend(base.iter().copied());
                }
            }
            ArrayD::from_shape_vec(IxDyn(target_shape), out)
                .context("Failed to broadcast 2Dt segmentation across Z")
        }
        _ => bail!(
            "Unsupported combine_channels layout conversion from {:?} to {:?}",
            source_layout,
            target_layout
        ),
    }
}

fn apply_binarize(values: &mut ArrayD<f32>, binarize: &str) -> Result<()> {
    match binarize {
        "No" => {}
        "binarize" => values.mapv_inplace(|value| if value > 0.0 { 1.0 } else { 0.0 }),
        "inverse binarize" => values.mapv_inplace(|value| if value > 0.0 { 0.0 } else { 1.0 }),
        other => bail!("Unsupported combine_channels binarize mode {:?}", other),
    }
    Ok(())
}

fn rescale_array(values: &ArrayD<f32>, out_min: f32, out_max: f32) -> ArrayD<f32> {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() || (max - min).abs() <= f32::EPSILON {
        return values.mapv(|_| out_min);
    }
    let scale = (out_max - out_min) / (max - min);
    values.mapv(|value| (value - min) * scale + out_min)
}

fn load_convertible_array(path: &Path) -> Result<ConvertibleArray> {
    match path_extension(path).as_deref() {
        Some("npz") => load_convertible_npz(path),
        Some("npy") => load_convertible_npy(path),
        Some("tif") | Some("tiff") => load_convertible_tiff(path),
        Some("h5") => load_convertible_h5(path),
        other => bail!(
            "Unsupported input format {:?} for {}. Supported formats are npz, npy, tif, tiff, and h5.",
            other,
            path.display()
        ),
    }
}

fn save_convertible_array(path: &Path, array: &ConvertibleArray) -> Result<()> {
    ensure_output_parent(path)?;
    match path_extension(path).as_deref() {
        Some("npz") => save_convertible_npz(path, array),
        Some("npy") => save_convertible_npy(path, array),
        Some("tif") | Some("tiff") => save_convertible_tiff(path, array),
        other => bail!(
            "Unsupported output format {:?} for {}. Supported formats are npz, npy, tif, and tiff.",
            other,
            path.display()
        ),
    }
}

fn load_convertible_npz(path: &Path) -> Result<ConvertibleArray> {
    macro_rules! try_npz {
        ($ty:ty, $scalar_type:expr) => {{
            let file = File::open(path)
                .with_context(|| format!("Failed to open NPZ {}", path.display()))?;
            let mut reader = NpzReader::new(file)
                .with_context(|| format!("Failed to read NPZ {}", path.display()))?;
            if let Ok(values) = reader.by_name::<ndarray::OwnedRepr<$ty>, IxDyn>("arr_0") {
                return Ok(ConvertibleArray {
                    values: values.mapv(|value| value as f32),
                    scalar_type: $scalar_type,
                });
            }
        }};
    }

    try_npz!(f32, ImageScalarType::F32);
    try_npz!(f64, ImageScalarType::F32);
    try_npz!(u8, ImageScalarType::U8);
    try_npz!(u16, ImageScalarType::U16);
    try_npz!(u32, ImageScalarType::U32);
    try_npz!(u64, ImageScalarType::F32);
    try_npz!(i8, ImageScalarType::F32);
    try_npz!(i16, ImageScalarType::F32);
    try_npz!(i32, ImageScalarType::F32);
    try_npz!(i64, ImageScalarType::F32);
    bail!(
        "Unsupported NPZ arr_0 element type in {}. The Cell-ACDC converter expects an arr_0 array.",
        path.display()
    )
}

fn load_convertible_npy(path: &Path) -> Result<ConvertibleArray> {
    macro_rules! try_npy {
        ($ty:ty, $scalar_type:expr) => {
            if let Ok(values) = read_npy::<_, ArrayD<$ty>>(path) {
                return Ok(ConvertibleArray {
                    values: values.mapv(|value| value as f32),
                    scalar_type: $scalar_type,
                });
            }
        };
    }

    try_npy!(f32, ImageScalarType::F32);
    try_npy!(f64, ImageScalarType::F32);
    try_npy!(u8, ImageScalarType::U8);
    try_npy!(u16, ImageScalarType::U16);
    try_npy!(u32, ImageScalarType::U32);
    try_npy!(u64, ImageScalarType::F32);
    try_npy!(i8, ImageScalarType::F32);
    try_npy!(i16, ImageScalarType::F32);
    try_npy!(i32, ImageScalarType::F32);
    try_npy!(i64, ImageScalarType::F32);
    bail!("Unsupported NPY element type in {}", path.display())
}

fn load_convertible_h5(path: &Path) -> Result<ConvertibleArray> {
    let file =
        Hdf5File::open(path).with_context(|| format!("Failed to open H5 {}", path.display()))?;
    let dataset = file
        .dataset("/data")
        .or_else(|_| file.dataset("data"))
        .with_context(|| format!("Failed to open dataset \"data\" in {}", path.display()))?;
    let shape = dataset
        .shape()
        .iter()
        .map(|dim| *dim as usize)
        .collect::<Vec<_>>();

    macro_rules! try_h5 {
        ($ty:ty, $scalar_type:expr) => {
            if let Ok(values) = dataset.read_array::<$ty>() {
                let values = values.into_iter().map(|value| value as f32).collect();
                return Ok(ConvertibleArray {
                    values: ArrayD::from_shape_vec(IxDyn(&shape), values).with_context(|| {
                        format!("Failed to shape H5 dataset from {}", path.display())
                    })?,
                    scalar_type: $scalar_type,
                });
            }
        };
    }

    try_h5!(f32, ImageScalarType::F32);
    try_h5!(f64, ImageScalarType::F32);
    try_h5!(u8, ImageScalarType::U8);
    try_h5!(u16, ImageScalarType::U16);
    try_h5!(u32, ImageScalarType::U32);
    try_h5!(u64, ImageScalarType::F32);
    try_h5!(i8, ImageScalarType::F32);
    try_h5!(i16, ImageScalarType::F32);
    try_h5!(i32, ImageScalarType::F32);
    try_h5!(i64, ImageScalarType::F32);
    bail!("Unsupported H5 dataset type in {}", path.display())
}

fn load_convertible_tiff(path: &Path) -> Result<ConvertibleArray> {
    let file =
        File::open(path).with_context(|| format!("Failed to open TIFF {}", path.display()))?;
    let mut decoder = tiff::decoder::Decoder::new(file)
        .with_context(|| format!("Failed to decode TIFF {}", path.display()))?;
    let mut pages = Vec::new();
    let mut expected_shape = None;
    let mut scalar_type = None;

    loop {
        let (width, height) = decoder
            .dimensions()
            .with_context(|| format!("Failed to read TIFF dimensions in {}", path.display()))?;
        let page_shape = (height as usize, width as usize);
        if let Some(expected) = expected_shape {
            if expected != page_shape {
                bail!(
                    "TIFF pages in {} do not share the same dimensions",
                    path.display()
                );
            }
        } else {
            expected_shape = Some(page_shape);
        }

        let result = decoder
            .read_image()
            .with_context(|| format!("Failed to read TIFF page in {}", path.display()))?;
        let page_scalar_type = tiff_scalar_type(&result);
        if let Some(expected) = scalar_type {
            if expected != page_scalar_type {
                scalar_type = Some(ImageScalarType::F32);
            }
        } else {
            scalar_type = Some(page_scalar_type);
        }
        pages.extend(tiff_result_to_f32(result));

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .with_context(|| format!("Failed to advance TIFF pages in {}", path.display()))?;
    }

    let (height, width) = expected_shape.unwrap_or((0, 0));
    let values = if pages.len() == height * width {
        ArrayD::from_shape_vec(IxDyn(&[height, width]), pages)
    } else {
        let frames = if height == 0 || width == 0 {
            0
        } else {
            pages.len() / (height * width)
        };
        ArrayD::from_shape_vec(IxDyn(&[frames, height, width]), pages)
    }
    .with_context(|| format!("Failed to shape TIFF data from {}", path.display()))?;

    Ok(ConvertibleArray {
        values,
        scalar_type: scalar_type.unwrap_or(ImageScalarType::F32),
    })
}

fn save_convertible_npz(path: &Path, array: &ConvertibleArray) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create NPZ {}", path.display()))?;
    let mut writer = NpzWriter::new_compressed(file);
    match array.scalar_type {
        ImageScalarType::U8 => writer.add_array("arr_0", &array.values.mapv(f32_to_u8_exact))?,
        ImageScalarType::U16 => writer.add_array("arr_0", &array.values.mapv(f32_to_u16_exact))?,
        ImageScalarType::U32 => writer.add_array("arr_0", &array.values.mapv(f32_to_u32_exact))?,
        ImageScalarType::F32 => writer.add_array("arr_0", &array.values)?,
    }
    writer
        .finish()
        .with_context(|| format!("Failed to finish NPZ {}", path.display()))?;
    Ok(())
}

fn save_convertible_npy(path: &Path, array: &ConvertibleArray) -> Result<()> {
    match array.scalar_type {
        ImageScalarType::U8 => write_npy(path, &array.values.mapv(f32_to_u8_exact))?,
        ImageScalarType::U16 => write_npy(path, &array.values.mapv(f32_to_u16_exact))?,
        ImageScalarType::U32 => write_npy(path, &array.values.mapv(f32_to_u32_exact))?,
        ImageScalarType::F32 => write_npy(path, &array.values)?,
    }
    Ok(())
}

fn save_convertible_tiff(path: &Path, array: &ConvertibleArray) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create TIFF {}", path.display()))?;
    let mut encoder = TiffEncoder::new(file)?;
    for plane in flatten_array_planes(&array.values)? {
        match array.scalar_type {
            ImageScalarType::U8 => {
                let pixels = plane
                    .pixels
                    .iter()
                    .copied()
                    .map(f32_to_u8_exact)
                    .collect::<Vec<_>>();
                encoder.write_image::<colortype::Gray8>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::U16 => {
                let pixels = plane
                    .pixels
                    .iter()
                    .copied()
                    .map(f32_to_u16_exact)
                    .collect::<Vec<_>>();
                encoder.write_image::<colortype::Gray16>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::U32 => {
                let pixels = plane
                    .pixels
                    .iter()
                    .copied()
                    .map(f32_to_u32_exact)
                    .collect::<Vec<_>>();
                encoder.write_image::<colortype::Gray32>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::F32 => {
                encoder.write_image::<colortype::Gray32Float>(
                    plane.width as u32,
                    plane.height as u32,
                    &plane.pixels,
                )?;
            }
        }
    }
    Ok(())
}

fn tiff_scalar_type(result: &tiff::decoder::DecodingResult) -> ImageScalarType {
    match result {
        tiff::decoder::DecodingResult::U8(_) => ImageScalarType::U8,
        tiff::decoder::DecodingResult::U16(_) => ImageScalarType::U16,
        tiff::decoder::DecodingResult::U32(_) => ImageScalarType::U32,
        _ => ImageScalarType::F32,
    }
}

fn tiff_result_to_f32(result: tiff::decoder::DecodingResult) -> Vec<f32> {
    match result {
        tiff::decoder::DecodingResult::U8(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::U16(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::U32(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::U64(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::I8(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::I16(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::I32(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::I64(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::F32(values) => values,
        tiff::decoder::DecodingResult::F64(values) => {
            values.into_iter().map(|value| value as f32).collect()
        }
        tiff::decoder::DecodingResult::F16(values) => {
            values.into_iter().map(|value| value.to_f32()).collect()
        }
    }
}

fn ensure_output_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn f32_to_u8_exact(value: f32) -> u8 {
    value.round().clamp(0.0, u8::MAX as f32) as u8
}

fn f32_to_u16_exact(value: f32) -> u16 {
    value.round().clamp(0.0, u16::MAX as f32) as u16
}

fn f32_to_u32_exact(value: f32) -> u32 {
    value.round().clamp(0.0, u32::MAX as f32) as u32
}

fn path_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
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

fn save_image_tiff(path: &Path, values: &ArrayD<f32>, scalar_type: ImageScalarType) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut encoder = TiffEncoder::new(file)?;
    for plane in flatten_array_planes(values)? {
        match scalar_type {
            ImageScalarType::U8 => {
                let pixels = convert_plane_to_u8(&plane.pixels);
                encoder.write_image::<colortype::Gray8>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::U16 => {
                let pixels = convert_plane_to_u16(&plane.pixels);
                encoder.write_image::<colortype::Gray16>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::U32 => {
                let pixels = convert_plane_to_u32(&plane.pixels);
                encoder.write_image::<colortype::Gray32>(
                    plane.width as u32,
                    plane.height as u32,
                    &pixels,
                )?;
            }
            ImageScalarType::F32 => {
                encoder.write_image::<colortype::Gray32Float>(
                    plane.width as u32,
                    plane.height as u32,
                    &plane.pixels,
                )?;
            }
        }
    }
    Ok(())
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
            let array = values.view().into_dimensionality::<ndarray::Ix2>()?;
            Ok(vec![PlaneF32 {
                height: array.shape()[0],
                width: array.shape()[1],
                pixels: array.iter().copied().collect(),
            }])
        }
        3 => {
            let array = values.view().into_dimensionality::<ndarray::Ix3>()?;
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
            let array = values.view().into_dimensionality::<ndarray::Ix4>()?;
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

fn convert_plane_to_integer(values: &[f32], dtype_max: f32) -> Vec<u32> {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let needs_scaling = min < 0.0 || max > dtype_max || (max <= 1.0 && dtype_max > 1.0);
    if !needs_scaling {
        return values
            .iter()
            .map(|value| value.round().clamp(0.0, dtype_max) as u32)
            .collect();
    }
    let scaled = if max <= 1.0 && min >= 0.0 {
        values
            .iter()
            .map(|value| (*value * dtype_max).round().clamp(0.0, dtype_max) as u32)
            .collect::<Vec<_>>()
    } else if (max - min).abs() <= f32::EPSILON {
        vec![0; values.len()]
    } else {
        values
            .iter()
            .map(|value| {
                (((*value - min) / (max - min)) * dtype_max)
                    .round()
                    .clamp(0.0, dtype_max) as u32
            })
            .collect::<Vec<_>>()
    };
    scaled
}

fn trackmate_xml_to_table(path: &Path) -> Result<Table> {
    let xml = fs::read_to_string(path)
        .with_context(|| format!("Failed to read TrackMate XML {}", path.display()))?;
    let doc = Document::parse(&xml)
        .with_context(|| format!("Failed to parse TrackMate XML {}", path.display()))?;
    let root = doc.root_element();
    let mut rows = Vec::new();
    for (particle_idx, particle) in root.children().filter(|node| node.is_element()).enumerate() {
        let id = (particle_idx + 1) as f64;
        for detection in particle.children().filter(|node| node.is_element()) {
            let frame_i = detection
                .attribute("t")
                .ok_or_else(|| anyhow!("TrackMate detection is missing attribute \"t\""))?
                .parse::<f64>()
                .with_context(|| format!("Invalid TrackMate frame value in {}", path.display()))?;
            let x = detection
                .attribute("x")
                .ok_or_else(|| anyhow!("TrackMate detection is missing attribute \"x\""))?
                .parse::<f64>()
                .with_context(|| format!("Invalid TrackMate x value in {}", path.display()))?;
            let y = detection
                .attribute("y")
                .ok_or_else(|| anyhow!("TrackMate detection is missing attribute \"y\""))?
                .parse::<f64>()
                .with_context(|| format!("Invalid TrackMate y value in {}", path.display()))?;
            let z = detection
                .attribute("z")
                .ok_or_else(|| anyhow!("TrackMate detection is missing attribute \"z\""))?
                .parse::<f64>()
                .with_context(|| format!("Invalid TrackMate z value in {}", path.display()))?;
            rows.push(row_from_pairs(vec![
                ("frame_i", TableValue::Number(frame_i)),
                ("ID", TableValue::Number(id)),
                ("x", TableValue::Number(x)),
                ("y", TableValue::Number(y)),
                ("z", TableValue::Number(z)),
            ]));
        }
    }
    Ok(rows_to_table(
        &[
            "frame_i".into(),
            "ID".into(),
            "x".into(),
            "y".into(),
            "z".into(),
        ],
        &rows,
    ))
}

fn write_equations_ini(
    path: &Path,
    sources: &[PathBuf],
    formulas: &BTreeMap<String, String>,
) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let mut lines = Vec::new();
    lines.push("[sources]".to_string());
    for (idx, source) in sources.iter().enumerate() {
        lines.push(format!("table{} = {}", idx + 1, source.display()));
    }
    lines.push(String::new());
    lines.push("[equations]".to_string());
    for (name, expression) in formulas {
        lines.push(format!("{name} = {expression}"));
    }
    fs::write(path, lines.join("\n"))?;
    Ok(())
}

pub(crate) fn objects_count_summary(masks: &MaskData) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    match masks.layout {
        SegmentationLayout::YX | SegmentationLayout::ZYX => {
            counts.insert(
                "In current position".into(),
                unique_nonzero(masks.values.iter().copied()).len(),
            );
        }
        SegmentationLayout::TYX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("valid TYX");
            let mut total = 0usize;
            let mut unique = BTreeSet::new();
            for frame in array.outer_iter() {
                let labels = unique_nonzero(frame.iter().copied());
                total += labels.len();
                unique.extend(labels);
            }
            counts.insert("In all visited frames".into(), total);
            counts.insert("In entire video".into(), total);
            counts.insert("Unique objects in all visited frames".into(), unique.len());
            counts.insert("Unique objects in entire video".into(), unique.len());
        }
        SegmentationLayout::TZYX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix4>()
                .expect("valid TZYX");
            let mut total = 0usize;
            let mut unique = BTreeSet::new();
            for frame in array.outer_iter() {
                let labels = unique_nonzero(frame.iter().copied());
                total += labels.len();
                unique.extend(labels);
            }
            counts.insert("In all visited frames".into(), total);
            counts.insert("In entire video".into(), total);
            counts.insert("Unique objects in all visited frames".into(), unique.len());
            counts.insert("Unique objects in entire video".into(), unique.len());
        }
    }
    counts
}

fn object_coordinates_table(masks: &MaskData) -> Result<Table> {
    match masks.layout {
        SegmentationLayout::YX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("valid YX");
            let mut coords_by_label = BTreeMap::<u32, Vec<(usize, usize)>>::new();
            for ((y, x), label) in array.indexed_iter() {
                if *label == 0 {
                    continue;
                }
                coords_by_label.entry(*label).or_default().push((y, x));
            }
            let mut rows = Vec::new();
            for (label, coords) in coords_by_label {
                for (y, x) in coords {
                    rows.push(vec![
                        TableValue::Number(0.0),
                        TableValue::Number(label as f64),
                        TableValue::Number(y as f64),
                        TableValue::Number(x as f64),
                    ]);
                }
            }
            Ok(Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "y".into(), "x".into()],
                rows,
            })
        }
        SegmentationLayout::TYX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("valid TYX");
            let mut coords_by_key = BTreeMap::<(usize, u32), Vec<(usize, usize)>>::new();
            for ((t, y, x), label) in array.indexed_iter() {
                if *label == 0 {
                    continue;
                }
                coords_by_key.entry((t, *label)).or_default().push((y, x));
            }
            let mut rows = Vec::new();
            for ((t, label), coords) in coords_by_key {
                for (y, x) in coords {
                    rows.push(vec![
                        TableValue::Number(t as f64),
                        TableValue::Number(label as f64),
                        TableValue::Number(y as f64),
                        TableValue::Number(x as f64),
                    ]);
                }
            }
            Ok(Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "y".into(), "x".into()],
                rows,
            })
        }
        SegmentationLayout::ZYX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("valid ZYX");
            let mut coords_by_label = BTreeMap::<u32, Vec<(usize, usize, usize)>>::new();
            for ((z, y, x), label) in array.indexed_iter() {
                if *label == 0 {
                    continue;
                }
                coords_by_label.entry(*label).or_default().push((z, y, x));
            }
            let mut rows = Vec::new();
            for (label, coords) in coords_by_label {
                for (z, y, x) in coords {
                    rows.push(vec![
                        TableValue::Number(0.0),
                        TableValue::Number(label as f64),
                        TableValue::Number(z as f64),
                        TableValue::Number(y as f64),
                        TableValue::Number(x as f64),
                    ]);
                }
            }
            Ok(Table {
                headers: vec![
                    "frame_i".into(),
                    "Cell_ID".into(),
                    "z".into(),
                    "y".into(),
                    "x".into(),
                ],
                rows,
            })
        }
        SegmentationLayout::TZYX => {
            let array = masks
                .values
                .view()
                .into_dimensionality::<ndarray::Ix4>()
                .expect("valid TZYX");
            let mut coords_by_key = BTreeMap::<(usize, u32), Vec<(usize, usize, usize)>>::new();
            for ((t, z, y, x), label) in array.indexed_iter() {
                if *label == 0 {
                    continue;
                }
                coords_by_key
                    .entry((t, *label))
                    .or_default()
                    .push((z, y, x));
            }
            let mut rows = Vec::new();
            for ((t, label), coords) in coords_by_key {
                for (z, y, x) in coords {
                    rows.push(vec![
                        TableValue::Number(t as f64),
                        TableValue::Number(label as f64),
                        TableValue::Number(z as f64),
                        TableValue::Number(y as f64),
                        TableValue::Number(x as f64),
                    ]);
                }
            }
            Ok(Table {
                headers: vec![
                    "frame_i".into(),
                    "Cell_ID".into(),
                    "z".into(),
                    "y".into(),
                    "x".into(),
                ],
                rows,
            })
        }
    }
}

fn fill_holes_in_mask_data(masks: &mut MaskData) -> Result<()> {
    match masks.layout {
        SegmentationLayout::YX => {
            let mut frame = masks.values.clone().into_dimensionality::<ndarray::Ix2>()?;
            let shape = frame.shape().to_vec();
            fill_holes_in_plane(
                frame.as_slice_mut().expect("contiguous"),
                shape[0],
                shape[1],
            );
            masks.values = frame.into_dyn();
        }
        SegmentationLayout::TYX | SegmentationLayout::ZYX => {
            let mut stack = masks.values.clone().into_dimensionality::<ndarray::Ix3>()?;
            for mut plane in stack.outer_iter_mut() {
                let shape = plane.shape().to_vec();
                fill_holes_in_plane(
                    plane.as_slice_mut().expect("contiguous"),
                    shape[0],
                    shape[1],
                );
            }
            masks.values = stack.into_dyn();
        }
        SegmentationLayout::TZYX => {
            let mut video = masks.values.clone().into_dimensionality::<ndarray::Ix4>()?;
            for mut stack in video.outer_iter_mut() {
                for mut plane in stack.outer_iter_mut() {
                    let shape = plane.shape().to_vec();
                    fill_holes_in_plane(
                        plane.as_slice_mut().expect("contiguous"),
                        shape[0],
                        shape[1],
                    );
                }
            }
            masks.values = video.into_dyn();
        }
    }
    Ok(())
}

fn fill_holes_in_plane(plane: &mut [u32], height: usize, width: usize) {
    let labels = unique_nonzero(plane.iter().copied());
    for label in labels {
        let mut min_y = height;
        let mut min_x = width;
        let mut max_y = 0usize;
        let mut max_x = 0usize;
        let mut found = false;
        for y in 0..height {
            for x in 0..width {
                if plane[y * width + x] == label {
                    min_y = min_y.min(y);
                    min_x = min_x.min(x);
                    max_y = max_y.max(y + 1);
                    max_x = max_x.max(x + 1);
                    found = true;
                }
            }
        }
        if !found {
            continue;
        }
        let sub_h = max_y - min_y;
        let sub_w = max_x - min_x;
        let mut object_mask = vec![false; sub_h * sub_w];
        for y in min_y..max_y {
            for x in min_x..max_x {
                if plane[y * width + x] == label {
                    object_mask[(y - min_y) * sub_w + (x - min_x)] = true;
                }
            }
        }
        let exterior = flood_background(&object_mask, sub_h, sub_w);
        for y in 0..sub_h {
            for x in 0..sub_w {
                let idx = y * sub_w + x;
                if !object_mask[idx] && !exterior[idx] {
                    plane[(min_y + y) * width + (min_x + x)] = label;
                }
            }
        }
    }
}

fn filter_mask_data(
    masks: &mut MaskData,
    coords_table: &Table,
    x_col: &str,
    y_col: &str,
    z_col: Option<&str>,
    frame_col: Option<&str>,
    position_col: Option<&str>,
    position_value: Option<&str>,
) -> Result<()> {
    let mut rows = table_to_rows(coords_table);
    if let (Some(column), Some(value)) = (position_col, position_value) {
        rows.retain(|row| {
            row.get(column).map(TableValue::as_string_lossy).as_deref() == Some(value)
        });
    }
    match masks.layout {
        SegmentationLayout::YX | SegmentationLayout::ZYX => {
            filter_single_volume(
                masks.values.view_mut().into_dyn(),
                &rows,
                x_col,
                y_col,
                z_col,
            )?;
        }
        SegmentationLayout::TYX => {
            let frame_col =
                frame_col.ok_or_else(|| anyhow!("frame_col is required for TYX filtering"))?;
            let mut array = masks.values.clone().into_dimensionality::<ndarray::Ix3>()?;
            for (frame_i, frame) in array.outer_iter_mut().enumerate() {
                let frame_rows = rows
                    .iter()
                    .filter(|row| get_required_i64(row, frame_col).ok() == Some(frame_i as i64))
                    .cloned()
                    .collect::<Vec<_>>();
                filter_single_volume(frame.into_dyn(), &frame_rows, x_col, y_col, None)?;
            }
            masks.values = array.into_dyn();
        }
        SegmentationLayout::TZYX => {
            let frame_col =
                frame_col.ok_or_else(|| anyhow!("frame_col is required for TZYX filtering"))?;
            let mut array = masks.values.clone().into_dimensionality::<ndarray::Ix4>()?;
            for (frame_i, volume) in array.outer_iter_mut().enumerate() {
                let frame_rows = rows
                    .iter()
                    .filter(|row| get_required_i64(row, frame_col).ok() == Some(frame_i as i64))
                    .cloned()
                    .collect::<Vec<_>>();
                filter_single_volume(volume.into_dyn(), &frame_rows, x_col, y_col, z_col)?;
            }
            masks.values = array.into_dyn();
        }
    }
    Ok(())
}

fn filter_single_volume(
    mut volume: ndarray::ArrayViewMutD<'_, u32>,
    rows: &[Row],
    x_col: &str,
    y_col: &str,
    z_col: Option<&str>,
) -> Result<()> {
    let mut ids_to_keep = BTreeSet::new();
    match volume.ndim() {
        2 => {
            let array = volume.view().into_dimensionality::<ndarray::Ix2>()?;
            for row in rows {
                let x = get_required_i64(row, x_col)? as usize;
                let y = get_required_i64(row, y_col)? as usize;
                if y < array.shape()[0] && x < array.shape()[1] {
                    let label = array[(y, x)];
                    if label > 0 {
                        ids_to_keep.insert(label);
                    }
                }
            }
            let mut array = volume.view_mut().into_dimensionality::<ndarray::Ix2>()?;
            for value in array.iter_mut() {
                if *value != 0 && !ids_to_keep.contains(value) {
                    *value = 0;
                }
            }
        }
        3 => {
            let z_col =
                z_col.ok_or_else(|| anyhow!("z_col is required for 3D coordinate filtering"))?;
            let array = volume.view().into_dimensionality::<ndarray::Ix3>()?;
            for row in rows {
                let z = get_required_i64(row, z_col)? as usize;
                let y = get_required_i64(row, y_col)? as usize;
                let x = get_required_i64(row, x_col)? as usize;
                if z < array.shape()[0] && y < array.shape()[1] && x < array.shape()[2] {
                    let label = array[(z, y, x)];
                    if label > 0 {
                        ids_to_keep.insert(label);
                    }
                }
            }
            let mut array = volume.view_mut().into_dimensionality::<ndarray::Ix3>()?;
            for value in array.iter_mut() {
                if *value != 0 && !ids_to_keep.contains(value) {
                    *value = 0;
                }
            }
        }
        ndim => bail!("Unsupported ndim {} for coordinate filtering", ndim),
    }
    Ok(())
}

fn apply_tracking_to_mask_data(
    masks: &mut MaskData,
    tracking_table: &Table,
    columns: &TrackingColumnMap,
) -> Result<(
    BTreeMap<String, serde_json::Value>,
    BTreeMap<String, Vec<u32>>,
)> {
    let rows = table_to_rows(tracking_table);
    let mut grouped = BTreeMap::<usize, Vec<Row>>::new();
    for row in rows {
        let mut frame_i = get_required_i64(&row, &columns.frame_index_col)?;
        if columns.is_first_frame_one {
            frame_i -= 1;
        }
        if frame_i < 0 {
            continue;
        }
        grouped.entry(frame_i as usize).or_default().push(row);
    }

    let mut tracked_ids_mapper = BTreeMap::<String, serde_json::Value>::new();
    let mut deleted_ids_mapper = BTreeMap::<String, Vec<u32>>::new();
    match masks.layout {
        SegmentationLayout::TYX => {
            let mut array = masks.values.clone().into_dimensionality::<ndarray::Ix3>()?;
            for (frame_i, rows) in grouped {
                if frame_i >= array.shape()[0] {
                    break;
                }
                let mut plane = array.index_axis_mut(ndarray::Axis(0), frame_i);
                let shape = plane.shape().to_vec();
                let (mapping, deleted) = apply_tracking_frame_2d(
                    plane.as_slice_mut().expect("contiguous"),
                    shape[0],
                    shape[1],
                    rows,
                    columns,
                )?;
                if !mapping.is_null() {
                    tracked_ids_mapper.insert(frame_i.to_string(), mapping);
                }
                if !deleted.is_empty() {
                    deleted_ids_mapper.insert(frame_i.to_string(), deleted);
                }
            }
            masks.values = array.into_dyn();
        }
        SegmentationLayout::TZYX => {
            let mut array = masks.values.clone().into_dimensionality::<ndarray::Ix4>()?;
            for (frame_i, rows) in grouped {
                if frame_i >= array.shape()[0] {
                    break;
                }
                let mut volume = array.index_axis_mut(ndarray::Axis(0), frame_i);
                let shape = volume.shape().to_vec();
                let (mapping, deleted) = apply_tracking_frame_3d(
                    volume.as_slice_mut().expect("contiguous"),
                    shape[0],
                    shape[1],
                    shape[2],
                    rows,
                    columns,
                )?;
                if !mapping.is_null() {
                    tracked_ids_mapper.insert(frame_i.to_string(), mapping);
                }
                if !deleted.is_empty() {
                    deleted_ids_mapper.insert(frame_i.to_string(), deleted);
                }
            }
            masks.values = array.into_dyn();
        }
        _ => bail!("Tracking is only supported for layouts with a time axis"),
    }
    Ok((tracked_ids_mapper, deleted_ids_mapper))
}

fn apply_tracking_frame_2d(
    frame: &mut [u32],
    height: usize,
    width: usize,
    rows: Vec<Row>,
    columns: &TrackingColumnMap,
) -> Result<(serde_json::Value, Vec<u32>)> {
    let mut parsed_rows = rows
        .into_iter()
        .map(|row| ParsedTrackingRow::from_row(row, columns, false))
        .collect::<Result<Vec<_>>>()?;
    let delete_ids = if columns.delete_untracked_ids {
        let tracked_ids = parsed_rows
            .iter()
            .filter_map(|row| row.resolve_mask_id_2d(frame, height, width).ok().flatten())
            .collect::<BTreeSet<_>>();
        let mut deleted = Vec::new();
        for label in unique_nonzero(frame.iter().copied()) {
            if tracked_ids.contains(&label) {
                continue;
            }
            replace_label(frame, label, 0);
            deleted.push(label);
        }
        deleted
    } else {
        Vec::new()
    };

    let mut first_pass = BTreeMap::<u32, u32>::new();
    let mut max_track_id = parsed_rows
        .iter()
        .map(|row| row.track_id)
        .max()
        .unwrap_or(0);
    let track_ids = parsed_rows
        .iter()
        .map(|row| row.track_id)
        .collect::<BTreeSet<_>>();
    for row in &mut parsed_rows {
        let Some(mask_id) = row.resolve_mask_id_2d(frame, height, width)? else {
            continue;
        };
        if mask_id == row.track_id || row.track_id == 0 || !frame.contains(&row.track_id) {
            continue;
        }
        let mut unique_id = frame.iter().copied().max().unwrap_or(0) + 1;
        if track_ids.contains(&unique_id) {
            max_track_id += 1;
            unique_id = max_track_id;
        }
        replace_label(frame, row.track_id, unique_id);
        first_pass.insert(row.track_id, unique_id);
        row.mask_id = Some(if mask_id == row.track_id {
            unique_id
        } else {
            mask_id
        });
    }

    let mut second_pass = BTreeMap::<u32, u32>::new();
    for row in &mut parsed_rows {
        let Some(mask_id) = row.resolve_mask_id_2d(frame, height, width)? else {
            continue;
        };
        if mask_id == row.track_id || row.track_id == 0 {
            continue;
        }
        replace_label(frame, mask_id, row.track_id);
        second_pass.insert(mask_id, row.track_id);
        row.mask_id = Some(row.track_id);
    }

    Ok((build_mapping_json(first_pass, second_pass), delete_ids))
}

fn apply_tracking_frame_3d(
    frame: &mut [u32],
    depth: usize,
    height: usize,
    width: usize,
    rows: Vec<Row>,
    columns: &TrackingColumnMap,
) -> Result<(serde_json::Value, Vec<u32>)> {
    let mut parsed_rows = rows
        .into_iter()
        .map(|row| ParsedTrackingRow::from_row(row, columns, true))
        .collect::<Result<Vec<_>>>()?;
    let delete_ids = if columns.delete_untracked_ids {
        let tracked_ids = parsed_rows
            .iter()
            .filter_map(|row| {
                row.resolve_mask_id_3d(frame, depth, height, width)
                    .ok()
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        let mut deleted = Vec::new();
        for label in unique_nonzero(frame.iter().copied()) {
            if tracked_ids.contains(&label) {
                continue;
            }
            replace_label(frame, label, 0);
            deleted.push(label);
        }
        deleted
    } else {
        Vec::new()
    };

    let mut first_pass = BTreeMap::<u32, u32>::new();
    let mut max_track_id = parsed_rows
        .iter()
        .map(|row| row.track_id)
        .max()
        .unwrap_or(0);
    let track_ids = parsed_rows
        .iter()
        .map(|row| row.track_id)
        .collect::<BTreeSet<_>>();
    for row in &mut parsed_rows {
        let Some(mask_id) = row.resolve_mask_id_3d(frame, depth, height, width)? else {
            continue;
        };
        if mask_id == row.track_id || row.track_id == 0 || !frame.contains(&row.track_id) {
            continue;
        }
        let mut unique_id = frame.iter().copied().max().unwrap_or(0) + 1;
        if track_ids.contains(&unique_id) {
            max_track_id += 1;
            unique_id = max_track_id;
        }
        replace_label(frame, row.track_id, unique_id);
        first_pass.insert(row.track_id, unique_id);
        row.mask_id = Some(if mask_id == row.track_id {
            unique_id
        } else {
            mask_id
        });
    }

    let mut second_pass = BTreeMap::<u32, u32>::new();
    for row in &mut parsed_rows {
        let Some(mask_id) = row.resolve_mask_id_3d(frame, depth, height, width)? else {
            continue;
        };
        if mask_id == row.track_id || row.track_id == 0 {
            continue;
        }
        replace_label(frame, mask_id, row.track_id);
        second_pass.insert(mask_id, row.track_id);
        row.mask_id = Some(row.track_id);
    }

    Ok((build_mapping_json(first_pass, second_pass), delete_ids))
}

fn apply_tracked_ids_mapper_to_table(
    table: &Table,
    tracked_ids_mapper: &BTreeMap<String, serde_json::Value>,
    deleted_ids_mapper: &BTreeMap<String, Vec<u32>>,
) -> Result<Table> {
    let frame_idx = table.header_index("frame_i")?;
    let cell_idx = table.header_index("Cell_ID")?;
    let rel_idx = table.maybe_header_index("relative_ID");
    let lineage_indices = [
        table.maybe_header_index("Cell_ID_tree"),
        table.maybe_header_index("parent_ID_tree"),
        table.maybe_header_index("root_ID_tree"),
        table.maybe_header_index("sister_ID_tree"),
    ];
    let mut rows = Vec::new();
    for row in &table.rows {
        let frame = row[frame_idx]
            .as_i64()
            .ok_or_else(|| anyhow!("Invalid frame_i value in acdc_output table"))?;
        let cell_id = row[cell_idx]
            .as_i64()
            .ok_or_else(|| anyhow!("Invalid Cell_ID value in acdc_output table"))?
            as u32;
        if deleted_ids_mapper
            .get(&frame.to_string())
            .map(|deleted| deleted.contains(&cell_id))
            .unwrap_or(false)
        {
            continue;
        }
        let mut new_row = row.clone();
        if let Some(mapper) = tracked_ids_mapper.get(&frame.to_string()) {
            let first_pass = mapper.get("first_pass").and_then(|value| value.as_object());
            let second_pass = mapper
                .get("second_pass")
                .and_then(|value| value.as_object());
            let remap = |value: u32| -> u32 {
                first_pass
                    .and_then(|mapping| mapping.get(&value.to_string()))
                    .and_then(|value| value.as_u64())
                    .map(|value| value as u32)
                    .or_else(|| {
                        second_pass
                            .and_then(|mapping| mapping.get(&value.to_string()))
                            .and_then(|value| value.as_u64())
                            .map(|value| value as u32)
                    })
                    .unwrap_or(value)
            };
            new_row[cell_idx] = TableValue::Number(remap(cell_id) as f64);
            if let Some(rel_idx) = rel_idx {
                if let Some(relative_id) = new_row[rel_idx].as_i64() {
                    if relative_id >= 0 {
                        new_row[rel_idx] = TableValue::Number(remap(relative_id as u32) as f64);
                    }
                }
            }
            for index in lineage_indices.into_iter().flatten() {
                if let Some(lineage_id) = new_row[index].as_i64() {
                    if lineage_id >= 0 {
                        new_row[index] = TableValue::Number(remap(lineage_id as u32) as f64);
                    }
                }
            }
        }
        rows.push(new_row);
    }
    let mut table = Table {
        headers: table.headers.clone(),
        rows,
    };
    sort_table_by_keys(&mut table, &["frame_i", "Cell_ID"]);
    Ok(table)
}

fn generate_mother_bud_total_rows(
    rows: &[Row],
    column_operation_mapper: &BTreeMap<String, String>,
    copy_all_nonselected_columns: bool,
    grouping_columns: &[String],
    entity_colname: &str,
) -> Result<Vec<Row>> {
    let g1_rows = rows
        .iter()
        .filter(|row| {
            row.get("cell_cycle_stage")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("G1")
        })
        .cloned()
        .collect::<Vec<_>>();
    let s_rows = rows
        .iter()
        .filter(|row| {
            row.get("cell_cycle_stage")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("S")
        })
        .cloned()
        .collect::<Vec<_>>();
    let s_bud_rows = s_rows
        .iter()
        .filter(|row| {
            row.get("relationship")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("bud")
        })
        .cloned()
        .collect::<Vec<_>>();
    let s_mother_rows = s_rows
        .iter()
        .filter(|row| {
            row.get("relationship")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("mother")
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut bud_by_key = BTreeMap::<Vec<String>, Row>::new();
    for row in s_bud_rows.clone() {
        let mut key = grouping_columns
            .iter()
            .map(|column| {
                row.get(column)
                    .map(TableValue::as_string_lossy)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        key.push(get_required_i64(&row, "frame_i")?.to_string());
        key.push(get_required_i64(&row, "relative_ID")?.to_string());
        bud_by_key.insert(key, row);
    }

    let columns_to_add = column_operation_mapper
        .iter()
        .filter(|(_, operation)| operation.to_ascii_lowercase().contains("sum"))
        .map(|(column, _)| column.clone())
        .collect::<Vec<_>>();

    let mut total_rows = Vec::new();
    for mother in s_mother_rows.clone() {
        let mut key = grouping_columns
            .iter()
            .map(|column| {
                mother
                    .get(column)
                    .map(TableValue::as_string_lossy)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        key.push(get_required_i64(&mother, "frame_i")?.to_string());
        key.push(get_required_i64(&mother, "Cell_ID")?.to_string());
        let Some(bud) = bud_by_key.get(&key) else {
            continue;
        };
        let mut total = if copy_all_nonselected_columns {
            mother.clone()
        } else {
            let mut row = Row::new();
            for column in column_operation_mapper.keys() {
                if let Some(value) = mother.get(column).cloned() {
                    row.insert(column.clone(), value);
                }
            }
            row
        };
        for column in &columns_to_add {
            let value = mother
                .get(column)
                .and_then(TableValue::as_f64)
                .unwrap_or(f64::NAN)
                + bud
                    .get(column)
                    .and_then(TableValue::as_f64)
                    .unwrap_or(f64::NAN);
            total.insert(column.clone(), TableValue::Number(value));
        }
        total_rows.push(total);
    }

    let mut output = Vec::new();
    for row in g1_rows {
        let mut row = row;
        row.insert(entity_colname.to_string(), TableValue::Text("G1".into()));
        output.push(row);
    }
    for row in s_mother_rows {
        let mut row = row;
        row.insert(
            entity_colname.to_string(),
            TableValue::Text("Mother".into()),
        );
        output.push(row);
    }
    for row in s_bud_rows {
        let mut row = row;
        row.insert(entity_colname.to_string(), TableValue::Text("Bud".into()));
        output.push(row);
    }
    for mut row in total_rows {
        row.insert(entity_colname.to_string(), TableValue::Text("Total".into()));
        output.push(row);
    }
    Ok(output)
}

fn ensure_required_columns(table: &Table, columns: &[&str]) -> Result<()> {
    for column in columns {
        table.header_index(column)?;
    }
    Ok(())
}

fn collect_images_dirs_from_scope(
    position_dir: Option<&Path>,
    experiment_dir: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    match (position_dir, experiment_dir) {
        (Some(position_dir), None) => Ok(vec![normalize_images_dir(position_dir)?]),
        (None, Some(experiment_dir)) => list_position_dirs(experiment_dir)?
            .into_iter()
            .map(|position| normalize_images_dir(&position))
            .collect(),
        _ => bail!("Provide exactly one of position_dir or experiment_dir"),
    }
}

fn collect_measurement_positions_from_scope(
    position_dir: Option<&Path>,
    experiment_dir: Option<&Path>,
) -> Result<Vec<crate::layout::MeasurementPositionSpec>> {
    match (position_dir, experiment_dir) {
        (Some(position_dir), None) => Ok(vec![resolve_measurement_position(position_dir)?]),
        (None, Some(experiment_dir)) => {
            Ok(discover_measurement_experiment(experiment_dir)?.positions)
        }
        _ => bail!("Provide exactly one of position_dir or experiment_dir"),
    }
}

fn normalize_images_dir(path: &Path) -> Result<PathBuf> {
    if path.file_name().and_then(|name| name.to_str()) == Some("Images") {
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }
        bail!("Images path is not a directory: {}", path.display());
    }
    let images_dir = path.join("Images");
    if images_dir.is_dir() {
        return Ok(images_dir);
    }
    bail!(
        "Expected a Cell-ACDC position directory or Images directory, got {}",
        path.display()
    )
}

fn find_table_by_endname(images_dir: &Path, endname: &str) -> Result<Option<PathBuf>> {
    let mut matches = fs::read_dir(images_dir)
        .with_context(|| format!("Failed to read {}", images_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(|name| {
                    name.ends_with(&format!("{endname}.csv"))
                        || name.ends_with(&format!("{endname}.xlsx"))
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches.into_iter().next())
}

fn find_file_by_endname(
    images_dir: &Path,
    endname: &str,
    extensions: &[&str],
) -> Result<Option<PathBuf>> {
    let mut matches = fs::read_dir(images_dir)
        .with_context(|| format!("Failed to read {}", images_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let ext_matches = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| {
                    let value = value.to_ascii_lowercase();
                    extensions.iter().any(|ext| value == *ext)
                })
                .unwrap_or(false);
            if !ext_matches {
                return false;
            }
            let filename = path.file_name().and_then(|value| value.to_str());
            let stem = path.file_stem().and_then(|value| value.to_str());
            filename
                .map(|name| name.ends_with(endname))
                .unwrap_or(false)
                || stem.map(|name| name.ends_with(endname)).unwrap_or(false)
                || filename
                    .map(|name| {
                        extensions
                            .iter()
                            .any(|ext| name.ends_with(&format!("{endname}.{ext}")))
                    })
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches.into_iter().next())
}

fn list_position_dirs(experiment_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut positions = fs::read_dir(experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .map(|name| name.starts_with("Position_"))
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    positions.sort();
    if positions.is_empty() {
        bail!(
            "No Cell-ACDC positions found under {}",
            experiment_dir.display()
        );
    }
    Ok(positions)
}

fn derive_acdc_output_path(source: &Path, segmentation_output: &Path) -> PathBuf {
    let source_ext = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("csv");
    let stem = segmentation_output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("segm")
        .replace("segm", "acdc_output");
    segmentation_output.with_file_name(format!("{stem}.{source_ext}"))
}

fn infer_tracking_source_acdc_output(images_dir: &Path, segm_endname: &str) -> Option<PathBuf> {
    let acdc_endname = if segm_endname.starts_with("segm") {
        segm_endname.replacen("segm", "acdc_output", 1)
    } else {
        format!("acdc_output_{segm_endname}")
    };
    find_table_by_endname(images_dir, &acdc_endname)
        .ok()
        .flatten()
}

fn append_to_file_stem(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if ext.is_empty() {
        path.with_file_name(format!("{stem}{suffix}"))
    } else {
        path.with_file_name(format!("{stem}{suffix}.{ext}"))
    }
}

fn append_text_to_filename(path: &Path, append_text: &str) -> Result<PathBuf> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("Invalid file path {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let suffix = if append_text.starts_with('_') {
        append_text.to_string()
    } else {
        format!("_{append_text}")
    };
    if ext.is_empty() {
        Ok(path.with_file_name(format!("{stem}{suffix}")))
    } else {
        Ok(path.with_file_name(format!("{stem}{suffix}.{ext}")))
    }
}

fn infer_table_basename(table_path: &Path, endname: &str) -> Result<String> {
    let filename = table_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("Invalid table path {}", table_path.display()))?;
    for ext in ["csv", "xlsx", "xlsm", "xls"] {
        let suffix = format!("{endname}.{ext}");
        if let Some(prefix) = filename.strip_suffix(&suffix) {
            return Ok(prefix.to_string());
        }
    }
    bail!(
        "Failed to infer Cell-ACDC basename from {} and endname {:?}",
        table_path.display(),
        endname
    )
}

fn replace_file_stem_suffix(path: &Path, replacement: &str) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("ini");
    format!("{replacement}_{}.{}", stem, extension)
}

fn table_extension(format: TableFormat) -> &'static str {
    match format {
        TableFormat::Csv => "csv",
        TableFormat::Xlsx => "xlsx",
    }
}

fn replace_output_name_metric(name: &str, from: &str, to: &str) -> String {
    name.replace(from, to)
}

fn selected_or_all_headers(selected: Option<&Vec<String>>, headers: &[String]) -> Vec<String> {
    selected.cloned().unwrap_or_else(|| headers.to_vec())
}

fn shared_key_columns(tables: &[Table]) -> Vec<String> {
    DEFAULT_INDEX_COLS
        .iter()
        .filter(|column| {
            tables
                .iter()
                .all(|table| table.headers.iter().any(|header| header == **column))
        })
        .map(|column| (*column).to_string())
        .collect()
}

fn sanitize_identifier(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "_".into()
    } else {
        out
    }
}

fn metric_alias(table_number: usize, header: &str) -> String {
    format!("table{}_{}", table_number, sanitize_identifier(header))
}

fn metric_alias_python_style(table_number: usize, header: &str) -> String {
    format!("{}_table{}", sanitize_identifier(header), table_number)
}

fn metric_aliases(table_number: usize, header: &str) -> [String; 2] {
    [
        metric_alias(table_number, header),
        metric_alias_python_style(table_number, header),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RowKey(Vec<String>);

fn build_key(row: &Row, columns: &[String], table_idx: usize, row_idx: usize) -> RowKey {
    if columns.len() == 1 && columns[0] == "__row_index" {
        return RowKey(vec![table_idx.to_string(), row_idx.to_string()]);
    }
    RowKey(
        columns
            .iter()
            .map(|column| {
                row.get(column)
                    .cloned()
                    .unwrap_or(TableValue::Empty)
                    .as_string_lossy()
            })
            .collect(),
    )
}

fn row_key(row: &Row) -> Result<(i64, i64)> {
    Ok((
        get_required_i64(row, "frame_i")?,
        get_required_i64(row, "Cell_ID")?,
    ))
}

type Row = BTreeMap<String, TableValue>;

fn table_to_rows(table: &Table) -> Vec<Row> {
    table
        .rows
        .iter()
        .map(|row| {
            table
                .headers
                .iter()
                .cloned()
                .zip(row.iter().cloned())
                .collect::<Row>()
        })
        .collect()
}

fn rows_to_table(headers: &[String], rows: &[Row]) -> Table {
    let mut table = Table::new(headers.to_vec());
    for row in rows {
        let values = headers
            .iter()
            .map(|header| row.get(header).cloned().unwrap_or(TableValue::Empty))
            .collect::<Vec<_>>();
        table.rows.push(values);
    }
    table
}

fn infer_headers_from_rows(rows: &[Row]) -> Vec<String> {
    let mut headers = Vec::new();
    for row in rows {
        extend_header_order(&mut headers, row.keys().cloned());
    }
    headers
}

fn extend_header_order(headers: &mut Vec<String>, values: impl IntoIterator<Item = String>) {
    for value in values {
        if !headers.iter().any(|header| header == &value) {
            headers.push(value);
        }
    }
}

fn sort_table_by_keys(table: &mut Table, keys: &[&str]) {
    let indices = keys
        .iter()
        .filter_map(|key| table.maybe_header_index(key))
        .collect::<Vec<_>>();
    table.rows.sort_by(|left, right| {
        indices
            .iter()
            .map(|idx| compare_table_values(&left[*idx], &right[*idx]))
            .find(|ordering| *ordering != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn compare_table_values(left: &TableValue, right: &TableValue) -> std::cmp::Ordering {
    match (left, right) {
        (TableValue::Number(left), TableValue::Number(right)) => {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        }
        _ => left.as_string_lossy().cmp(&right.as_string_lossy()),
    }
}

fn get_required_i64(row: &Row, column: &str) -> Result<i64> {
    row.get(column)
        .and_then(TableValue::as_i64)
        .ok_or_else(|| anyhow!("Missing or invalid integer column {column:?}"))
}

fn get_required_bool(row: &Row, column: &str) -> Result<bool> {
    row.get(column)
        .map(|value| match value {
            TableValue::Bool(value) => Some(*value),
            TableValue::Number(value) => Some(*value != 0.0),
            TableValue::Text(value) => match value.to_ascii_lowercase().as_str() {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            },
            TableValue::Empty => None,
        })
        .flatten()
        .ok_or_else(|| anyhow!("Missing or invalid boolean column {column:?}"))
}

fn unique_nonzero(values: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
    values.into_iter().filter(|value| *value != 0).collect()
}

fn replace_label(frame: &mut [u32], from: u32, to: u32) {
    for value in frame.iter_mut() {
        if *value == from {
            *value = to;
        }
    }
}

fn build_mapping_json(
    first_pass: BTreeMap<u32, u32>,
    second_pass: BTreeMap<u32, u32>,
) -> serde_json::Value {
    if first_pass.is_empty() && second_pass.is_empty() {
        return serde_json::Value::Null;
    }
    let mut object = serde_json::Map::new();
    if !first_pass.is_empty() {
        object.insert(
            "first_pass".into(),
            json!(first_pass
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect::<BTreeMap<_, _>>()),
        );
    }
    if !second_pass.is_empty() {
        object.insert(
            "second_pass".into(),
            json!(second_pass
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect::<BTreeMap<_, _>>()),
        );
    }
    serde_json::Value::Object(object)
}

fn flood_background(mask: &[bool], height: usize, width: usize) -> Vec<bool> {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    if height == 0 || width == 0 {
        return visited;
    }
    for x in 0..width {
        for y in [0usize, height - 1] {
            let idx = y * width + x;
            if !mask[idx] && !visited[idx] {
                visited[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    for y in 0..height {
        for x in [0usize, width - 1] {
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

#[derive(Debug, Clone)]
struct ParsedTrackingRow {
    track_id: u32,
    mask_id: Option<u32>,
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
}

impl ParsedTrackingRow {
    fn from_row(row: Row, columns: &TrackingColumnMap, allow_z: bool) -> Result<Self> {
        let track_id = get_required_i64(&row, &columns.track_ids_col)? as u32;
        let mask_id = columns
            .mask_ids_col
            .as_ref()
            .and_then(|column| row.get(column))
            .and_then(TableValue::as_i64)
            .map(|value| value as u32);
        let x = columns
            .x_centroid_col
            .as_ref()
            .and_then(|column| row.get(column))
            .and_then(TableValue::as_i64)
            .map(|value| value as usize);
        let y = columns
            .y_centroid_col
            .as_ref()
            .and_then(|column| row.get(column))
            .and_then(TableValue::as_i64)
            .map(|value| value as usize);
        let z = if allow_z {
            columns
                .z_centroid_col
                .as_ref()
                .and_then(|column| row.get(column))
                .and_then(TableValue::as_i64)
                .map(|value| value as usize)
        } else {
            None
        };
        Ok(Self {
            track_id,
            mask_id,
            x,
            y,
            z,
        })
    }

    fn resolve_mask_id_2d(
        &self,
        frame: &[u32],
        height: usize,
        width: usize,
    ) -> Result<Option<u32>> {
        if let Some(mask_id) = self.mask_id {
            return Ok(Some(mask_id));
        }
        let (Some(x), Some(y)) = (self.x, self.y) else {
            return Ok(None);
        };
        if y >= height || x >= width {
            return Ok(None);
        }
        let label = frame[y * width + x];
        if label == 0 {
            return Ok(None);
        }
        Ok(Some(label))
    }

    fn resolve_mask_id_3d(
        &self,
        frame: &[u32],
        depth: usize,
        height: usize,
        width: usize,
    ) -> Result<Option<u32>> {
        if let Some(mask_id) = self.mask_id {
            return Ok(Some(mask_id));
        }
        let (Some(x), Some(y), Some(z)) = (self.x, self.y, self.z) else {
            return Ok(None);
        };
        if z >= depth || y >= height || x >= width {
            return Ok(None);
        }
        let label = frame[z * height * width + y * width + x];
        if label == 0 {
            return Ok(None);
        }
        Ok(Some(label))
    }
}

fn row_from_pairs(entries: Vec<(&str, TableValue)>) -> Row {
    let mut row = Row::new();
    for (key, value) in entries {
        row.insert(key.to_string(), value);
    }
    row
}

#[derive(Debug, Clone)]
enum ArrayExpr {
    Number(f32),
    Variable(String),
    Neg(Box<ArrayExpr>),
    Add(Box<ArrayExpr>, Box<ArrayExpr>),
    Sub(Box<ArrayExpr>, Box<ArrayExpr>),
    Mul(Box<ArrayExpr>, Box<ArrayExpr>),
    Div(Box<ArrayExpr>, Box<ArrayExpr>),
}

#[derive(Debug, Clone)]
enum ArrayValue {
    Scalar(f32),
    Array(ArrayD<f32>),
}

impl ArrayExpr {
    fn evaluate(&self, variables: &BTreeMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>> {
        let value = self.eval_value(variables)?;
        match value {
            ArrayValue::Array(array) => Ok(array),
            ArrayValue::Scalar(_) => bail!("Array formula must reference at least one channel"),
        }
    }

    fn eval_value(&self, variables: &BTreeMap<String, ArrayD<f32>>) -> Result<ArrayValue> {
        match self {
            Self::Number(value) => Ok(ArrayValue::Scalar(*value)),
            Self::Variable(name) => variables
                .get(name)
                .cloned()
                .map(ArrayValue::Array)
                .ok_or_else(|| anyhow!("Unknown combine_channels variable {:?}", name)),
            Self::Neg(value) => match value.eval_value(variables)? {
                ArrayValue::Scalar(value) => Ok(ArrayValue::Scalar(-value)),
                ArrayValue::Array(array) => Ok(ArrayValue::Array(array.mapv(|value| -value))),
            },
            Self::Add(left, right) => apply_array_binary(left, right, variables, |l, r| l + r),
            Self::Sub(left, right) => apply_array_binary(left, right, variables, |l, r| l - r),
            Self::Mul(left, right) => apply_array_binary(left, right, variables, |l, r| l * r),
            Self::Div(left, right) => apply_array_binary(left, right, variables, |l, r| l / r),
        }
    }
}

fn apply_array_binary(
    left: &ArrayExpr,
    right: &ArrayExpr,
    variables: &BTreeMap<String, ArrayD<f32>>,
    op: impl Fn(f32, f32) -> f32 + Copy,
) -> Result<ArrayValue> {
    let left = left.eval_value(variables)?;
    let right = right.eval_value(variables)?;
    Ok(match (left, right) {
        (ArrayValue::Scalar(left), ArrayValue::Scalar(right)) => {
            ArrayValue::Scalar(op(left, right))
        }
        (ArrayValue::Array(left), ArrayValue::Scalar(right)) => {
            ArrayValue::Array(left.mapv(|value| op(value, right)))
        }
        (ArrayValue::Scalar(left), ArrayValue::Array(right)) => {
            ArrayValue::Array(right.mapv(|value| op(left, value)))
        }
        (ArrayValue::Array(left), ArrayValue::Array(right)) => {
            if left.shape() != right.shape() {
                bail!(
                    "combine_channels formula shape mismatch: {:?} vs {:?}",
                    left.shape(),
                    right.shape()
                );
            }
            let values = left
                .iter()
                .copied()
                .zip(right.iter().copied())
                .map(|(l, r)| op(l, r))
                .collect::<Vec<_>>();
            ArrayValue::Array(ArrayD::from_shape_vec(IxDyn(left.shape()), values)?)
        }
    })
}

fn parse_array_expression(source: &str) -> Result<ArrayExpr> {
    let mut parser = ArrayExprParser::new(source);
    let expr = parser.parse_expression()?;
    parser.skip_whitespace();
    if !parser.is_eof() {
        bail!(
            "Unexpected trailing tokens in combine_channels formula near {:?}",
            parser.remaining()
        );
    }
    Ok(expr)
}

struct ArrayExprParser<'a> {
    source: &'a str,
    index: usize,
}

impl<'a> ArrayExprParser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, index: 0 }
    }

    fn parse_expression(&mut self) -> Result<ArrayExpr> {
        let mut expr = self.parse_term()?;
        loop {
            self.skip_whitespace();
            if self.consume('+') {
                expr = ArrayExpr::Add(Box::new(expr), Box::new(self.parse_term()?));
            } else if self.consume('-') {
                expr = ArrayExpr::Sub(Box::new(expr), Box::new(self.parse_term()?));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<ArrayExpr> {
        let mut expr = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            if self.consume('*') {
                expr = ArrayExpr::Mul(Box::new(expr), Box::new(self.parse_factor()?));
            } else if self.consume('/') {
                expr = ArrayExpr::Div(Box::new(expr), Box::new(self.parse_factor()?));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<ArrayExpr> {
        self.skip_whitespace();
        if self.consume('(') {
            let expr = self.parse_expression()?;
            self.skip_whitespace();
            if !self.consume(')') {
                bail!("Missing closing ')' in combine_channels formula");
            }
            return Ok(expr);
        }
        if self.consume('-') {
            return Ok(ArrayExpr::Neg(Box::new(self.parse_factor()?)));
        }
        if let Some(number) = self.parse_number()? {
            return Ok(ArrayExpr::Number(number));
        }
        let identifier = self.parse_identifier()?;
        Ok(ArrayExpr::Variable(identifier))
    }

    fn parse_number(&mut self) -> Result<Option<f32>> {
        self.skip_whitespace();
        let start = self.index;
        let mut saw_digit = false;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                saw_digit = true;
                self.index += ch.len_utf8();
            } else {
                break;
            }
        }
        if !saw_digit {
            self.index = start;
            return Ok(None);
        }
        Ok(Some(
            self.source[start..self.index]
                .parse::<f32>()
                .with_context(|| format!("Invalid number {:?}", &self.source[start..self.index]))?,
        ))
    }

    fn parse_identifier(&mut self) -> Result<String> {
        self.skip_whitespace();
        let start = self.index;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.index += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.index == start {
            bail!(
                "Expected identifier in combine_channels formula near {:?}",
                self.remaining()
            );
        }
        Ok(self.source[start..self.index].to_string())
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.index += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.index += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.index..].chars().next()
    }

    fn is_eof(&self) -> bool {
        self.index >= self.source.len()
    }

    fn remaining(&self) -> &str {
        &self.source[self.index..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn counts_objects_and_writes_csv() -> Result<()> {
        let temp = tempdir()?;
        let segm_path = temp.path().join("segm.npz");
        let out_path = temp.path().join("counts.csv");
        let masks = MaskData {
            values: Array3::from_shape_vec(
                (2, 2, 3),
                vec![
                    0, 1, 1, 0, 0, 2, //
                    0, 1, 1, 0, 3, 3, //
                ],
            )?
            .into_dyn(),
            layout: SegmentationLayout::TYX,
            source_path: segm_path.clone(),
        };
        save_mask_data(&segm_path, &masks)?;
        let result = count_objects(CountObjectsConfig {
            segmentation_path: segm_path,
            output_path: out_path.clone(),
            resolution: Some(MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        })?;
        assert_eq!(
            result.summary.counts.get("Unique objects in entire video"),
            Some(&3)
        );
        assert!(out_path.exists());
        Ok(())
    }

    #[test]
    fn writes_object_coordinates_from_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let segm_path = temp.path().join("segm.npz");
        let out_path = temp.path().join("objects_coordinates.csv");
        let masks = MaskData {
            values: Array3::from_shape_vec(
                (2, 2, 2),
                vec![
                    0, 1, //
                    2, 2, //
                    3, 0, //
                    3, 0,
                ],
            )?
            .into_dyn(),
            layout: SegmentationLayout::TYX,
            source_path: segm_path.clone(),
        };
        save_mask_data(&segm_path, &masks)?;
        segmentation_to_object_coords(ObjectCoordinatesConfig {
            segmentation_path: segm_path,
            output_path: out_path.clone(),
            resolution: Some(MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        })?;
        let table = read_table(&out_path)?;
        assert_eq!(table.headers, vec!["frame_i", "Cell_ID", "y", "x"]);
        assert_eq!(table.rows.len(), 5);
        let frame_col = table.header_index("frame_i")?;
        let id_col = table.header_index("Cell_ID")?;
        let y_col = table.header_index("y")?;
        let x_col = table.header_index("x")?;
        assert!(table.rows.iter().any(|row| {
            row[frame_col].as_i64() == Some(1)
                && row[id_col].as_i64() == Some(3)
                && row[y_col].as_i64() == Some(1)
                && row[x_col].as_i64() == Some(0)
        }));
        Ok(())
    }

    #[test]
    fn adds_lineage_tree_columns() -> Result<()> {
        let table = rows_to_table(
            &[
                "frame_i".into(),
                "Cell_ID".into(),
                "cell_cycle_stage".into(),
                "generation_num".into(),
                "relative_ID".into(),
                "relationship".into(),
                "is_history_known".into(),
            ],
            &[
                row(vec![
                    ("frame_i", 0.0.into()),
                    ("Cell_ID", 1.0.into()),
                    ("cell_cycle_stage", "G1".into()),
                    ("generation_num", 1.0.into()),
                    ("relative_ID", (-1.0).into()),
                    ("relationship", "mother".into()),
                    ("is_history_known", 1.0.into()),
                ]),
                row(vec![
                    ("frame_i", 1.0.into()),
                    ("Cell_ID", 1.0.into()),
                    ("cell_cycle_stage", "S".into()),
                    ("generation_num", 1.0.into()),
                    ("relative_ID", (-1.0).into()),
                    ("relationship", "mother".into()),
                    ("is_history_known", 1.0.into()),
                ]),
            ],
        );
        let state = build_lineage_state(&table)?;
        let exported = state.to_table();
        let rows = table_to_rows(&exported);
        assert_eq!(
            rows[0].get("Cell_ID_tree").and_then(TableValue::as_i64),
            Some(1)
        );
        assert_eq!(
            rows[1].get("root_ID_tree").and_then(TableValue::as_i64),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn fills_holes_in_4d_masks() -> Result<()> {
        let temp = tempdir()?;
        let segm_path = temp.path().join("segm.npz");
        let out_path = temp.path().join("filled.npz");
        let masks = MaskData {
            values: Array4::from_shape_vec(
                (1, 1, 3, 3),
                vec![
                    1, 1, 1, //
                    1, 0, 1, //
                    1, 1, 1, //
                ],
            )?
            .into_dyn(),
            layout: SegmentationLayout::TZYX,
            source_path: segm_path.clone(),
        };
        save_mask_data(&segm_path, &masks)?;
        fill_holes(FillHolesConfig {
            segmentation_path: segm_path,
            output_path: out_path.clone(),
            resolution: Some(MaskPathResolution {
                size_t: Some(1),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TZYX),
            }),
        })?;
        let loaded = load_mask_data(
            &out_path,
            Some(&MaskPathResolution {
                size_t: Some(1),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TZYX),
            }),
        )?;
        assert_eq!(loaded.values.iter().filter(|value| **value == 0).count(), 0);
        Ok(())
    }

    #[test]
    fn concatenates_acdc_outputs_across_positions() -> Result<()> {
        let temp = tempdir()?;
        for pos in ["Position_1", "Position_2"] {
            let images = temp.path().join(pos).join("Images");
            fs::create_dir_all(&images)?;
            fs::write(
                images.join("demo_metadata.csv"),
                "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
            )?;
            write_table(
                &images.join("demo_acdc_output.csv"),
                &Table {
                    headers: vec!["frame_i".into(), "Cell_ID".into(), "value".into()],
                    rows: vec![vec![
                        TableValue::Number(0.0),
                        TableValue::Number(if pos.ends_with('1') { 1.0 } else { 2.0 }),
                        TableValue::Text(pos.to_string()),
                    ]],
                },
            )?;
        }

        let result = concat_acdc_outputs(ConcatConfig {
            experiment_dirs: vec![temp.path().to_path_buf()],
            table_endname: "acdc_output".into(),
            output_format: TableFormat::Csv,
            selected_columns: None,
            output_name: None,
            multi_experiment_dir: None,
        })?;
        let table = read_table(&result.all_position_outputs[0])?;
        assert_eq!(table.rows.len(), 2);
        assert!(table.headers.iter().any(|header| header == "Position_n"));
        Ok(())
    }

    #[test]
    fn combines_metrics_from_multiple_tables() -> Result<()> {
        let temp = tempdir()?;
        let source1 = temp.path().join("source1.csv");
        let source2 = temp.path().join("source2.csv");
        write_table(
            &source1,
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "signal".into()],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Number(2.0),
                ]],
            },
        )?;
        write_table(
            &source2,
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "signal".into()],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Number(3.0),
                ]],
            },
        )?;
        let output = temp.path().join("combined.csv");
        combine_metrics(CombineMetricsConfig {
            source_paths: vec![source1, source2],
            formulas: BTreeMap::from([(
                "sum_signal".into(),
                "table1_signal + table2_signal".into(),
            )]),
            output_path: output.clone(),
            equations_path: None,
        })?;
        let table = read_table(&output)?;
        let sum_idx = table.header_index("sum_signal")?;
        assert_eq!(table.rows[0][sum_idx].as_i64(), Some(5));
        Ok(())
    }

    #[test]
    fn computes_multi_channel_tables_with_python_style_aliases() -> Result<()> {
        let temp = tempdir()?;
        let images = temp.path().join("Position_1").join("Images");
        fs::create_dir_all(&images)?;
        let source1 = images.join("demo_acdc_output_first.csv");
        let source2 = images.join("demo_acdc_output_second.csv");
        write_table(
            &source1,
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "signal".into()],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Number(2.0),
                ]],
            },
        )?;
        write_table(
            &source2,
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "signal".into()],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Number(3.0),
                ]],
            },
        )?;

        let result = compute_multi_channel(ComputeMultiChannelConfig {
            position_dir: Some(temp.path().join("Position_1")),
            experiment_dir: None,
            source_endnames: vec!["acdc_output_first".into(), "acdc_output_second".into()],
            formulas: BTreeMap::from([(
                "sum_signal".into(),
                "signal_table1 + signal_table2".into(),
            )]),
            append_name: "combined_metrics".into(),
        })?;

        assert_eq!(result.outputs.len(), 1);
        let table = read_table(&result.outputs[0].output_path)?;
        let sum_idx = table.header_index("sum_signal")?;
        assert_eq!(table.rows[0][sum_idx].as_i64(), Some(5));
        assert!(result.outputs[0]
            .equations_path
            .ends_with("demo_equations_combined_metrics.ini"));
        Ok(())
    }

    #[test]
    fn combines_channels_and_broadcasts_2d_segmentation_over_z() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack_u16(&images.join("demo_ch1.tif"), &[10, 20])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,2\n",
        )?;
        save_mask_data(
            &images.join("demo_segm_cells.npz"),
            &MaskData {
                values: ndarray::Array2::from_shape_vec((2, 2), vec![0, 1, 1, 0])?.into_dyn(),
                layout: SegmentationLayout::YX,
                source_path: images.join("demo_segm_cells.npz"),
            },
        )?;
        let recipe_path = temp.path().join("recipe.json");
        fs::write(
            &recipe_path,
            serde_json::to_vec_pretty(&json!({
                "1": {
                    "name": "img",
                    "channel": "ch1",
                    "binarize": "No",
                    "min_val": 0.0,
                    "max_val": 1.0
                },
                "2": {
                    "name": "mask",
                    "channel": "segm_cells",
                    "binarize": "binarize",
                    "min_val": 0.0,
                    "max_val": 1.0
                },
                "formula": "mask",
                "keep_input_data_type": true,
                "save_as_segm": true
            }))?,
        )?;

        let result = combine_channels(CombineChannelsConfig {
            position_dir: Some(position),
            experiment_dir: None,
            recipe_path,
            append_name: "combined".into(),
        })?;

        let output = &result.output_paths[0];
        let loaded = load_mask_data(
            output,
            Some(&MaskPathResolution {
                size_t: Some(1),
                size_z: Some(2),
                layout: Some(SegmentationLayout::ZYX),
            }),
        )?;
        assert_eq!(loaded.layout, SegmentationLayout::ZYX);
        assert_eq!(loaded.values.shape(), &[2, 2, 2]);
        let values = loaded.values.iter().copied().collect::<Vec<_>>();
        assert_eq!(values, vec![0, 1, 1, 0, 0, 1, 1, 0]);
        Ok(())
    }

    #[test]
    fn combines_channels_into_tiff_and_preserves_integer_dtype() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_planes_u16(
            &images.join("demo_a.tif"),
            &[vec![0, 0, 65535, 65535]],
            2,
            2,
        )?;
        write_test_planes_u16(
            &images.join("demo_b.tif"),
            &[vec![0, 65535, 0, 65535]],
            2,
            2,
        )?;
        let recipe_path = temp.path().join("image_recipe.json");
        fs::write(
            &recipe_path,
            serde_json::to_vec_pretty(&json!({
                "1": {
                    "name": "A",
                    "channel": "a",
                    "binarize": "No",
                    "min_val": 0.0,
                    "max_val": 1.0
                },
                "2": {
                    "name": "B",
                    "channel": "b",
                    "binarize": "No",
                    "min_val": 0.0,
                    "max_val": 1.0
                },
                "formula": "B - A",
                "keep_input_data_type": true,
                "save_as_segm": false
            }))?,
        )?;

        let result = combine_channels(CombineChannelsConfig {
            position_dir: Some(position),
            experiment_dir: None,
            recipe_path,
            append_name: "combined".into(),
        })?;

        let (pixels, shape) = crate::image_io::load_image_stack_as_f32(&result.output_paths[0])?;
        assert_eq!(shape.frames, 1);
        assert_eq!(shape.height, 2);
        assert_eq!(shape.width, 2);
        assert_eq!(pixels[1], 65535.0);
        assert_eq!(pixels[2], 0.0);
        Ok(())
    }

    #[test]
    fn applies_trackmate_xml_tracking_and_remaps_acdc_output() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        save_mask_data(
            &images.join("demo_segm.npz"),
            &MaskData {
                values: Array3::from_shape_vec(
                    (2, 2, 2),
                    vec![
                        1, 1, 0, 0, //
                        2, 2, 0, 0, //
                    ],
                )?
                .into_dyn(),
                layout: SegmentationLayout::TYX,
                source_path: images.join("demo_segm.npz"),
            },
        )?;
        write_table(
            &images.join("demo_acdc_output.csv"),
            &Table {
                headers: vec!["frame_i".into(), "Cell_ID".into(), "relative_ID".into()],
                rows: vec![vec![
                    TableValue::Number(1.0),
                    TableValue::Number(2.0),
                    TableValue::Number(-1.0),
                ]],
            },
        )?;
        let xml_path = temp.path().join("tracks.xml");
        fs::write(
            &xml_path,
            r#"<Tracks>
<particle>
  <detection t="0" x="0" y="0" z="0"/>
  <detection t="1" x="0" y="0" z="0"/>
</particle>
</Tracks>"#,
        )?;

        let result = apply_tracking_from_trackmate_xml(ApplyTrackingFromTrackMateXmlConfig {
            position_dir: position,
            segm_endname: "segm".into(),
            xml_path,
            output_segmentation_path: None,
            source_acdc_output_path: None,
            output_acdc_output_path: None,
            delete_untracked_ids: false,
        })?;

        assert!(result.primary_path.ends_with("demo_segm_tracked.npz"));
        let output_acdc = result
            .secondary_paths
            .iter()
            .find(|path| path.ends_with("demo_acdc_output_tracked.csv"))
            .cloned()
            .expect("tracked acdc_output path");
        let table = read_table(&output_acdc)?;
        assert_eq!(
            table.rows[0][table.header_index("Cell_ID")?].as_i64(),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn filters_segmentation_from_coordinate_table() -> Result<()> {
        let temp = tempdir()?;
        let segm_path = temp.path().join("segm.npz");
        let table_path = temp.path().join("coords.csv");
        let output = temp.path().join("filtered.npz");
        let masks = MaskData {
            values: Array3::from_shape_vec(
                (2, 2, 2),
                vec![
                    1, 1, 2, 2, //
                    3, 3, 0, 0, //
                ],
            )?
            .into_dyn(),
            layout: SegmentationLayout::TYX,
            source_path: segm_path.clone(),
        };
        save_mask_data(&segm_path, &masks)?;
        write_table(
            &table_path,
            &Table {
                headers: vec!["frame_i".into(), "x".into(), "y".into()],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(0.0),
                    TableValue::Number(0.0),
                ]],
            },
        )?;
        filter_segm_from_table(CoordinateFilterConfig {
            segmentation_path: segm_path,
            coords_table_path: table_path,
            output_path: output.clone(),
            x_col: "x".into(),
            y_col: "y".into(),
            z_col: None,
            frame_col: Some("frame_i".into()),
            position_col: None,
            position_value: None,
            resolution: Some(MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        })?;
        let filtered = load_mask_data(
            &output,
            Some(&MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        )?;
        assert!(filtered
            .values
            .iter()
            .copied()
            .all(|value| value == 0 || value == 1 || value == 3));
        Ok(())
    }

    #[test]
    fn applies_tracking_table_and_updates_acdc_output() -> Result<()> {
        let temp = tempdir()?;
        let segm_path = temp.path().join("segm.npz");
        let tracking_path = temp.path().join("tracking.csv");
        let acdc_path = temp.path().join("acdc_output.csv");
        let output = temp.path().join("tracked.npz");
        let output_acdc = temp.path().join("tracked_acdc_output.csv");
        let masks = MaskData {
            values: Array3::from_shape_vec(
                (2, 2, 2),
                vec![
                    1, 1, 0, 0, //
                    2, 2, 0, 0, //
                ],
            )?
            .into_dyn(),
            layout: SegmentationLayout::TYX,
            source_path: segm_path.clone(),
        };
        save_mask_data(&segm_path, &masks)?;
        write_table(
            &tracking_path,
            &Table {
                headers: vec!["frame".into(), "track_id".into(), "mask_id".into()],
                rows: vec![vec![
                    TableValue::Number(1.0),
                    TableValue::Number(1.0),
                    TableValue::Number(2.0),
                ]],
            },
        )?;
        write_table(
            &acdc_path,
            &Table {
                headers: vec![
                    "frame_i".into(),
                    "Cell_ID".into(),
                    "relative_ID".into(),
                    "Cell_ID_tree".into(),
                    "parent_ID_tree".into(),
                    "root_ID_tree".into(),
                    "sister_ID_tree".into(),
                ],
                rows: vec![vec![
                    TableValue::Number(0.0),
                    TableValue::Number(2.0),
                    TableValue::Number(-1.0),
                    TableValue::Number(2.0),
                    TableValue::Number(-1.0),
                    TableValue::Number(2.0),
                    TableValue::Number(-1.0),
                ]],
            },
        )?;
        let result = apply_tracking_from_table(ApplyTrackingConfig {
            segmentation_path: segm_path,
            tracking_table_path: tracking_path,
            output_path: output.clone(),
            columns: TrackingColumnMap {
                frame_index_col: "frame".into(),
                is_first_frame_one: true,
                track_ids_col: "track_id".into(),
                mask_ids_col: Some("mask_id".into()),
                x_centroid_col: None,
                y_centroid_col: None,
                z_centroid_col: None,
                delete_untracked_ids: false,
            },
            resolution: Some(MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
            source_acdc_output_path: Some(acdc_path),
            output_acdc_output_path: Some(output_acdc.clone()),
        })?;
        assert!(result
            .secondary_paths
            .iter()
            .any(|path| path == &output_acdc));
        let table = read_table(&output_acdc)?;
        assert_eq!(
            table.rows[0][table.header_index("Cell_ID")?].as_i64(),
            Some(1)
        );
        assert_eq!(
            table.rows[0][table.header_index("Cell_ID_tree")?].as_i64(),
            Some(1)
        );
        assert_eq!(
            table.rows[0][table.header_index("root_ID_tree")?].as_i64(),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn generates_mother_bud_total_rows() -> Result<()> {
        let temp = tempdir()?;
        let input = temp.path().join("acdc.csv");
        let output = temp.path().join("mother_bud_total.csv");
        write_table(
            &input,
            &Table {
                headers: vec![
                    "frame_i".into(),
                    "Cell_ID".into(),
                    "relative_ID".into(),
                    "cell_cycle_stage".into(),
                    "relationship".into(),
                    "cell_area_um2".into(),
                ],
                rows: vec![
                    vec![
                        TableValue::Number(0.0),
                        TableValue::Number(1.0),
                        TableValue::Number(-1.0),
                        TableValue::Text("G1".into()),
                        TableValue::Text("mother".into()),
                        TableValue::Number(10.0),
                    ],
                    vec![
                        TableValue::Number(1.0),
                        TableValue::Number(1.0),
                        TableValue::Number(2.0),
                        TableValue::Text("S".into()),
                        TableValue::Text("mother".into()),
                        TableValue::Number(10.0),
                    ],
                    vec![
                        TableValue::Number(1.0),
                        TableValue::Number(2.0),
                        TableValue::Number(1.0),
                        TableValue::Text("S".into()),
                        TableValue::Text("bud".into()),
                        TableValue::Number(5.0),
                    ],
                ],
            },
        )?;
        generate_mother_bud_total(GenerateMotherBudTotalConfig {
            input_path: input,
            output_path: output.clone(),
            column_operation_mapper: BTreeMap::from([("cell_area_um2".into(), "sum".into())]),
            copy_all_nonselected_columns: true,
            grouping_columns: Vec::new(),
            entity_colname: "entity".into(),
        })?;
        let table = read_table(&output)?;
        assert!(table.rows.iter().any(|row| {
            row[table.header_index("entity").expect("entity")].as_string_lossy() == "Total"
        }));
        Ok(())
    }

    fn row(entries: Vec<(&str, RowValue)>) -> Row {
        let mut row = Row::new();
        for (key, value) in entries {
            row.insert(key.to_string(), value.0);
        }
        row
    }

    fn write_test_stack_u16(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frame_values {
            let data = vec![*value; 4];
            encoder.write_image::<colortype::Gray16>(2, 2, &data)?;
        }
        Ok(())
    }

    fn write_test_planes_u16(
        path: &Path,
        planes: &[Vec<u16>],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for plane in planes {
            encoder.write_image::<colortype::Gray16>(width as u32, height as u32, plane)?;
        }
        Ok(())
    }

    struct RowValue(TableValue);

    impl From<f64> for RowValue {
        fn from(value: f64) -> Self {
            Self(TableValue::Number(value))
        }
    }

    impl From<&str> for RowValue {
        fn from(value: &str) -> Self {
            Self(TableValue::Text(value.to_string()))
        }
    }
}
