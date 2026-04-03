use crate::mask_io::{
    load_mask_data, save_mask_data, MaskData, MaskPathResolution, SegmentationLayout,
};
use crate::tabular::{read_table, write_table, Table, TableFormat, TableValue};
use crate::zstack::{connect_3d_lab_z_boundaries, stack_2d_lab_to_3d};
use anyhow::{anyhow, bail, Context, Result};
use evalexpr::{ContextWithMutableVariables, HashMapContext, Value as EvalValue};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_INDEX_COLS: &[&str] = &[
    "experiment_folderpath",
    "experiment_foldername",
    "Position_n",
    "frame_i",
    "Cell_ID",
];
const LINEAGE_TREE_COLS: &[&str] = &[
    "Cell_ID_tree",
    "generation_num_tree",
    "parent_ID_tree",
    "root_ID_tree",
    "sister_ID_tree",
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
pub struct CountObjectsConfig {
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
pub struct LineageTreeConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
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
                let alias = metric_alias(table_idx + 1, header);
                let value = row
                    .and_then(|row| row.get(header))
                    .and_then(TableValue::as_f64)
                    .unwrap_or(f64::NAN);
                context.set_value(alias, EvalValue::Float(value))?;
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
            let connected =
                connect_3d_lab_z_boundaries(&values, shape[0], shape[1], shape[2]);
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
    let mut tracked = masks.clone();
    let (tracked_ids_mapper, deleted_ids_mapper) =
        apply_tracking_to_mask_data(&mut tracked, &tracking_table, &config.columns)?;
    save_mask_data(&config.output_path, &tracked)?;

    let mapper_base = config.output_path.with_extension("");
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
    if let Some(source_path) = config.source_acdc_output_path.as_ref() {
        if source_path.exists() {
            let table = read_table(source_path)?;
            let remapped = apply_tracked_ids_mapper_to_table(
                &table,
                &tracked_ids_mapper,
                &deleted_ids_mapper,
            )?;
            let acdc_output_path = config
                .output_acdc_output_path
                .clone()
                .unwrap_or_else(|| derive_acdc_output_path(source_path, &config.output_path));
            write_table(&acdc_output_path, &remapped)?;
            secondary_paths.push(acdc_output_path);
        }
    }

    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths,
    })
}

pub fn add_lineage_tree(config: LineageTreeConfig) -> Result<UtilityOutputPaths> {
    let table = read_table(&config.input_path)?;
    let mut rows = table_to_rows(&table);
    ensure_required_columns(&table, REQUIRED_LINEAGE_COLS)?;
    add_lineage_tree_columns(&mut rows)?;
    let mut headers = table.headers.clone();
    for column in LINEAGE_TREE_COLS {
        if !headers.iter().any(|header| header == column) {
            headers.push((*column).to_string());
        }
    }
    write_table(&config.output_path, &rows_to_table(&headers, &rows))?;
    Ok(UtilityOutputPaths {
        primary_path: config.output_path,
        secondary_paths: Vec::new(),
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

fn objects_count_summary(masks: &MaskData) -> BTreeMap<String, usize> {
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

fn add_lineage_tree_columns(rows: &mut [Row]) -> Result<()> {
    let indexed = rows
        .iter()
        .enumerate()
        .map(|(idx, row)| Ok((row_key(row)?, idx)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut g1_indices = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            row.get("cell_cycle_stage")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("G1")
        })
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    g1_indices.sort_by_key(|idx| row_key(&rows[*idx]).expect("sorted"));
    if g1_indices.is_empty() {
        bail!("Lineage tree generation requires at least one G1 row");
    }

    for idx in &g1_indices {
        let cell_id = get_required_i64(&rows[*idx], "Cell_ID")? as i64;
        let generation_num = get_required_i64(&rows[*idx], "generation_num")?;
        rows[*idx].insert("Cell_ID_tree".into(), TableValue::Number(cell_id as f64));
        rows[*idx].insert("parent_ID_tree".into(), TableValue::Number(-1.0));
        rows[*idx].insert("root_ID_tree".into(), TableValue::Number(-1.0));
        rows[*idx].insert("sister_ID_tree".into(), TableValue::Number(-1.0));
        let mut generation_num_tree = generation_num;
        if get_required_bool(&rows[*idx], "is_history_known")? == false && generation_num > 1 {
            generation_num_tree -= 1;
        }
        rows[*idx].insert(
            "generation_num_tree".into(),
            TableValue::Number(generation_num_tree as f64),
        );
    }

    let mut unique_id = rows
        .iter()
        .filter_map(|row| row.get("Cell_ID").and_then(TableValue::as_i64))
        .max()
        .unwrap_or(0)
        + 1;
    let mut not_annotated_ids = g1_indices
        .iter()
        .filter_map(|idx| rows[*idx].get("Cell_ID").and_then(TableValue::as_i64))
        .collect::<BTreeSet<_>>();
    let mut branch_start_gen_num = BTreeMap::<i64, i64>::new();
    let mut root_ids_trees = BTreeMap::<i64, i64>::new();
    let mut gen_groups = BTreeMap::<(i64, i64), Vec<usize>>::new();
    for idx in &g1_indices {
        let cell_id = get_required_i64(&rows[*idx], "Cell_ID")?;
        let generation_num = get_required_i64(&rows[*idx], "generation_num")?;
        gen_groups
            .entry((cell_id, generation_num))
            .or_default()
            .push(*idx);
    }
    for group in gen_groups.values_mut() {
        group.sort_by_key(|idx| row_key(&rows[*idx]).expect("valid"));
    }
    let mut gen_groups_by_tree = BTreeMap::<i64, Vec<usize>>::new();
    let frame_order = g1_indices
        .iter()
        .map(|idx| get_required_i64(&rows[*idx], "frame_i").expect("frame"))
        .collect::<BTreeSet<_>>();
    let mut built_groups = BTreeMap::<(i64, i64), Vec<usize>>::new();

    for frame_i in frame_order {
        let ids = g1_indices
            .iter()
            .filter(|idx| get_required_i64(&rows[**idx], "frame_i").ok() == Some(frame_i))
            .filter_map(|idx| rows[*idx].get("Cell_ID").and_then(TableValue::as_i64))
            .collect::<BTreeSet<_>>();
        for id in ids {
            if !not_annotated_ids.contains(&id) {
                continue;
            }
            let mut is_new_tree = true;
            let groups = gen_groups
                .iter()
                .filter(|((cell_id, _), _)| *cell_id == id)
                .map(|(key, value)| (*key, value.clone()))
                .collect::<Vec<_>>();
            for ((_, _generation_num), group_indices) in groups {
                let rel_id = get_required_i64(&rows[group_indices[0]], "relative_ID")?;
                let start_frame = get_required_i64(&rows[group_indices[0]], "frame_i")?;
                let gen_num_tree_base =
                    get_required_i64(&rows[group_indices[0]], "generation_num_tree")?;
                let gen_num_rel_id_tree = if is_new_tree {
                    indexed
                        .get(&(start_frame, rel_id))
                        .and_then(|idx| {
                            rows[*idx]
                                .get("generation_num_tree")
                                .and_then(TableValue::as_i64)
                        })
                        .map(|value| value - 1)
                        .unwrap_or(0)
                } else {
                    *branch_start_gen_num.get(&id).unwrap_or(&0)
                };
                branch_start_gen_num.insert(id, gen_num_rel_id_tree);
                let gen_num_tree = gen_num_tree_base + gen_num_rel_id_tree;
                for idx in &group_indices {
                    rows[*idx].insert(
                        "generation_num_tree".into(),
                        TableValue::Number(gen_num_tree as f64),
                    );
                }

                let mut cell_id_tree = if is_new_tree { id } else { unique_id };
                if !is_new_tree {
                    unique_id += 1;
                }
                let mut parent_id = -1i64;
                let mut prev_gen_exists = false;
                if gen_num_tree > 1 {
                    let prev_gen_num_tree = gen_num_tree - 1;
                    let prev_group = built_groups
                        .get(&(id, prev_gen_num_tree))
                        .or_else(|| built_groups.get(&(rel_id, prev_gen_num_tree)));
                    if let Some(prev_group) = prev_group {
                        prev_gen_exists = true;
                        let parent_row = prev_group
                            .iter()
                            .find(|idx| {
                                get_required_i64(&rows[**idx], "Cell_ID").ok() == Some(rel_id)
                            })
                            .or_else(|| prev_group.first())
                            .copied()
                            .ok_or_else(|| anyhow!("Missing parent lineage group"))?;
                        parent_id = get_required_i64(&rows[parent_row], "Cell_ID_tree")?;
                    } else {
                        let prior_row = indexed.get(&(start_frame - 1, id)).copied();
                        let was_bud = prior_row
                            .map(|idx| {
                                rows[idx]
                                    .get("relationship")
                                    .map(TableValue::as_string_lossy)
                                    .unwrap_or_default()
                                    == "bud"
                            })
                            .unwrap_or(false);
                        if was_bud {
                            parent_id = prior_row
                                .and_then(|idx| {
                                    rows[idx].get("relative_ID").and_then(TableValue::as_i64)
                                })
                                .unwrap_or(id);
                            let branch_base = branch_start_gen_num
                                .get(&parent_id)
                                .copied()
                                .map(|value| value + 2)
                                .unwrap_or(2);
                            branch_start_gen_num.insert(id, branch_base);
                        } else {
                            parent_id = id;
                        }
                        cell_id_tree = unique_id;
                        unique_id += 1;
                    }
                }
                let root_id = if is_new_tree {
                    if gen_num_tree == 2 && prev_gen_exists && parent_id > 0 {
                        parent_id
                    } else if parent_id > 0 {
                        gen_groups_by_tree
                            .get(&parent_id)
                            .and_then(|indices| indices.first())
                            .and_then(|idx| {
                                rows[*idx].get("root_ID_tree").and_then(TableValue::as_i64)
                            })
                            .filter(|value| *value > 0)
                            .unwrap_or(parent_id)
                    } else {
                        id
                    }
                } else {
                    *root_ids_trees.get(&id).unwrap_or(&id)
                };
                root_ids_trees.insert(id, root_id);
                for idx in &group_indices {
                    rows[*idx].insert(
                        "Cell_ID_tree".into(),
                        TableValue::Number(cell_id_tree as f64),
                    );
                    rows[*idx].insert(
                        "parent_ID_tree".into(),
                        TableValue::Number(parent_id as f64),
                    );
                    rows[*idx].insert("root_ID_tree".into(), TableValue::Number(root_id as f64));
                }
                built_groups.insert((id, gen_num_tree), group_indices.clone());
                gen_groups_by_tree.insert(cell_id_tree, group_indices.clone());
                is_new_tree = false;
            }
            not_annotated_ids.remove(&id);
        }
    }

    let grouped_by_tree =
        g1_indices
            .iter()
            .fold(BTreeMap::<i64, Vec<usize>>::new(), |mut acc, idx| {
                if let Some(tree_id) = rows[*idx].get("Cell_ID_tree").and_then(TableValue::as_i64) {
                    acc.entry(tree_id).or_default().push(*idx);
                }
                acc
            });
    for indices in grouped_by_tree.values() {
        let relative_id = get_required_i64(&rows[indices[0]], "relative_ID")?;
        if relative_id == -1 {
            continue;
        }
        let start_frame = get_required_i64(&rows[indices[0]], "frame_i")?;
        let sister_idx = indexed
            .get(&(start_frame, relative_id))
            .copied()
            .ok_or_else(|| anyhow!("Failed to resolve sister lineage row for relative_ID {relative_id} at frame {start_frame}"))?;
        let sister_tree_id = get_required_i64(&rows[sister_idx], "Cell_ID_tree")?;
        for idx in indices {
            rows[*idx].insert(
                "sister_ID_tree".into(),
                TableValue::Number(sister_tree_id as f64),
            );
        }
    }

    let g1_lookup = g1_indices
        .iter()
        .map(|idx| {
            let row = &rows[*idx];
            (
                (
                    get_required_i64(row, "Cell_ID").expect("cell"),
                    get_required_i64(row, "generation_num").expect("generation"),
                ),
                *idx,
            )
        })
        .collect::<BTreeMap<_, _>>();

    let s_indices = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            row.get("cell_cycle_stage")
                .map(TableValue::as_string_lossy)
                .as_deref()
                == Some("S")
        })
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    for idx in s_indices {
        let frame_i = get_required_i64(&rows[idx], "frame_i")?;
        let relationship = rows[idx]
            .get("relationship")
            .map(TableValue::as_string_lossy)
            .unwrap_or_default();
        let idx_id = if relationship == "mother" {
            get_required_i64(&rows[idx], "Cell_ID")?
        } else {
            get_required_i64(&rows[idx], "relative_ID")?
        };
        let idx_generation = if relationship == "mother" {
            get_required_i64(&rows[idx], "generation_num")?
        } else {
            indexed
                .get(&(frame_i, idx_id))
                .and_then(|row_index| {
                    rows[*row_index]
                        .get("generation_num")
                        .and_then(TableValue::as_i64)
                })
                .unwrap_or(1)
        };
        if let Some(g1_idx) = g1_lookup.get(&(idx_id, idx_generation)).copied() {
            let values = LINEAGE_TREE_COLS
                .iter()
                .filter_map(|column| {
                    rows[g1_idx]
                        .get(*column)
                        .cloned()
                        .map(|value| ((*column).to_string(), value))
                })
                .collect::<Vec<_>>();
            for (column, value) in values {
                rows[idx].insert(column, value);
            }
        } else {
            let sister_id = rows[idx]
                .get("relative_ID")
                .and_then(TableValue::as_i64)
                .unwrap_or(-1);
            rows[idx].insert("Cell_ID_tree".into(), TableValue::Number(idx_id as f64));
            rows[idx].insert("parent_ID_tree".into(), TableValue::Number(-1.0));
            rows[idx].insert("root_ID_tree".into(), TableValue::Number(idx_id as f64));
            rows[idx].insert("generation_num_tree".into(), TableValue::Number(1.0));
            rows[idx].insert(
                "sister_ID_tree".into(),
                TableValue::Number(sister_id as f64),
            );
        }
    }

    for row in rows.iter_mut() {
        for column in LINEAGE_TREE_COLS {
            row.entry((*column).to_string())
                .or_insert(TableValue::Number(-1.0));
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};
    use tempfile::tempdir;

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
    fn adds_lineage_tree_columns() -> Result<()> {
        let mut rows = vec![
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
        ];
        add_lineage_tree_columns(&mut rows)?;
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
