mod gui;

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use cellacdc_rs::{
    add_lineage_tree, apply_tracking_from_table, apply_tracking_from_trackmate_xml,
    connect_3d_segm, count_objects, fill_holes, generate_mother_bud_total, run_workflow_file,
    ApplyTrackingConfig, ApplyTrackingFromTrackMateXmlConfig, Connect3DSegmConfig,
    CoordinateFilterConfig, CountObjectsConfig, FillHolesConfig, GenerateMotherBudTotalConfig,
    LineageTreeConfig, MaskPathResolution, SegmentationLayout, Stack2DSegmTo3DConfig,
    TrackingColumnMap, WorkflowRunOptions,
};

#[derive(Debug, Parser)]
#[command(name = "cellacdc-rs")]
#[command(about = "Cell-ACDC-compatible Rust GUI and workflow runner")]
#[command(disable_version_flag = true)]
struct Cli {
    #[arg(
        short = 'p',
        long = "params",
        value_name = "PATH_TO_PARAMS",
        help = "Path to a workflow INI file"
    )]
    params: Option<PathBuf>,
    #[arg(
        short = 'v',
        long = "version",
        action = ArgAction::SetTrue,
        help = "Print Cell-ACDC Rust version and environment info"
    )]
    version: bool,
    #[arg(
        long = "info",
        action = ArgAction::SetTrue,
        help = "Print Cell-ACDC Rust version and environment info. Alias: -info"
    )]
    info: bool,
    #[arg(
        long = "reset",
        action = ArgAction::SetTrue,
        help = "Reset the Python-compatible Cell-ACDC settings folder"
    )]
    reset: bool,
    #[arg(
        short = 'y',
        long = "yes",
        action = ArgAction::SetTrue,
        help = "Automatically confirm prompts where supported"
    )]
    yes: bool,
    #[arg(
        short = 'd',
        long = "debug",
        action = ArgAction::SetTrue,
        help = "Enable verbose workflow logging for params-file runs"
    )]
    debug: bool,
    #[arg(
        long = "install_details",
        value_name = "PATH_TO_INSTALL_DETAILS",
        help = "Path to install_details.json (Python installer compatibility flag; not used by Rust)"
    )]
    install_details: Option<PathBuf>,
    #[arg(
        long = "count_objects",
        action = ArgAction::SetTrue,
        help = "Count labels in a segmentation mask and write an objects-count CSV"
    )]
    count_objects: bool,
    #[arg(
        long = "fill_holes",
        action = ArgAction::SetTrue,
        help = "Fill holes in segmentation masks and write the corrected mask"
    )]
    fill_holes: bool,
    #[arg(
        long = "connect_3d_segm",
        action = ArgAction::SetTrue,
        help = "Connect labels across z-slice boundaries in 3D segmentation masks"
    )]
    connect_3d_segm: bool,
    #[arg(
        long = "stack_2d_segm_to_3d",
        action = ArgAction::SetTrue,
        help = "Broadcast 2D segmentation masks into a 3D z-stack"
    )]
    stack_2d_segm_to_3d: bool,
    #[arg(
        long = "filter_segm_from_table",
        action = ArgAction::SetTrue,
        help = "Keep only segmentation labels touched by coordinates in a table"
    )]
    filter_segm_from_table: bool,
    #[arg(
        long = "apply_tracking_from_table",
        action = ArgAction::SetTrue,
        help = "Apply tracking IDs from a table to a time-series segmentation mask"
    )]
    apply_tracking_from_table: bool,
    #[arg(
        long = "apply_tracking_from_trackmate_xml",
        action = ArgAction::SetTrue,
        help = "Apply tracking IDs from a TrackMate XML file to a position segmentation mask"
    )]
    apply_tracking_from_trackmate_xml: bool,
    #[arg(
        long = "add_lineage_tree",
        action = ArgAction::SetTrue,
        help = "Add lineage-tree columns to an acdc_output table"
    )]
    add_lineage_tree: bool,
    #[arg(
        long = "generate_mother_bud_total",
        action = ArgAction::SetTrue,
        help = "Generate G1/mother/bud/total rows from an acdc_output table"
    )]
    generate_mother_bud_total: bool,
    #[arg(
        long = "segmentation_path",
        value_name = "PATH_TO_SEGM",
        help = "Segmentation mask path for utility modes"
    )]
    segmentation_path: Option<PathBuf>,
    #[arg(
        long = "output_path",
        value_name = "PATH_TO_OUTPUT",
        help = "Output table or mask path for utility modes"
    )]
    output_path: Option<PathBuf>,
    #[arg(
        long = "input_path",
        value_name = "PATH_TO_INPUT",
        help = "Input table path for table utility modes"
    )]
    input_path: Option<PathBuf>,
    #[arg(
        long = "position_dir",
        value_name = "PATH_TO_POSITION",
        help = "Position folder path for position-scoped utility modes"
    )]
    position_dir: Option<PathBuf>,
    #[arg(
        long = "segm_endname",
        value_name = "ENDNAME",
        help = "Segmentation endname for position-scoped utility modes"
    )]
    segm_endname: Option<String>,
    #[arg(
        long = "xml_path",
        value_name = "PATH_TO_XML",
        help = "TrackMate XML path for --apply_tracking_from_trackmate_xml"
    )]
    xml_path: Option<PathBuf>,
    #[arg(
        long = "column_operation",
        value_name = "COLUMN=OPERATION",
        action = ArgAction::Append,
        help = "Column operation for --generate_mother_bud_total, for example cell_area_um2=sum"
    )]
    column_operations: Vec<String>,
    #[arg(
        long = "grouping_column",
        value_name = "COLUMN",
        action = ArgAction::Append,
        help = "Grouping column for --generate_mother_bud_total"
    )]
    grouping_columns: Vec<String>,
    #[arg(
        long = "entity_colname",
        value_name = "COLUMN",
        default_value = "entity",
        help = "Entity-label output column for --generate_mother_bud_total"
    )]
    entity_colname: String,
    #[arg(
        long = "no_copy_all_nonselected_columns",
        action = ArgAction::SetTrue,
        help = "Only keep columns named by --column_operation in --generate_mother_bud_total output"
    )]
    no_copy_all_nonselected_columns: bool,
    #[arg(
        long = "coords_table_path",
        value_name = "PATH_TO_COORDS_TABLE",
        help = "Coordinate table path for --filter_segm_from_table"
    )]
    coords_table_path: Option<PathBuf>,
    #[arg(
        long = "x_col",
        value_name = "COLUMN",
        default_value = "x",
        help = "X-coordinate column for --filter_segm_from_table"
    )]
    x_col: String,
    #[arg(
        long = "y_col",
        value_name = "COLUMN",
        default_value = "y",
        help = "Y-coordinate column for --filter_segm_from_table"
    )]
    y_col: String,
    #[arg(
        long = "z_col",
        value_name = "COLUMN",
        help = "Z-coordinate column for 3D --filter_segm_from_table"
    )]
    z_col: Option<String>,
    #[arg(
        long = "frame_col",
        value_name = "COLUMN",
        help = "Frame-index column for time-series --filter_segm_from_table"
    )]
    frame_col: Option<String>,
    #[arg(
        long = "position_col",
        value_name = "COLUMN",
        help = "Position column used to subset --filter_segm_from_table coordinates"
    )]
    position_col: Option<String>,
    #[arg(
        long = "position_value",
        value_name = "VALUE",
        help = "Position value used to subset --filter_segm_from_table coordinates"
    )]
    position_value: Option<String>,
    #[arg(
        long = "tracking_table_path",
        value_name = "PATH_TO_TRACKING_TABLE",
        help = "Tracking table path for --apply_tracking_from_table"
    )]
    tracking_table_path: Option<PathBuf>,
    #[arg(
        long = "frame_index_col",
        value_name = "COLUMN",
        default_value = "frame_i",
        help = "Frame-index column for tracking-table utilities"
    )]
    frame_index_col: String,
    #[arg(
        long = "first_frame_one",
        action = ArgAction::SetTrue,
        help = "Treat tracking table frame indices as one-based"
    )]
    first_frame_one: bool,
    #[arg(
        long = "track_ids_col",
        value_name = "COLUMN",
        default_value = "track_id",
        help = "Tracking-ID column for --apply_tracking_from_table"
    )]
    track_ids_col: String,
    #[arg(
        long = "mask_ids_col",
        value_name = "COLUMN",
        help = "Mask-label column for --apply_tracking_from_table"
    )]
    mask_ids_col: Option<String>,
    #[arg(
        long = "x_centroid_col",
        value_name = "COLUMN",
        help = "X-centroid column used to resolve mask labels for tracking"
    )]
    x_centroid_col: Option<String>,
    #[arg(
        long = "y_centroid_col",
        value_name = "COLUMN",
        help = "Y-centroid column used to resolve mask labels for tracking"
    )]
    y_centroid_col: Option<String>,
    #[arg(
        long = "z_centroid_col",
        value_name = "COLUMN",
        help = "Z-centroid column used to resolve 3D mask labels for tracking"
    )]
    z_centroid_col: Option<String>,
    #[arg(
        long = "delete_untracked_ids",
        action = ArgAction::SetTrue,
        help = "Delete segmentation labels not represented in the tracking table"
    )]
    delete_untracked_ids: bool,
    #[arg(
        long = "source_acdc_output_path",
        value_name = "PATH_TO_ACDC_OUTPUT",
        help = "Optional source acdc_output table to remap with tracking"
    )]
    source_acdc_output_path: Option<PathBuf>,
    #[arg(
        long = "output_acdc_output_path",
        value_name = "PATH_TO_OUTPUT_ACDC",
        help = "Optional output acdc_output path for remapped tracking metadata"
    )]
    output_acdc_output_path: Option<PathBuf>,
    #[arg(
        long = "size_t",
        value_name = "SIZE_T",
        help = "Metadata SizeT for resolving ambiguous segmentation layouts"
    )]
    size_t: Option<usize>,
    #[arg(
        long = "size_z",
        value_name = "SIZE_Z",
        help = "Metadata SizeZ for resolving ambiguous segmentation layouts"
    )]
    size_z: Option<usize>,
    #[arg(
        long = "segm_layout",
        value_name = "YX|TYX|ZYX|TZYX",
        value_parser = parse_segmentation_layout,
        help = "Explicit segmentation layout for utility modes"
    )]
    segm_layout: Option<SegmentationLayout>,
    #[arg(long = "cpModelsDownload", action = ArgAction::SetTrue, hide = true)]
    cp_models_download: bool,
    #[arg(long = "YeaZModelsDownload", action = ArgAction::SetTrue, hide = true)]
    yeaz_models_download: bool,
    #[arg(long = "DeepSeaModelsDownload", action = ArgAction::SetTrue, hide = true)]
    deepsea_models_download: bool,
    #[arg(long = "StarDistModelsDownload", action = ArgAction::SetTrue, hide = true)]
    stardist_models_download: bool,
    #[arg(long = "TrackastraModelsDownload", action = ArgAction::SetTrue, hide = true)]
    trackastra_models_download: bool,
    #[arg(long = "AllModelsDownload", action = ArgAction::SetTrue, hide = true)]
    all_models_download: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse_from(preprocess_args());
    let mode_count = usize::from(cli.params.is_some())
        + usize::from(cli.version || cli.info)
        + usize::from(cli.reset)
        + usize::from(cli.count_objects)
        + usize::from(cli.fill_holes)
        + usize::from(cli.connect_3d_segm)
        + usize::from(cli.stack_2d_segm_to_3d)
        + usize::from(cli.filter_segm_from_table)
        + usize::from(cli.apply_tracking_from_table)
        + usize::from(cli.apply_tracking_from_trackmate_xml)
        + usize::from(cli.add_lineage_tree)
        + usize::from(cli.generate_mother_bud_total);
    if mode_count > 1 {
        bail!(
            "Use only one of --params, --version/--info, --reset, --count_objects, --fill_holes, --connect_3d_segm, --stack_2d_segm_to_3d, --filter_segm_from_table, --apply_tracking_from_table, --apply_tracking_from_trackmate_xml, --add_lineage_tree, or --generate_mother_bud_total"
        );
    }
    if cli.debug && cli.params.is_none() {
        bail!("--debug is only supported together with --params");
    }
    let _install_details = cli
        .install_details
        .as_deref()
        .map(load_install_details)
        .transpose()?;
    if cli.cp_models_download
        || cli.yeaz_models_download
        || cli.deepsea_models_download
        || cli.stardist_models_download
        || cli.trackastra_models_download
        || cli.all_models_download
    {
        bail!("Python model download flags are not supported by cellacdc-rs; provide an explicit model path in the workflow [rust_cli] section");
    }

    if cli.version || cli.info {
        println!("{}", build_info_text());
        return Ok(());
    }

    if cli.reset {
        println!("{}", reset_settings(cli.yes)?);
        return Ok(());
    }

    if cli.count_objects {
        println!("{}", run_count_objects(&cli)?);
        return Ok(());
    }

    if cli.fill_holes {
        println!("{}", run_fill_holes(&cli)?);
        return Ok(());
    }

    if cli.connect_3d_segm {
        println!("{}", run_connect_3d_segm(&cli)?);
        return Ok(());
    }

    if cli.stack_2d_segm_to_3d {
        println!("{}", run_stack_2d_segm_to_3d(&cli)?);
        return Ok(());
    }

    if cli.filter_segm_from_table {
        println!("{}", run_filter_segm_from_table(&cli)?);
        return Ok(());
    }

    if cli.apply_tracking_from_table {
        println!("{}", run_apply_tracking_from_table(&cli)?);
        return Ok(());
    }

    if cli.apply_tracking_from_trackmate_xml {
        println!("{}", run_apply_tracking_from_trackmate_xml(&cli)?);
        return Ok(());
    }

    if cli.add_lineage_tree {
        println!("{}", run_add_lineage_tree(&cli)?);
        return Ok(());
    }

    if cli.generate_mother_bud_total {
        println!("{}", run_generate_mother_bud_total(&cli)?);
        return Ok(());
    }

    reject_utility_args_without_mode(&cli)?;

    if let Some(params_path) = cli.params {
        let report = run_workflow_file(params_path, WorkflowRunOptions { debug: cli.debug })?;
        if !report.segmentation_results.is_empty() {
            println!(
                "Segmented {} position(s)",
                report.segmentation_results.len()
            );
        }
        if !report.measurement_results.is_empty() {
            println!("Measured {} position(s)", report.measurement_results.len());
        }
        return Ok(());
    }

    gui::launch_gui()
}

fn run_count_objects(cli: &Cli) -> Result<String> {
    let segmentation_path = cli
        .segmentation_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--count_objects requires --segmentation_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--count_objects requires --output_path"))?;
    let resolution = utility_mask_resolution(cli);
    let result = count_objects(CountObjectsConfig {
        segmentation_path,
        output_path,
        resolution,
    })?;
    let mut lines = vec![format!(
        "Saved object counts table to {}",
        result.summary.output_path.display()
    )];
    for (name, value) in result.summary.counts {
        lines.push(format!("{name}: {value}"));
    }
    Ok(lines.join("\n"))
}

fn run_fill_holes(cli: &Cli) -> Result<String> {
    let segmentation_path = cli
        .segmentation_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--fill_holes requires --segmentation_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--fill_holes requires --output_path"))?;
    let result = fill_holes(FillHolesConfig {
        segmentation_path,
        output_path,
        resolution: utility_mask_resolution(cli),
    })?;
    Ok(format!(
        "Saved hole-filled segmentation mask to {}",
        result.primary_path.display()
    ))
}

fn run_connect_3d_segm(cli: &Cli) -> Result<String> {
    let segmentation_path = cli
        .segmentation_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--connect_3d_segm requires --segmentation_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--connect_3d_segm requires --output_path"))?;
    let result = connect_3d_segm(Connect3DSegmConfig {
        segmentation_path,
        output_path,
        resolution: utility_mask_resolution(cli),
    })?;
    Ok(format!(
        "Saved 3D-connected segmentation mask to {}",
        result.primary_path.display()
    ))
}

fn run_stack_2d_segm_to_3d(cli: &Cli) -> Result<String> {
    let segmentation_path = cli
        .segmentation_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--stack_2d_segm_to_3d requires --segmentation_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--stack_2d_segm_to_3d requires --output_path"))?;
    let size_z = cli
        .size_z
        .ok_or_else(|| anyhow::anyhow!("--stack_2d_segm_to_3d requires --size_z"))?;
    let result = cellacdc_rs::stack_2d_segm_to_3d(Stack2DSegmTo3DConfig {
        segmentation_path,
        output_path,
        size_z,
        resolution: utility_mask_resolution(cli),
    })?;
    Ok(format!(
        "Saved 2D segmentation mask stacked to 3D at {}",
        result.primary_path.display()
    ))
}

fn run_filter_segm_from_table(cli: &Cli) -> Result<String> {
    let segmentation_path = cli
        .segmentation_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--filter_segm_from_table requires --segmentation_path"))?;
    let coords_table_path = cli
        .coords_table_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--filter_segm_from_table requires --coords_table_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--filter_segm_from_table requires --output_path"))?;
    let result = cellacdc_rs::filter_segm_from_table(CoordinateFilterConfig {
        segmentation_path,
        coords_table_path,
        output_path,
        x_col: cli.x_col.clone(),
        y_col: cli.y_col.clone(),
        z_col: cli.z_col.clone(),
        frame_col: cli.frame_col.clone(),
        position_col: cli.position_col.clone(),
        position_value: cli.position_value.clone(),
        resolution: utility_mask_resolution(cli),
    })?;
    Ok(format!(
        "Saved coordinate-filtered segmentation mask to {}",
        result.primary_path.display()
    ))
}

fn run_apply_tracking_from_table(cli: &Cli) -> Result<String> {
    let segmentation_path = cli.segmentation_path.clone().ok_or_else(|| {
        anyhow::anyhow!("--apply_tracking_from_table requires --segmentation_path")
    })?;
    let tracking_table_path = cli.tracking_table_path.clone().ok_or_else(|| {
        anyhow::anyhow!("--apply_tracking_from_table requires --tracking_table_path")
    })?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--apply_tracking_from_table requires --output_path"))?;
    let result = apply_tracking_from_table(ApplyTrackingConfig {
        segmentation_path,
        tracking_table_path,
        output_path,
        columns: TrackingColumnMap {
            frame_index_col: cli.frame_index_col.clone(),
            is_first_frame_one: cli.first_frame_one,
            track_ids_col: cli.track_ids_col.clone(),
            mask_ids_col: cli.mask_ids_col.clone(),
            x_centroid_col: cli.x_centroid_col.clone(),
            y_centroid_col: cli.y_centroid_col.clone(),
            z_centroid_col: cli.z_centroid_col.clone(),
            delete_untracked_ids: cli.delete_untracked_ids,
        },
        resolution: utility_mask_resolution(cli),
        source_acdc_output_path: cli.source_acdc_output_path.clone(),
        output_acdc_output_path: cli.output_acdc_output_path.clone(),
    })?;
    let mut lines = vec![format!(
        "Saved tracked segmentation mask to {}",
        result.primary_path.display()
    )];
    for path in result.secondary_paths {
        lines.push(format!("Saved tracking sidecar to {}", path.display()));
    }
    Ok(lines.join("\n"))
}

fn run_apply_tracking_from_trackmate_xml(cli: &Cli) -> Result<String> {
    let position_dir = cli.position_dir.clone().ok_or_else(|| {
        anyhow::anyhow!("--apply_tracking_from_trackmate_xml requires --position_dir")
    })?;
    let segm_endname = cli.segm_endname.clone().ok_or_else(|| {
        anyhow::anyhow!("--apply_tracking_from_trackmate_xml requires --segm_endname")
    })?;
    let xml_path = cli.xml_path.clone().ok_or_else(|| {
        anyhow::anyhow!("--apply_tracking_from_trackmate_xml requires --xml_path")
    })?;
    let result = apply_tracking_from_trackmate_xml(ApplyTrackingFromTrackMateXmlConfig {
        position_dir,
        segm_endname,
        xml_path,
        output_segmentation_path: cli.output_path.clone(),
        source_acdc_output_path: cli.source_acdc_output_path.clone(),
        output_acdc_output_path: cli.output_acdc_output_path.clone(),
        delete_untracked_ids: cli.delete_untracked_ids,
    })?;
    let mut lines = vec![format!(
        "Saved TrackMate-tracked segmentation mask to {}",
        result.primary_path.display()
    )];
    for path in result.secondary_paths {
        lines.push(format!("Saved tracking sidecar to {}", path.display()));
    }
    Ok(lines.join("\n"))
}

fn run_add_lineage_tree(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--add_lineage_tree requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--add_lineage_tree requires --output_path"))?;
    let result = add_lineage_tree(LineageTreeConfig {
        input_path,
        output_path,
    })?;
    Ok(format!(
        "Saved lineage-tree table to {}",
        result.primary_path.display()
    ))
}

fn run_generate_mother_bud_total(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--generate_mother_bud_total requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--generate_mother_bud_total requires --output_path"))?;
    let result = generate_mother_bud_total(GenerateMotherBudTotalConfig {
        input_path,
        output_path,
        column_operation_mapper: parse_column_operations(&cli.column_operations)?,
        copy_all_nonselected_columns: !cli.no_copy_all_nonselected_columns,
        grouping_columns: cli.grouping_columns.clone(),
        entity_colname: cli.entity_colname.clone(),
    })?;
    Ok(format!(
        "Saved mother-bud-total table to {}",
        result.primary_path.display()
    ))
}

fn parse_column_operations(values: &[String]) -> Result<BTreeMap<String, String>> {
    let mut mapper = BTreeMap::new();
    for value in values {
        let Some((column, operation)) = value.split_once('=') else {
            bail!("--column_operation must use COLUMN=OPERATION syntax");
        };
        let column = column.trim();
        let operation = operation.trim();
        if column.is_empty() || operation.is_empty() {
            bail!("--column_operation must use non-empty COLUMN=OPERATION values");
        }
        mapper.insert(column.to_string(), operation.to_string());
    }
    Ok(mapper)
}

fn utility_mask_resolution(cli: &Cli) -> Option<MaskPathResolution> {
    if cli.size_t.is_none() && cli.size_z.is_none() && cli.segm_layout.is_none() {
        return None;
    }
    Some(MaskPathResolution {
        size_t: cli.size_t,
        size_z: cli.size_z,
        layout: cli.segm_layout,
    })
}

fn reject_utility_args_without_mode(cli: &Cli) -> Result<()> {
    if cli.segmentation_path.is_some()
        || cli.output_path.is_some()
        || cli.input_path.is_some()
        || cli.position_dir.is_some()
        || cli.segm_endname.is_some()
        || cli.xml_path.is_some()
        || !cli.column_operations.is_empty()
        || !cli.grouping_columns.is_empty()
        || cli.entity_colname != "entity"
        || cli.no_copy_all_nonselected_columns
        || cli.size_t.is_some()
        || cli.size_z.is_some()
        || cli.segm_layout.is_some()
        || cli.coords_table_path.is_some()
        || cli.x_col != "x"
        || cli.y_col != "y"
        || cli.z_col.is_some()
        || cli.frame_col.is_some()
        || cli.position_col.is_some()
        || cli.position_value.is_some()
        || cli.tracking_table_path.is_some()
        || cli.frame_index_col != "frame_i"
        || cli.first_frame_one
        || cli.track_ids_col != "track_id"
        || cli.mask_ids_col.is_some()
        || cli.x_centroid_col.is_some()
        || cli.y_centroid_col.is_some()
        || cli.z_centroid_col.is_some()
        || cli.delete_untracked_ids
        || cli.source_acdc_output_path.is_some()
        || cli.output_acdc_output_path.is_some()
    {
        bail!("Utility path/layout flags require a utility mode such as --count_objects, --fill_holes, --connect_3d_segm, --stack_2d_segm_to_3d, --filter_segm_from_table, --apply_tracking_from_table, --apply_tracking_from_trackmate_xml, --add_lineage_tree, or --generate_mother_bud_total");
    }
    Ok(())
}

fn parse_segmentation_layout(value: &str) -> Result<SegmentationLayout, String> {
    match value.trim().to_ascii_uppercase().as_str() {
        "YX" => Ok(SegmentationLayout::YX),
        "TYX" => Ok(SegmentationLayout::TYX),
        "ZYX" => Ok(SegmentationLayout::ZYX),
        "TZYX" => Ok(SegmentationLayout::TZYX),
        _ => Err("expected one of YX, TYX, ZYX, or TZYX".to_string()),
    }
}

fn preprocess_args() -> Vec<OsString> {
    std::env::args_os()
        .map(|arg| {
            if arg == OsString::from("-info") {
                OsString::from("--info")
            } else {
                arg
            }
        })
        .collect()
}

fn build_info_text() -> String {
    let user_profile = user_profile_dir();
    let settings_dir = python_compatible_settings_dir();
    let working_dir = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "Unknown".to_string());
    format!(
        "cellacdc-rs {}\nInstalled in: {}\nUser profile folder: {}\nSettings folder: {}\nOS: {}\nARCH: {}\nProfile: {}\nWorking directory: {}",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_MANIFEST_DIR"),
        user_profile.display(),
        settings_dir.display(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        option_env!("PROFILE").unwrap_or("release"),
        working_dir,
    )
}

fn load_install_details(path: &Path) -> Result<serde_json::Value> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Failed to read install details file {}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse install details file {}", path.display()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Install details file must contain a JSON object"))?;

    for key in ["conda_path", "clone_path", "venv_path", "target_dir"] {
        if let Some(entry) = object.get_mut(key) {
            if let Some(path_text) = entry.as_str() {
                let path = PathBuf::from(path_text);
                let absolute = if path.is_absolute() {
                    path
                } else {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join(path)
                };
                *entry = serde_json::Value::String(absolute.display().to_string());
            }
        }
    }

    Ok(value)
}

fn reset_settings(auto_confirm: bool) -> Result<String> {
    let settings_dir = python_compatible_settings_dir();
    if !auto_confirm && !confirm_reset(&settings_dir)? {
        return Ok("Resetting Cell-ACDC settings cancelled.".to_string());
    }

    if settings_dir.is_dir() {
        fs::remove_dir_all(&settings_dir)?;
    } else if settings_dir.is_file() {
        fs::remove_file(&settings_dir)?;
    } else {
        return Ok(format!(
            "Cell-ACDC settings folder was not found.\n\nSettings folder path:\n{}",
            settings_dir.display()
        ));
    }

    Ok(format!(
        "Cell-ACDC settings have been reset.\n\nThe following folder was deleted:\n\n{}",
        settings_dir.display()
    ))
}

fn confirm_reset(settings_dir: &PathBuf) -> Result<bool> {
    let info_text = format!(
        "If you reset Cell-ACDC settings, the folder below will be deleted.\n\n\
This means deleting things like custom shortcuts, recent paths, last selections, \
and GUI preferences.\n\nSettings folder path: \"{}\"",
        settings_dir.display()
    );

    loop {
        print!("\nDo you want to reset Cell-ACDC settings - type \"h\" for help - (y/[n]/h)? ");
        io::stdout().flush()?;

        let mut answer = String::new();
        if io::stdin().read_line(&mut answer)? == 0 {
            return Ok(false);
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "" | "n" | "no" => return Ok(false),
            "y" | "yes" => return Ok(true),
            "h" | "help" => {
                println!("{}\n{}", "-".repeat(100), info_text);
                println!("{}", "=".repeat(100));
            }
            other => {
                println!(
                    "\"{other}\" is not a valid answer. Type \"y\" for \"yes\", \"n\" for \"no\", or \"h\" for help."
                );
            }
        }
    }
}

fn python_compatible_settings_dir() -> PathBuf {
    user_profile_dir().join(".acdc-settings")
}

fn user_profile_dir() -> PathBuf {
    let data_dir = cell_acdc_user_data_dir();
    let profile_pointer = data_dir.join("acdc_user_profile_location.txt");
    if let Ok(text) = fs::read_to_string(profile_pointer) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return expand_home(trimmed);
        }
    }

    home_dir()
        .map(|home| home.join("acdc-appdata"))
        .unwrap_or_else(|| PathBuf::from("acdc-appdata"))
}

fn cell_acdc_user_data_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().unwrap_or_else(|| PathBuf::from(".")))
            .join("Cell_ACDC")
    } else if cfg!(target_os = "macos") {
        home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Library")
            .join("Application Support")
            .join("Cell_ACDC")
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".local")
                    .join("share")
            })
            .join("Cell_ACDC")
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

fn expand_home(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}
