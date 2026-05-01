mod gui;

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use cellacdc_rs::{
    add_lineage_tree, add_lineage_tree_to_tables, apply_alignment, apply_tracking_from_table,
    apply_tracking_from_trackmate_xml, combine_channels, combine_metrics, compute_alignment_shifts,
    compute_background_roi_archives, compute_multi_channel, concat_acdc_outputs, connect_3d_segm,
    connect_3d_segm_in_positions, convert_file_format, count_objects, count_objects_in_positions,
    discover_measurement_experiment, export_lineage_info_file, fill_holes, fill_holes_in_positions,
    filter_segm_from_table_in_positions, generate_mother_bud_total, images_to_positions,
    measure_experiment, measure_position, move_channel_tiffs_to_positions,
    prepare_zstack_segm_info, propagate_lineage_file, read_background_roi_json, rename_files,
    resolve_measurement_position, run_workflow_file, segmentation_to_object_coords,
    segmentation_to_object_coords_in_positions, stack_2d_segm_to_3d_in_positions,
    update_lineage_frame_file, AlignmentRunConfig, ApplyTrackingConfig,
    ApplyTrackingFromTrackMateXmlConfig, CombineChannelsConfig, CombineMetricsConfig,
    ComputeMultiChannelConfig, ConcatConfig, Connect3DSegmBatchConfig, Connect3DSegmConfig,
    ConvertFileFormatConfig, CoordinateFilterBatchConfig, CoordinateFilterConfig,
    CountObjectsBatchConfig, CountObjectsConfig, FillHolesBatchConfig, FillHolesConfig,
    GenerateMotherBudTotalConfig, ImagesToPositionsConfig, LineageInfoConfig,
    LineagePropagateConfig, LineageTreeBatchConfig, LineageTreeConfig, LineageUpdateConfig,
    MaskPathResolution, MeasurementExperimentConfig, MeasurementRunConfig, MoveChannelTiffsConfig,
    ObjectCoordinatesBatchConfig, ObjectCoordinatesConfig, OverwritePolicy, PrepareSegmInfoTarget,
    PrepareZStackSegmInfoConfig, RenameFilesConfig, SegmentationLayout, Stack2DSegmTo3DBatchConfig,
    Stack2DSegmTo3DConfig, TableFormat, TrackingColumnMap, WorkflowRunOptions,
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
        long = "to_obj_coords",
        action = ArgAction::SetTrue,
        help = "Convert segmentation labels to an object-coordinate CSV/XLSX table"
    )]
    to_obj_coords: bool,
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
        long = "align_frames",
        action = ArgAction::SetTrue,
        help = "Align position image frames and write aligned channel NPZ files"
    )]
    align_frames: bool,
    #[arg(
        long = "measure",
        action = ArgAction::SetTrue,
        help = "Compute Cell-ACDC measurements for a position or experiment"
    )]
    measure: bool,
    #[arg(
        long = "prepare_zstack_segm_info",
        action = ArgAction::SetTrue,
        help = "Write default z-stack segmInfo.csv files for positions"
    )]
    prepare_zstack_segm_info: bool,
    #[arg(
        long = "compute_background_roi_data",
        action = ArgAction::SetTrue,
        help = "Write background ROI data archives from Data Prep ROI sidecars"
    )]
    compute_background_roi_data: bool,
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
        long = "export_lineage_info",
        action = ArgAction::SetTrue,
        help = "Export new, orphan, and lost lineage cell info for one frame"
    )]
    export_lineage_info: bool,
    #[arg(
        long = "propagate_lineage",
        action = ArgAction::SetTrue,
        help = "Propagate lineage edits from one frame to future frames"
    )]
    propagate_lineage: bool,
    #[arg(
        long = "update_lineage_frame",
        action = ArgAction::SetTrue,
        help = "Apply lineage edits to one frame of an acdc_output table"
    )]
    update_lineage_frame: bool,
    #[arg(
        long = "generate_mother_bud_total",
        action = ArgAction::SetTrue,
        help = "Generate G1/mother/bud/total rows from an acdc_output table"
    )]
    generate_mother_bud_total: bool,
    #[arg(
        long = "combine_metrics",
        action = ArgAction::SetTrue,
        help = "Combine metrics from two or more tables using formulas"
    )]
    combine_metrics: bool,
    #[arg(
        long = "compute_multi_channel",
        action = ArgAction::SetTrue,
        help = "Compute combined multi-channel metric tables for a position or experiment"
    )]
    compute_multi_channel: bool,
    #[arg(
        long = "concat_acdc_outputs",
        action = ArgAction::SetTrue,
        help = "Concatenate acdc_output tables across Cell-ACDC positions and experiments"
    )]
    concat_acdc_outputs: bool,
    #[arg(
        long = "combine_channels",
        action = ArgAction::SetTrue,
        help = "Combine raw/segmentation channels from a JSON recipe"
    )]
    combine_channels: bool,
    #[arg(
        long = "convert_file_format",
        action = ArgAction::SetTrue,
        help = "Convert an image/array file between Cell-ACDC-compatible formats"
    )]
    convert_file_format: bool,
    #[arg(
        long = "rename_files",
        action = ArgAction::SetTrue,
        help = "Append text to one or more filenames"
    )]
    rename_files: bool,
    #[arg(
        long = "images_to_positions",
        action = ArgAction::SetTrue,
        help = "Convert a flat image folder into Cell-ACDC Position_n folders"
    )]
    images_to_positions: bool,
    #[arg(
        long = "move_channel_tiffs_to_positions",
        action = ArgAction::SetTrue,
        help = "Move separate channel TIFF files into Cell-ACDC Position_n folders"
    )]
    move_channel_tiffs_to_positions: bool,
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
        long = "experiment_dir",
        value_name = "PATH_TO_EXPERIMENT",
        help = "Experiment folder path for experiment-scoped utility modes"
    )]
    experiment_dir: Option<PathBuf>,
    #[arg(
        long = "concat_experiment_dir",
        value_name = "PATH_TO_EXPERIMENT",
        action = ArgAction::Append,
        help = "Experiment folder path for --concat_acdc_outputs; repeat for multi-experiment output"
    )]
    concat_experiment_dirs: Vec<PathBuf>,
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
        long = "recipe_path",
        value_name = "PATH_TO_JSON",
        help = "JSON recipe path for --combine_channels"
    )]
    recipe_path: Option<PathBuf>,
    #[arg(
        long = "column_operation",
        value_name = "COLUMN=OPERATION",
        action = ArgAction::Append,
        help = "Column operation for --generate_mother_bud_total, for example cell_area_um2=sum"
    )]
    column_operations: Vec<String>,
    #[arg(
        long = "source_path",
        value_name = "PATH_TO_TABLE",
        action = ArgAction::Append,
        help = "Source table path for --combine_metrics"
    )]
    source_paths: Vec<PathBuf>,
    #[arg(
        long = "source_endname",
        value_name = "ENDNAME",
        action = ArgAction::Append,
        help = "Source table endname for --compute_multi_channel"
    )]
    source_endnames: Vec<String>,
    #[arg(
        long = "formula",
        value_name = "COLUMN=EXPRESSION",
        action = ArgAction::Append,
        help = "Formula for metric-combination modes, for example sum_signal=table1_signal+table2_signal"
    )]
    formulas: Vec<String>,
    #[arg(
        long = "equations_path",
        value_name = "PATH_TO_INI",
        help = "Optional equations INI output path for --combine_metrics"
    )]
    equations_path: Option<PathBuf>,
    #[arg(
        long = "append_name",
        value_name = "NAME",
        default_value = "combined_metrics",
        help = "Append name for position-scoped computed metric outputs"
    )]
    append_name: String,
    #[arg(
        long = "table_endname",
        value_name = "ENDNAME",
        default_value = "acdc_output",
        help = "Table endname for --concat_acdc_outputs"
    )]
    table_endname: String,
    #[arg(
        long = "output_format",
        value_name = "csv|xlsx",
        default_value = "csv",
        value_parser = parse_table_format,
        help = "Output table format for --concat_acdc_outputs"
    )]
    output_format: TableFormat,
    #[arg(
        long = "selected_column",
        value_name = "COLUMN",
        action = ArgAction::Append,
        help = "Selected output column for --concat_acdc_outputs; repeat to keep multiple columns"
    )]
    selected_columns: Vec<String>,
    #[arg(
        long = "output_name",
        value_name = "FILENAME",
        help = "Output filename for --concat_acdc_outputs"
    )]
    output_name: Option<String>,
    #[arg(
        long = "multi_experiment_dir",
        value_name = "PATH_TO_OUTPUT_DIR",
        help = "Output directory for multi-experiment --concat_acdc_outputs results"
    )]
    multi_experiment_dir: Option<PathBuf>,
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
        long = "cast_segm_uint32",
        action = ArgAction::SetTrue,
        help = "Cast converted segmentation-like data to uint32 for --convert_file_format"
    )]
    cast_segm_uint32: bool,
    #[arg(
        long = "file_path",
        value_name = "PATH_TO_FILE",
        action = ArgAction::Append,
        help = "File path for --rename_files; repeat to rename multiple files"
    )]
    file_paths: Vec<PathBuf>,
    #[arg(
        long = "rename_append_text",
        value_name = "TEXT",
        help = "Text to append to filenames for --rename_files"
    )]
    rename_append_text: Option<String>,
    #[arg(
        long = "source_dir",
        value_name = "PATH_TO_SOURCE_DIR",
        help = "Source directory for --images_to_positions"
    )]
    source_dir: Option<PathBuf>,
    #[arg(
        long = "target_dir",
        value_name = "PATH_TO_TARGET_DIR",
        help = "Target directory for --images_to_positions"
    )]
    target_dir: Option<PathBuf>,
    #[arg(
        long = "images_append_text",
        value_name = "TEXT",
        help = "Text to append to converted TIFF names for --images_to_positions"
    )]
    images_append_text: Option<String>,
    #[arg(
        long = "channel_name",
        value_name = "CHANNEL",
        action = ArgAction::Append,
        help = "Channel name for channel-scoped utility modes; repeat for multiple channels"
    )]
    channel_names: Vec<String>,
    #[arg(
        long = "reference_channel",
        value_name = "CHANNEL",
        help = "Reference channel for --align_frames"
    )]
    reference_channel: Option<String>,
    #[arg(
        long = "stop_frame",
        value_name = "N",
        help = "Stop frame count for --measure"
    )]
    stop_frame: Option<usize>,
    #[arg(
        long = "save_object_counts",
        action = ArgAction::SetTrue,
        help = "Write acdc_objects_count output when running --measure"
    )]
    save_object_counts: bool,
    #[arg(
        long = "tiff_extension",
        value_name = "EXT",
        default_value = "tif",
        help = "TIFF extension for --move_channel_tiffs_to_positions"
    )]
    tiff_extension: String,
    #[arg(
        long = "segm_append_name",
        value_name = "TEXT",
        help = "Text to append to segmentation output filenames for batch segmentation utilities"
    )]
    segm_append_name: Option<String>,
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
        long = "frame_i",
        value_name = "INDEX",
        help = "Frame index for lineage helper modes"
    )]
    frame_i: Option<i64>,
    #[arg(
        long = "cell_id",
        value_name = "ID",
        action = ArgAction::Append,
        help = "Cell ID for lineage helper modes; repeat to select multiple cells"
    )]
    cell_ids: Vec<i64>,
    #[arg(
        long = "edits_table_path",
        value_name = "PATH_TO_TABLE",
        help = "Lineage edit CSV/XLSX table path for --update_lineage_frame"
    )]
    edits_table_path: Option<PathBuf>,
    #[arg(
        long = "edits_json_path",
        value_name = "PATH_TO_JSON",
        help = "Lineage edit JSON path for --update_lineage_frame"
    )]
    edits_json_path: Option<PathBuf>,
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
        + usize::from(cli.to_obj_coords)
        + usize::from(cli.fill_holes)
        + usize::from(cli.connect_3d_segm)
        + usize::from(cli.stack_2d_segm_to_3d)
        + usize::from(cli.filter_segm_from_table)
        + usize::from(cli.align_frames)
        + usize::from(cli.measure)
        + usize::from(cli.prepare_zstack_segm_info)
        + usize::from(cli.compute_background_roi_data)
        + usize::from(cli.apply_tracking_from_table)
        + usize::from(cli.apply_tracking_from_trackmate_xml)
        + usize::from(cli.add_lineage_tree)
        + usize::from(cli.export_lineage_info)
        + usize::from(cli.propagate_lineage)
        + usize::from(cli.update_lineage_frame)
        + usize::from(cli.generate_mother_bud_total)
        + usize::from(cli.combine_metrics)
        + usize::from(cli.compute_multi_channel)
        + usize::from(cli.concat_acdc_outputs)
        + usize::from(cli.combine_channels)
        + usize::from(cli.convert_file_format)
        + usize::from(cli.rename_files)
        + usize::from(cli.images_to_positions)
        + usize::from(cli.move_channel_tiffs_to_positions);
    if mode_count > 1 {
        bail!(
            "Use only one of --params, --version/--info, --reset, --count_objects, --to_obj_coords, --fill_holes, --connect_3d_segm, --stack_2d_segm_to_3d, --filter_segm_from_table, --align_frames, --measure, --prepare_zstack_segm_info, --compute_background_roi_data, --apply_tracking_from_table, --apply_tracking_from_trackmate_xml, --add_lineage_tree, --export_lineage_info, --propagate_lineage, --update_lineage_frame, --generate_mother_bud_total, --combine_metrics, --compute_multi_channel, --concat_acdc_outputs, --combine_channels, --convert_file_format, --rename_files, --images_to_positions, or --move_channel_tiffs_to_positions"
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

    if cli.to_obj_coords {
        println!("{}", run_to_obj_coords(&cli)?);
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

    if cli.align_frames {
        println!("{}", run_align_frames(&cli)?);
        return Ok(());
    }

    if cli.measure {
        println!("{}", run_measure(&cli)?);
        return Ok(());
    }

    if cli.prepare_zstack_segm_info {
        println!("{}", run_prepare_zstack_segm_info(&cli)?);
        return Ok(());
    }

    if cli.compute_background_roi_data {
        println!("{}", run_compute_background_roi_data(&cli)?);
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

    if cli.export_lineage_info {
        println!("{}", run_export_lineage_info(&cli)?);
        return Ok(());
    }

    if cli.propagate_lineage {
        println!("{}", run_propagate_lineage(&cli)?);
        return Ok(());
    }

    if cli.update_lineage_frame {
        println!("{}", run_update_lineage_frame(&cli)?);
        return Ok(());
    }

    if cli.generate_mother_bud_total {
        println!("{}", run_generate_mother_bud_total(&cli)?);
        return Ok(());
    }

    if cli.combine_metrics {
        println!("{}", run_combine_metrics(&cli)?);
        return Ok(());
    }

    if cli.compute_multi_channel {
        println!("{}", run_compute_multi_channel(&cli)?);
        return Ok(());
    }

    if cli.concat_acdc_outputs {
        println!("{}", run_concat_acdc_outputs(&cli)?);
        return Ok(());
    }

    if cli.combine_channels {
        println!("{}", run_combine_channels(&cli)?);
        return Ok(());
    }

    if cli.convert_file_format {
        println!("{}", run_convert_file_format(&cli)?);
        return Ok(());
    }

    if cli.rename_files {
        println!("{}", run_rename_files(&cli)?);
        return Ok(());
    }

    if cli.images_to_positions {
        println!("{}", run_images_to_positions(&cli)?);
        return Ok(());
    }

    if cli.move_channel_tiffs_to_positions {
        println!("{}", run_move_channel_tiffs_to_positions(&cli)?);
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
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
            let result = count_objects(CountObjectsConfig {
                segmentation_path,
                output_path,
                resolution: utility_mask_resolution(cli),
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
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let segm_endname = cli
                .segm_endname
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--count_objects batch mode requires --segm_endname"))?;
            let result = count_objects_in_positions(CountObjectsBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved object counts for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!("Saved object counts table to {}", path.display()));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--count_objects requires either --segmentation_path and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname"
        ),
    }
}

fn run_to_obj_coords(cli: &Cli) -> Result<String> {
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
            let result = segmentation_to_object_coords(ObjectCoordinatesConfig {
                segmentation_path,
                output_path,
                resolution: utility_mask_resolution(cli),
            })?;
            Ok(format!(
                "Saved object-coordinate table to {}",
                result.primary_path.display()
            ))
        }
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let segm_endname = cli.segm_endname.clone().ok_or_else(|| {
                anyhow::anyhow!("--to_obj_coords batch mode requires --segm_endname")
            })?;
            let result = segmentation_to_object_coords_in_positions(ObjectCoordinatesBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved object-coordinate tables for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!("Saved object-coordinate table to {}", path.display()));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--to_obj_coords requires either --segmentation_path and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname"
        ),
    }
}

fn run_fill_holes(cli: &Cli) -> Result<String> {
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
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
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let segm_endname = cli
                .segm_endname
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--fill_holes batch mode requires --segm_endname"))?;
            let result = fill_holes_in_positions(FillHolesBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                append_name: cli.segm_append_name.clone(),
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved hole-filled segmentation masks for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!(
                    "Saved hole-filled segmentation mask to {}",
                    path.display()
                ));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--fill_holes requires either --segmentation_path and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname"
        ),
    }
}

fn run_connect_3d_segm(cli: &Cli) -> Result<String> {
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
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
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let segm_endname = cli.segm_endname.clone().ok_or_else(|| {
                anyhow::anyhow!("--connect_3d_segm batch mode requires --segm_endname")
            })?;
            let append_name = cli.segm_append_name.clone().ok_or_else(|| {
                anyhow::anyhow!("--connect_3d_segm batch mode requires --segm_append_name")
            })?;
            let result = connect_3d_segm_in_positions(Connect3DSegmBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                append_name,
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved 3D-connected segmentation masks for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!(
                    "Saved 3D-connected segmentation mask to {}",
                    path.display()
                ));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--connect_3d_segm requires either --segmentation_path and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname and --segm_append_name"
        ),
    }
}

fn run_stack_2d_segm_to_3d(cli: &Cli) -> Result<String> {
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
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
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let segm_endname = cli.segm_endname.clone().ok_or_else(|| {
                anyhow::anyhow!("--stack_2d_segm_to_3d batch mode requires --segm_endname")
            })?;
            let append_name = cli.segm_append_name.clone().ok_or_else(|| {
                anyhow::anyhow!("--stack_2d_segm_to_3d batch mode requires --segm_append_name")
            })?;
            let size_z = cli
                .size_z
                .ok_or_else(|| anyhow::anyhow!("--stack_2d_segm_to_3d requires --size_z"))?;
            let result = stack_2d_segm_to_3d_in_positions(Stack2DSegmTo3DBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                append_name,
                size_z,
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved 2D segmentation masks stacked to 3D for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!(
                    "Saved 2D segmentation mask stacked to 3D at {}",
                    path.display()
                ));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--stack_2d_segm_to_3d requires either --segmentation_path and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname, --segm_append_name, and --size_z"
        ),
    }
}

fn run_filter_segm_from_table(cli: &Cli) -> Result<String> {
    match (
        cli.segmentation_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(segmentation_path), Some(output_path), None, None) => {
            let coords_table_path = cli.coords_table_path.clone().ok_or_else(|| {
                anyhow::anyhow!("--filter_segm_from_table requires --coords_table_path")
            })?;
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
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let coords_table_path = cli.coords_table_path.clone().ok_or_else(|| {
                anyhow::anyhow!("--filter_segm_from_table batch mode requires --coords_table_path")
            })?;
            let segm_endname = cli.segm_endname.clone().ok_or_else(|| {
                anyhow::anyhow!("--filter_segm_from_table batch mode requires --segm_endname")
            })?;
            let append_name = cli.segm_append_name.clone().ok_or_else(|| {
                anyhow::anyhow!("--filter_segm_from_table batch mode requires --segm_append_name")
            })?;
            let result = filter_segm_from_table_in_positions(CoordinateFilterBatchConfig {
                position_dir,
                experiment_dir,
                segm_endname,
                coords_table_path,
                append_name,
                x_col: cli.x_col.clone(),
                y_col: cli.y_col.clone(),
                z_col: cli.z_col.clone(),
                frame_col: cli.frame_col.clone(),
                position_col: cli.position_col.clone(),
                position_value: cli.position_value.clone(),
                resolution: utility_mask_resolution(cli),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!(
                "Saved coordinate-filtered segmentation masks for {} position(s)",
                outputs.len()
            )];
            for path in outputs {
                lines.push(format!(
                    "Saved coordinate-filtered segmentation mask to {}",
                    path.display()
                ));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--filter_segm_from_table requires either --segmentation_path, --coords_table_path, and --output_path, or exactly one of --position_dir and --experiment_dir with --segm_endname, --coords_table_path, and --segm_append_name"
        ),
    }
}

fn run_align_frames(cli: &Cli) -> Result<String> {
    let reference_channel = cli
        .reference_channel
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--align_frames requires --reference_channel"))?;
    let position_dirs = match (cli.position_dir.clone(), cli.experiment_dir.clone()) {
        (Some(position_dir), None) => {
            vec![resolve_measurement_position(&position_dir)?.position_dir]
        }
        (None, Some(experiment_dir)) => discover_measurement_experiment(&experiment_dir)?
            .positions
            .into_iter()
            .map(|position| position.position_dir)
            .collect(),
        _ => bail!("--align_frames requires exactly one of --position_dir and --experiment_dir"),
    };
    let mut aligned_count = 0usize;
    let mut lines = Vec::new();
    for position_dir in position_dirs {
        let position = resolve_measurement_position(&position_dir)?;
        let channels_to_align = if cli.channel_names.is_empty() {
            position
                .channels
                .iter()
                .map(|channel| channel.name.clone())
                .collect::<Vec<_>>()
        } else {
            cli.channel_names.clone()
        };
        let config = AlignmentRunConfig {
            position_dir: position.position_dir.clone(),
            reference_channel: reference_channel.clone(),
            channels_to_align,
            frame_range: None,
            overwrite: cli.yes,
        };
        let shifts = compute_alignment_shifts(&config)?;
        let result = apply_alignment(config, &shifts)?;
        aligned_count += result.aligned_outputs.len();
        lines.push(format!(
            "Saved alignment shifts to {}",
            result.shifts_path.display()
        ));
        for path in result.aligned_outputs {
            lines.push(format!("Saved aligned channel to {}", path.display()));
        }
    }
    lines.insert(
        0,
        format!(
            "Aligned {} channel output(s) across selected position(s)",
            aligned_count
        ),
    );
    Ok(lines.join("\n"))
}

fn run_measure(cli: &Cli) -> Result<String> {
    let overwrite_policy = if cli.yes {
        OverwritePolicy::Overwrite
    } else {
        OverwritePolicy::Refuse
    };
    let channel_names = (!cli.channel_names.is_empty()).then(|| cli.channel_names.clone());
    let results = match (cli.position_dir.clone(), cli.experiment_dir.clone()) {
        (Some(position_path), None) => vec![measure_position(MeasurementRunConfig {
            position_path,
            segm_endname: cli.segm_endname.clone(),
            overwrite_policy,
            stop_frame: cli.stop_frame,
            channel_names,
            metric_options: None,
            save_object_counts_table: cli.save_object_counts,
        })?],
        (None, Some(experiment_dir)) => measure_experiment(MeasurementExperimentConfig {
            experiment_dir,
            segm_endname: cli.segm_endname.clone(),
            overwrite_policy,
            stop_frame: cli.stop_frame,
            channel_names,
            metric_options: None,
            save_object_counts_table: cli.save_object_counts,
        })?,
        _ => bail!("--measure requires exactly one of --position_dir and --experiment_dir"),
    };
    let mut lines = vec![format!(
        "Computed measurements for {} position(s)",
        results.len()
    )];
    for result in results {
        lines.push(format!(
            "Saved acdc_output table to {}",
            result.outputs.acdc_output_csv_path.display()
        ));
        if cli.save_object_counts {
            lines.push(format!(
                "Saved object counts table to {}",
                result.outputs.objects_count_csv_path.display()
            ));
        }
    }
    Ok(lines.join("\n"))
}

fn run_prepare_zstack_segm_info(cli: &Cli) -> Result<String> {
    let target = match (cli.position_dir.clone(), cli.experiment_dir.clone()) {
        (Some(position_dir), None) => PrepareSegmInfoTarget::Position(position_dir),
        (None, Some(experiment_dir)) => PrepareSegmInfoTarget::Experiment(experiment_dir),
        _ => bail!(
            "--prepare_zstack_segm_info requires exactly one of --position_dir and --experiment_dir"
        ),
    };
    let paths = prepare_zstack_segm_info(PrepareZStackSegmInfoConfig {
        target,
        overwrite_policy: if cli.yes {
            OverwritePolicy::Overwrite
        } else {
            OverwritePolicy::Refuse
        },
    })?;
    let mut lines = vec![format!(
        "Prepared z-stack segmInfo files for {} position(s)",
        paths.len()
    )];
    for path in paths {
        lines.push(format!(
            "Saved z-stack segmInfo table to {}",
            path.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn run_compute_background_roi_data(cli: &Cli) -> Result<String> {
    let paths = match (cli.position_dir.clone(), cli.experiment_dir.clone()) {
        (Some(position_dir), None) => {
            compute_background_roi_data_for_position(&position_dir, &cli.channel_names)?
        }
        (None, Some(experiment_dir)) => {
            let experiment = discover_measurement_experiment(experiment_dir)?;
            let mut paths = Vec::new();
            for position in experiment.positions {
                paths.extend(compute_background_roi_data_for_position(
                    &position.position_dir,
                    &cli.channel_names,
                )?);
            }
            paths
        }
        _ => bail!(
            "--compute_background_roi_data requires exactly one of --position_dir and --experiment_dir"
        ),
    };
    if paths.is_empty() {
        bail!("No background ROI data archives were written");
    }
    let mut lines = vec![format!(
        "Computed background ROI data for {} channel output(s)",
        paths.len()
    )];
    for path in paths {
        lines.push(format!(
            "Saved background ROI data archive to {}",
            path.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn compute_background_roi_data_for_position(
    position_dir: &Path,
    selected_channels: &[String],
) -> Result<Vec<PathBuf>> {
    let spec = resolve_measurement_position(position_dir)?;
    let background_rois_path = spec
        .data_prep_background_rois_path
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--compute_background_roi_data requires an existing *dataPrep_bkgrROIs.json under {}",
                spec.images_dir.display()
            )
        })?;
    let rois = read_background_roi_json(background_rois_path)?;
    let channel_names = if selected_channels.is_empty() {
        spec.channels
            .iter()
            .map(|channel| channel.name.clone())
            .collect::<Vec<_>>()
    } else {
        for channel_name in selected_channels {
            if !spec
                .channels
                .iter()
                .any(|channel| channel.name == *channel_name)
            {
                bail!(
                    "No supported file found for channel {:?} under {}",
                    channel_name,
                    spec.images_dir.display()
                );
            }
        }
        selected_channels.to_vec()
    };
    compute_background_roi_archives(position_dir, &channel_names, &rois)
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
    match (
        cli.input_path.clone(),
        cli.output_path.clone(),
        cli.position_dir.clone(),
        cli.experiment_dir.clone(),
    ) {
        (Some(input_path), Some(output_path), None, None) => {
            let result = add_lineage_tree(LineageTreeConfig {
                input_path,
                output_path,
            })?;
            Ok(format!(
                "Saved lineage-tree table to {}",
                result.primary_path.display()
            ))
        }
        (None, None, position_dir, experiment_dir)
            if position_dir.is_some() ^ experiment_dir.is_some() =>
        {
            let result = add_lineage_tree_to_tables(LineageTreeBatchConfig {
                position_dir,
                experiment_dir,
                table_endname: cli.table_endname.clone(),
            })?;
            let mut outputs = vec![result.primary_path];
            outputs.extend(result.secondary_paths);
            let mut lines = vec![format!("Added lineage-tree columns to {} table(s)", outputs.len())];
            for path in outputs {
                lines.push(format!("Saved lineage-tree table to {}", path.display()));
            }
            Ok(lines.join("\n"))
        }
        _ => bail!(
            "--add_lineage_tree requires either --input_path and --output_path, or exactly one of --position_dir and --experiment_dir"
        ),
    }
}

fn run_export_lineage_info(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--export_lineage_info requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--export_lineage_info requires --output_path"))?;
    let frame_i = cli
        .frame_i
        .ok_or_else(|| anyhow::anyhow!("--export_lineage_info requires --frame_i"))?;
    let result = export_lineage_info_file(LineageInfoConfig {
        input_path,
        output_path,
        frame_i,
    })?;
    Ok(format!(
        "Saved lineage frame info to {}",
        result.primary_path.display()
    ))
}

fn run_propagate_lineage(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--propagate_lineage requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--propagate_lineage requires --output_path"))?;
    let frame_i = cli
        .frame_i
        .ok_or_else(|| anyhow::anyhow!("--propagate_lineage requires --frame_i"))?;
    let result = propagate_lineage_file(LineagePropagateConfig {
        input_path,
        output_path,
        frame_i,
        cell_ids: if cli.cell_ids.is_empty() {
            None
        } else {
            Some(cli.cell_ids.clone())
        },
    })?;
    Ok(format!(
        "Saved propagated lineage table to {}",
        result.primary_path.display()
    ))
}

fn run_update_lineage_frame(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--update_lineage_frame requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--update_lineage_frame requires --output_path"))?;
    let frame_i = cli
        .frame_i
        .ok_or_else(|| anyhow::anyhow!("--update_lineage_frame requires --frame_i"))?;
    let result = update_lineage_frame_file(LineageUpdateConfig {
        input_path,
        output_path,
        frame_i,
        edits_table_path: cli.edits_table_path.clone(),
        edits_json_path: cli.edits_json_path.clone(),
    })?;
    Ok(format!(
        "Saved updated lineage table to {}",
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

fn run_combine_metrics(cli: &Cli) -> Result<String> {
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--combine_metrics requires --output_path"))?;
    let result = combine_metrics(CombineMetricsConfig {
        source_paths: cli.source_paths.clone(),
        formulas: parse_name_value_pairs(&cli.formulas, "--formula")?,
        output_path,
        equations_path: cli.equations_path.clone(),
    })?;
    Ok(format!(
        "Saved combined metrics table to {}\nSaved combined metrics equations to {}",
        result.output_path.display(),
        result.equations_path.display()
    ))
}

fn run_compute_multi_channel(cli: &Cli) -> Result<String> {
    let result = compute_multi_channel(ComputeMultiChannelConfig {
        position_dir: cli.position_dir.clone(),
        experiment_dir: cli.experiment_dir.clone(),
        source_endnames: cli.source_endnames.clone(),
        formulas: parse_name_value_pairs(&cli.formulas, "--formula")?,
        append_name: cli.append_name.clone(),
    })?;
    let mut lines = vec![format!(
        "Computed multi-channel metrics for {} position(s)",
        result.outputs.len()
    )];
    for output in result.outputs {
        lines.push(format!(
            "Saved combined metrics table to {}",
            output.output_path.display()
        ));
        lines.push(format!(
            "Saved combined metrics equations to {}",
            output.equations_path.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn run_concat_acdc_outputs(cli: &Cli) -> Result<String> {
    let result = concat_acdc_outputs(ConcatConfig {
        experiment_dirs: cli.concat_experiment_dirs.clone(),
        table_endname: cli.table_endname.clone(),
        output_format: cli.output_format,
        selected_columns: if cli.selected_columns.is_empty() {
            None
        } else {
            Some(cli.selected_columns.clone())
        },
        output_name: cli.output_name.clone(),
        multi_experiment_dir: cli.multi_experiment_dir.clone(),
    })?;
    let mut lines = Vec::new();
    for path in result.all_position_outputs {
        lines.push(format!(
            "Saved concatenated position table to {}",
            path.display()
        ));
    }
    for path in result.all_position_count_outputs {
        lines.push(format!(
            "Saved concatenated object-count table to {}",
            path.display()
        ));
    }
    if let Some(path) = result.multi_experiment_output {
        lines.push(format!(
            "Saved concatenated multi-experiment table to {}",
            path.display()
        ));
    }
    if let Some(path) = result.multi_experiment_count_output {
        lines.push(format!(
            "Saved concatenated multi-experiment object-count table to {}",
            path.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn run_combine_channels(cli: &Cli) -> Result<String> {
    let recipe_path = cli
        .recipe_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--combine_channels requires --recipe_path"))?;
    let result = combine_channels(CombineChannelsConfig {
        position_dir: cli.position_dir.clone(),
        experiment_dir: cli.experiment_dir.clone(),
        recipe_path,
        append_name: cli.append_name.clone(),
    })?;
    let mut lines = vec![format!(
        "Combined channels for {} position(s)",
        result.output_paths.len()
    )];
    for path in result.output_paths {
        lines.push(format!(
            "Saved combined channel output to {}",
            path.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn run_convert_file_format(cli: &Cli) -> Result<String> {
    let input_path = cli
        .input_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--convert_file_format requires --input_path"))?;
    let output_path = cli
        .output_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--convert_file_format requires --output_path"))?;
    let result = convert_file_format(ConvertFileFormatConfig {
        input_path,
        output_path,
        cast_segm_uint32: cli.cast_segm_uint32,
    })?;
    Ok(format!(
        "Saved converted file to {}",
        result.primary_path.display()
    ))
}

fn run_rename_files(cli: &Cli) -> Result<String> {
    let append_text = cli
        .rename_append_text
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--rename_files requires --rename_append_text"))?;
    let result = rename_files(RenameFilesConfig {
        file_paths: cli.file_paths.clone(),
        append_text,
    })?;
    let mut outputs = vec![result.primary_path];
    outputs.extend(result.secondary_paths);
    let mut lines = vec![format!("Renamed {} file(s)", outputs.len())];
    for path in outputs {
        lines.push(format!("Saved renamed file to {}", path.display()));
    }
    Ok(lines.join("\n"))
}

fn run_images_to_positions(cli: &Cli) -> Result<String> {
    let source_dir = cli
        .source_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--images_to_positions requires --source_dir"))?;
    let target_dir = cli
        .target_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--images_to_positions requires --target_dir"))?;
    let append_text = cli
        .images_append_text
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--images_to_positions requires --images_append_text"))?;
    let result = images_to_positions(ImagesToPositionsConfig {
        source_dir,
        target_dir,
        append_text,
    })?;
    let mut outputs = vec![result.primary_path];
    outputs.extend(result.secondary_paths);
    let mut lines = vec![format!(
        "Converted {} image file(s) to Position folders",
        outputs.len()
    )];
    for path in outputs {
        lines.push(format!("Saved converted image to {}", path.display()));
    }
    Ok(lines.join("\n"))
}

fn run_move_channel_tiffs_to_positions(cli: &Cli) -> Result<String> {
    let source_dir = cli.source_dir.clone().ok_or_else(|| {
        anyhow::anyhow!("--move_channel_tiffs_to_positions requires --source_dir")
    })?;
    let result = move_channel_tiffs_to_positions(MoveChannelTiffsConfig {
        source_dir,
        channel_names: cli.channel_names.clone(),
        extension: cli.tiff_extension.clone(),
    })?;
    let mut outputs = vec![result.primary_path];
    outputs.extend(result.secondary_paths);
    let mut lines = vec![format!(
        "Moved channel TIFFs into {} position folder(s)",
        outputs.len()
    )];
    for path in outputs {
        lines.push(format!("Created position Images folder {}", path.display()));
    }
    Ok(lines.join("\n"))
}

fn parse_column_operations(values: &[String]) -> Result<BTreeMap<String, String>> {
    parse_name_value_pairs(values, "--column_operation")
}

fn parse_name_value_pairs(values: &[String], flag_name: &str) -> Result<BTreeMap<String, String>> {
    let mut mapper = BTreeMap::new();
    for value in values {
        let Some((column, operation)) = value.split_once('=') else {
            bail!("{flag_name} must use NAME=VALUE syntax");
        };
        let column = column.trim();
        let operation = operation.trim();
        if column.is_empty() || operation.is_empty() {
            bail!("{flag_name} must use non-empty NAME=VALUE values");
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
        || cli.experiment_dir.is_some()
        || !cli.concat_experiment_dirs.is_empty()
        || cli.segm_endname.is_some()
        || cli.xml_path.is_some()
        || cli.recipe_path.is_some()
        || !cli.column_operations.is_empty()
        || !cli.source_paths.is_empty()
        || !cli.source_endnames.is_empty()
        || !cli.formulas.is_empty()
        || cli.equations_path.is_some()
        || cli.append_name != "combined_metrics"
        || cli.table_endname != "acdc_output"
        || cli.output_format != TableFormat::Csv
        || !cli.selected_columns.is_empty()
        || cli.output_name.is_some()
        || cli.multi_experiment_dir.is_some()
        || !cli.grouping_columns.is_empty()
        || cli.entity_colname != "entity"
        || cli.no_copy_all_nonselected_columns
        || cli.cast_segm_uint32
        || !cli.file_paths.is_empty()
        || cli.rename_append_text.is_some()
        || cli.source_dir.is_some()
        || cli.target_dir.is_some()
        || cli.images_append_text.is_some()
        || !cli.channel_names.is_empty()
        || cli.reference_channel.is_some()
        || cli.stop_frame.is_some()
        || cli.save_object_counts
        || cli.tiff_extension != "tif"
        || cli.segm_append_name.is_some()
        || cli.size_t.is_some()
        || cli.size_z.is_some()
        || cli.segm_layout.is_some()
        || cli.coords_table_path.is_some()
        || cli.x_col != "x"
        || cli.y_col != "y"
        || cli.z_col.is_some()
        || cli.frame_col.is_some()
        || cli.frame_i.is_some()
        || !cli.cell_ids.is_empty()
        || cli.edits_table_path.is_some()
        || cli.edits_json_path.is_some()
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
        bail!("Utility path/layout flags require a utility mode such as --count_objects, --to_obj_coords, --fill_holes, --connect_3d_segm, --stack_2d_segm_to_3d, --filter_segm_from_table, --align_frames, --measure, --prepare_zstack_segm_info, --compute_background_roi_data, --apply_tracking_from_table, --apply_tracking_from_trackmate_xml, --add_lineage_tree, --export_lineage_info, --propagate_lineage, --update_lineage_frame, --generate_mother_bud_total, --combine_metrics, --compute_multi_channel, --concat_acdc_outputs, --combine_channels, --convert_file_format, --rename_files, --images_to_positions, or --move_channel_tiffs_to_positions");
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

fn parse_table_format(value: &str) -> Result<TableFormat, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "csv" => Ok(TableFormat::Csv),
        "xlsx" => Ok(TableFormat::Xlsx),
        _ => Err("expected csv or xlsx".to_string()),
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
