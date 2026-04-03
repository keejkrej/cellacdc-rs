use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::collections::BTreeMap;
use std::path::PathBuf;

use cellacdc_rs::{
    add_lineage_tree, apply_tracking_from_table, build_lineage_state_file, combine_metrics,
    concat_acdc_outputs, connect_3d_segm, count_objects, export_lineage_info_file, fill_holes,
    filter_segm_from_table, generate_mother_bud_total, measure_experiment, measure_position,
    prepare_zstack_segm_info, propagate_lineage_file, resolve_position, run_experiment,
    run_position, stack_2d_segm_to_3d, update_lineage_frame_file, ApplyTrackingConfig,
    CombineMetricsConfig, ConcatConfig, Connect3DSegmConfig, CoordinateFilterConfig,
    CountObjectsConfig, ExperimentRunConfig, FillHolesConfig, GenerateMotherBudTotalConfig,
    LineageBuildConfig, LineageInfoConfig, LineagePropagateConfig, LineageTreeConfig,
    LineageUpdateConfig, MaskPathResolution, MeasurementExperimentConfig, MeasurementRunConfig,
    OverwritePolicy, PrepareSegmInfoTarget, PrepareZStackSegmInfoConfig, SegmentationLayout,
    SegmentationParams, SegmentationRunConfig, Stack2DSegmTo3DConfig, TableFormat,
    TrackingColumnMap, TrackingConfig,
};

#[derive(Debug, Parser)]
#[command(name = "cellacdc-rs")]
#[command(about = "Cell-ACDC-compatible Rust segmentation runner", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    RunPosition(RunPositionArgs),
    RunExperiment(RunExperimentArgs),
    MeasurePosition(MeasurePositionArgs),
    MeasureExperiment(MeasureExperimentArgs),
    ConcatAcdcOutput(ConcatAcdcOutputArgs),
    CombineMetrics(CombineMetricsArgs),
    CountObjects(CountObjectsArgs),
    FillHoles(FillHolesArgs),
    #[command(name = "prepare-zstack-segm-info")]
    PrepareZstackSegmInfo(PrepareZstackSegmInfoArgs),
    #[command(name = "connect-3d-segm")]
    Connect3DSegm(Connect3DSegmArgs),
    #[command(name = "stack-2d-segm-to-3d")]
    Stack2DSegmTo3D(Stack2DSegmTo3DArgs),
    FilterSegmFromTable(FilterSegmFromTableArgs),
    ApplyTrackingFromTable(ApplyTrackingFromTableArgs),
    AddLineageTree(AddLineageTreeArgs),
    #[command(name = "build-lineage-state")]
    BuildLineageState(BuildLineageStateArgs),
    #[command(name = "update-lineage-frame")]
    UpdateLineageFrame(UpdateLineageFrameArgs),
    #[command(name = "propagate-lineage")]
    PropagateLineage(PropagateLineageArgs),
    #[command(name = "export-lineage-info")]
    ExportLineageInfo(ExportLineageInfoArgs),
    GenerateMotherBudTotal(GenerateMotherBudTotalArgs),
}

#[derive(Debug, Args)]
struct CommonArgs {
    #[arg(long)]
    phase_channel: String,
    #[arg(long)]
    fluo_channel: String,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    segm_endname: Option<String>,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 256)]
    tile: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.0)]
    cellprob_threshold: f32,
    #[arg(long, default_value_t = 200)]
    niter: usize,
    #[arg(long, default_value_t = 15)]
    min_size: usize,
    #[arg(long)]
    track: bool,
    #[arg(long, default_value_t = 0.4)]
    track_ioa_threshold: f32,
}

#[derive(Debug, Args)]
struct RunPositionArgs {
    #[arg(long)]
    position: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Debug, Args)]
struct RunExperimentArgs {
    #[arg(long)]
    experiment: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Debug, Args)]
struct MeasurePositionArgs {
    #[arg(long)]
    position: PathBuf,
    #[arg(long)]
    segm_endname: Option<String>,
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Args)]
struct MeasureExperimentArgs {
    #[arg(long)]
    experiment: PathBuf,
    #[arg(long)]
    segm_endname: Option<String>,
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TableFormatArg {
    Csv,
    Xlsx,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LayoutArg {
    Yx,
    Tyx,
    Zyx,
    Tzyx,
}

#[derive(Debug, Args)]
struct MaskResolutionArgs {
    #[arg(long)]
    size_t: Option<usize>,
    #[arg(long)]
    size_z: Option<usize>,
    #[arg(long, value_enum)]
    layout: Option<LayoutArg>,
}

#[derive(Debug, Args)]
struct ConcatAcdcOutputArgs {
    #[arg(long = "experiment", required = true)]
    experiments: Vec<PathBuf>,
    #[arg(long, default_value = "acdc_output")]
    table_endname: String,
    #[arg(long, value_enum, default_value_t = TableFormatArg::Csv)]
    format: TableFormatArg,
    #[arg(long = "column")]
    columns: Vec<String>,
    #[arg(long)]
    output_name: Option<String>,
    #[arg(long)]
    multi_experiment_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct CombineMetricsArgs {
    #[arg(long = "source", required = true)]
    sources: Vec<PathBuf>,
    #[arg(long = "formula", required = true)]
    formulas: Vec<String>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    equations: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct CountObjectsArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct FillHolesArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct PrepareZstackSegmInfoArgs {
    #[arg(long, conflicts_with = "experiment")]
    position: Option<PathBuf>,
    #[arg(long, conflicts_with = "position")]
    experiment: Option<PathBuf>,
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Args)]
struct Connect3DSegmArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct Stack2DSegmTo3DArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    size_z: usize,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct FilterSegmFromTableArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    table: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    x_col: String,
    #[arg(long)]
    y_col: String,
    #[arg(long)]
    z_col: Option<String>,
    #[arg(long)]
    frame_col: Option<String>,
    #[arg(long)]
    position_col: Option<String>,
    #[arg(long)]
    position_value: Option<String>,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct ApplyTrackingFromTableArgs {
    #[arg(long)]
    segmentation: PathBuf,
    #[arg(long)]
    table: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    frame_index_col: String,
    #[arg(long)]
    track_ids_col: String,
    #[arg(long)]
    mask_ids_col: Option<String>,
    #[arg(long)]
    x_centroid_col: Option<String>,
    #[arg(long)]
    y_centroid_col: Option<String>,
    #[arg(long)]
    z_centroid_col: Option<String>,
    #[arg(long)]
    first_frame_one: bool,
    #[arg(long)]
    delete_untracked_ids: bool,
    #[arg(long)]
    source_acdc_output: Option<PathBuf>,
    #[arg(long)]
    output_acdc_output: Option<PathBuf>,
    #[command(flatten)]
    resolution: MaskResolutionArgs,
}

#[derive(Debug, Args)]
struct AddLineageTreeArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct BuildLineageStateArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct UpdateLineageFrameArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    frame_i: i64,
    #[arg(long, conflicts_with = "edits_json")]
    edits_table: Option<PathBuf>,
    #[arg(long, conflicts_with = "edits_table")]
    edits_json: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct PropagateLineageArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    frame_i: i64,
    #[arg(long = "cell-id")]
    cell_ids: Vec<i64>,
}

#[derive(Debug, Args)]
struct ExportLineageInfoArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    frame_i: i64,
}

#[derive(Debug, Args)]
struct GenerateMotherBudTotalArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long = "column-operation", required = true)]
    column_operations: Vec<String>,
    #[arg(long = "grouping-column")]
    grouping_columns: Vec<String>,
    #[arg(long, default_value = "entity")]
    entity_colname: String,
    #[arg(long)]
    selected_columns_only: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::RunPosition(args) => run_position_command(args),
        Command::RunExperiment(args) => run_experiment_command(args),
        Command::MeasurePosition(args) => measure_position_command(args),
        Command::MeasureExperiment(args) => measure_experiment_command(args),
        Command::ConcatAcdcOutput(args) => concat_acdc_output_command(args),
        Command::CombineMetrics(args) => combine_metrics_command(args),
        Command::CountObjects(args) => count_objects_command(args),
        Command::FillHoles(args) => fill_holes_command(args),
        Command::PrepareZstackSegmInfo(args) => prepare_zstack_segm_info_command(args),
        Command::Connect3DSegm(args) => connect_3d_segm_command(args),
        Command::Stack2DSegmTo3D(args) => stack_2d_segm_to_3d_command(args),
        Command::FilterSegmFromTable(args) => filter_segm_from_table_command(args),
        Command::ApplyTrackingFromTable(args) => apply_tracking_from_table_command(args),
        Command::AddLineageTree(args) => add_lineage_tree_command(args),
        Command::BuildLineageState(args) => build_lineage_state_command(args),
        Command::UpdateLineageFrame(args) => update_lineage_frame_command(args),
        Command::PropagateLineage(args) => propagate_lineage_command(args),
        Command::ExportLineageInfo(args) => export_lineage_info_command(args),
        Command::GenerateMotherBudTotal(args) => generate_mother_bud_total_command(args),
    }
}

fn run_position_command(args: RunPositionArgs) -> Result<()> {
    let common = args.common;
    let position = resolve_position(
        &args.position,
        common.phase_channel.clone(),
        common.fluo_channel.clone(),
    )?;
    let config = SegmentationRunConfig {
        position,
        model_path: common.model.clone(),
        segm_endname: common.segm_endname.clone(),
        overwrite_policy: overwrite_policy(common.overwrite),
        cpu: common.cpu,
        params: common.params(),
        tracking: common.tracking(),
    };
    let result = run_position(config)?;
    println!(
        "Segmented {} frame(s) in {} and wrote {} / {}",
        result.frames_processed,
        result.position_dir.display(),
        result.outputs.segm_npz_path.display(),
        result.outputs.acdc_output_csv_path.display()
    );
    Ok(())
}

fn run_experiment_command(args: RunExperimentArgs) -> Result<()> {
    let common = args.common;
    let params = common.params();
    let tracking = common.tracking();
    let results = run_experiment(ExperimentRunConfig {
        experiment_dir: args.experiment,
        phase_channel: common.phase_channel,
        fluo_channel: common.fluo_channel,
        model_path: common.model,
        segm_endname: common.segm_endname,
        overwrite_policy: overwrite_policy(common.overwrite),
        cpu: common.cpu,
        params,
        tracking,
    })?;
    println!("Segmented {} positions", results.len());
    for result in results {
        println!(
            "{} ({} frame(s)) -> {}",
            result.position_dir.display(),
            result.frames_processed,
            result.outputs.segm_npz_path.display()
        );
    }
    Ok(())
}

fn measure_position_command(args: MeasurePositionArgs) -> Result<()> {
    let result = measure_position(MeasurementRunConfig {
        position_path: args.position,
        segm_endname: args.segm_endname,
        overwrite_policy: overwrite_policy(args.overwrite),
    })?;
    println!(
        "Measured {} frame(s) in {} and wrote {}",
        result.frames_processed,
        result.position_dir.display(),
        result.outputs.acdc_output_csv_path.display()
    );
    Ok(())
}

fn measure_experiment_command(args: MeasureExperimentArgs) -> Result<()> {
    let results = measure_experiment(MeasurementExperimentConfig {
        experiment_dir: args.experiment,
        segm_endname: args.segm_endname,
        overwrite_policy: overwrite_policy(args.overwrite),
    })?;
    println!("Measured {} positions", results.len());
    for result in results {
        println!(
            "{} ({} frame(s)) -> {}",
            result.position_dir.display(),
            result.frames_processed,
            result.outputs.acdc_output_csv_path.display()
        );
    }
    Ok(())
}

fn concat_acdc_output_command(args: ConcatAcdcOutputArgs) -> Result<()> {
    let result = concat_acdc_outputs(ConcatConfig {
        experiment_dirs: args.experiments,
        table_endname: args.table_endname,
        output_format: args.format.into(),
        selected_columns: (!args.columns.is_empty()).then_some(args.columns),
        output_name: args.output_name,
        multi_experiment_dir: args.multi_experiment_dir,
    })?;
    println!(
        "Wrote {} all-position table(s)",
        result.all_position_outputs.len()
    );
    for path in result.all_position_outputs {
        println!("{}", path.display());
    }
    if let Some(path) = result.multi_experiment_output {
        println!("Multi-experiment output: {}", path.display());
    }
    Ok(())
}

fn combine_metrics_command(args: CombineMetricsArgs) -> Result<()> {
    let result = combine_metrics(CombineMetricsConfig {
        source_paths: args.sources,
        formulas: parse_assignments(args.formulas, "formula")?,
        output_path: args.output,
        equations_path: args.equations,
    })?;
    println!(
        "Wrote combined metrics to {} and equations to {}",
        result.output_path.display(),
        result.equations_path.display()
    );
    Ok(())
}

fn count_objects_command(args: CountObjectsArgs) -> Result<()> {
    let result = count_objects(CountObjectsConfig {
        segmentation_path: args.segmentation,
        output_path: args.output,
        resolution: Some(args.resolution.into()),
    })?;
    println!(
        "Wrote object counts to {}",
        result.summary.output_path.display()
    );
    Ok(())
}

fn fill_holes_command(args: FillHolesArgs) -> Result<()> {
    let result = fill_holes(FillHolesConfig {
        segmentation_path: args.segmentation,
        output_path: args.output,
        resolution: Some(args.resolution.into()),
    })?;
    println!(
        "Wrote filled segmentation to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn prepare_zstack_segm_info_command(args: PrepareZstackSegmInfoArgs) -> Result<()> {
    let target = match (args.position, args.experiment) {
        (Some(position), None) => PrepareSegmInfoTarget::Position(position),
        (None, Some(experiment)) => PrepareSegmInfoTarget::Experiment(experiment),
        _ => anyhow::bail!("Provide exactly one of --position or --experiment."),
    };
    let outputs = prepare_zstack_segm_info(PrepareZStackSegmInfoConfig {
        target,
        overwrite_policy: overwrite_policy(args.overwrite),
    })?;
    for path in outputs {
        println!("{}", path.display());
    }
    Ok(())
}

fn connect_3d_segm_command(args: Connect3DSegmArgs) -> Result<()> {
    let result = connect_3d_segm(Connect3DSegmConfig {
        segmentation_path: args.segmentation,
        output_path: args.output,
        resolution: Some(args.resolution.into()),
    })?;
    println!(
        "Wrote connected 3D mask to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn stack_2d_segm_to_3d_command(args: Stack2DSegmTo3DArgs) -> Result<()> {
    let result = stack_2d_segm_to_3d(Stack2DSegmTo3DConfig {
        segmentation_path: args.segmentation,
        output_path: args.output,
        size_z: args.size_z,
        resolution: Some(args.resolution.into()),
    })?;
    println!("Wrote stacked 3D mask to {}", result.primary_path.display());
    Ok(())
}

fn filter_segm_from_table_command(args: FilterSegmFromTableArgs) -> Result<()> {
    let result = filter_segm_from_table(CoordinateFilterConfig {
        segmentation_path: args.segmentation,
        coords_table_path: args.table,
        output_path: args.output,
        x_col: args.x_col,
        y_col: args.y_col,
        z_col: args.z_col,
        frame_col: args.frame_col,
        position_col: args.position_col,
        position_value: args.position_value,
        resolution: Some(args.resolution.into()),
    })?;
    println!(
        "Wrote filtered segmentation to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn apply_tracking_from_table_command(args: ApplyTrackingFromTableArgs) -> Result<()> {
    let result = apply_tracking_from_table(ApplyTrackingConfig {
        segmentation_path: args.segmentation,
        tracking_table_path: args.table,
        output_path: args.output,
        columns: TrackingColumnMap {
            frame_index_col: args.frame_index_col,
            is_first_frame_one: args.first_frame_one,
            track_ids_col: args.track_ids_col,
            mask_ids_col: args.mask_ids_col,
            x_centroid_col: args.x_centroid_col,
            y_centroid_col: args.y_centroid_col,
            z_centroid_col: args.z_centroid_col,
            delete_untracked_ids: args.delete_untracked_ids,
        },
        resolution: Some(args.resolution.into()),
        source_acdc_output_path: args.source_acdc_output,
        output_acdc_output_path: args.output_acdc_output,
    })?;
    println!(
        "Wrote tracked segmentation to {}",
        result.primary_path.display()
    );
    for path in result.secondary_paths {
        println!("{}", path.display());
    }
    Ok(())
}

fn add_lineage_tree_command(args: AddLineageTreeArgs) -> Result<()> {
    let result = add_lineage_tree(LineageTreeConfig {
        input_path: args.input,
        output_path: args.output,
    })?;
    println!("Wrote lineage table to {}", result.primary_path.display());
    Ok(())
}

fn build_lineage_state_command(args: BuildLineageStateArgs) -> Result<()> {
    let result = build_lineage_state_file(LineageBuildConfig {
        input_path: args.input,
        output_path: args.output,
    })?;
    println!(
        "Wrote normalized lineage state to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn update_lineage_frame_command(args: UpdateLineageFrameArgs) -> Result<()> {
    let result = update_lineage_frame_file(LineageUpdateConfig {
        input_path: args.input,
        output_path: args.output,
        frame_i: args.frame_i,
        edits_table_path: args.edits_table,
        edits_json_path: args.edits_json,
    })?;
    println!(
        "Wrote updated lineage frame table to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn propagate_lineage_command(args: PropagateLineageArgs) -> Result<()> {
    let result = propagate_lineage_file(LineagePropagateConfig {
        input_path: args.input,
        output_path: args.output,
        frame_i: args.frame_i,
        cell_ids: (!args.cell_ids.is_empty()).then_some(args.cell_ids),
    })?;
    println!(
        "Wrote propagated lineage table to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn export_lineage_info_command(args: ExportLineageInfoArgs) -> Result<()> {
    let result = export_lineage_info_file(LineageInfoConfig {
        input_path: args.input,
        output_path: args.output,
        frame_i: args.frame_i,
    })?;
    println!(
        "Wrote lineage frame info to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn generate_mother_bud_total_command(args: GenerateMotherBudTotalArgs) -> Result<()> {
    let result = generate_mother_bud_total(GenerateMotherBudTotalConfig {
        input_path: args.input,
        output_path: args.output,
        column_operation_mapper: parse_assignments(args.column_operations, "column-operation")?,
        copy_all_nonselected_columns: !args.selected_columns_only,
        grouping_columns: args.grouping_columns,
        entity_colname: args.entity_colname,
    })?;
    println!(
        "Wrote mother-bud-total table to {}",
        result.primary_path.display()
    );
    Ok(())
}

fn overwrite_policy(overwrite: bool) -> OverwritePolicy {
    if overwrite {
        OverwritePolicy::Overwrite
    } else {
        OverwritePolicy::Refuse
    }
}

impl CommonArgs {
    fn params(&self) -> SegmentationParams {
        SegmentationParams {
            tile: self.tile,
            batch_size: self.batch_size,
            cellprob_threshold: self.cellprob_threshold,
            niter: self.niter,
            min_size: self.min_size,
        }
    }

    fn tracking(&self) -> Option<TrackingConfig> {
        self.track.then(|| TrackingConfig {
            ioa_threshold: self.track_ioa_threshold,
        })
    }
}

impl From<TableFormatArg> for TableFormat {
    fn from(value: TableFormatArg) -> Self {
        match value {
            TableFormatArg::Csv => TableFormat::Csv,
            TableFormatArg::Xlsx => TableFormat::Xlsx,
        }
    }
}

impl From<LayoutArg> for SegmentationLayout {
    fn from(value: LayoutArg) -> Self {
        match value {
            LayoutArg::Yx => SegmentationLayout::YX,
            LayoutArg::Tyx => SegmentationLayout::TYX,
            LayoutArg::Zyx => SegmentationLayout::ZYX,
            LayoutArg::Tzyx => SegmentationLayout::TZYX,
        }
    }
}

impl From<MaskResolutionArgs> for MaskPathResolution {
    fn from(value: MaskResolutionArgs) -> Self {
        Self {
            size_t: value.size_t,
            size_z: value.size_z,
            layout: value.layout.map(Into::into),
        }
    }
}

fn parse_assignments(items: Vec<String>, flag_name: &str) -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    for item in items {
        let Some((name, expression)) = item.split_once('=') else {
            anyhow::bail!("Invalid --{flag_name} value {item:?}. Expected NAME=EXPRESSION.");
        };
        parsed.insert(name.trim().to_string(), expression.trim().to_string());
    }
    Ok(parsed)
}
