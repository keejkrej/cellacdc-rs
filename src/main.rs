use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use cellacdc_rs::{
    resolve_position, run_experiment, run_position, ExperimentRunConfig, OverwritePolicy,
    SegmentationParams, SegmentationRunConfig, TrackingConfig,
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
    #[arg(long, default_value_t = 25.0)]
    track_max_distance_px: f32,
    #[arg(long, default_value_t = 1)]
    track_min_overlap_px: usize,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::RunPosition(args) => run_position_command(args),
        Command::RunExperiment(args) => run_experiment_command(args),
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
            max_distance_px: self.track_max_distance_px,
            min_overlap_px: self.track_min_overlap_px,
        })
    }
}
