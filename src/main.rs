mod gui;

use anyhow::{bail, Result};
use clap::{ArgAction, Parser};
use std::ffi::OsString;
use std::path::PathBuf;

use cellacdc_rs::{run_workflow_file, WorkflowRunOptions};

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
        short = 'd',
        long = "debug",
        action = ArgAction::SetTrue,
        help = "Enable verbose workflow logging for params-file runs"
    )]
    debug: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse_from(preprocess_args());
    let mode_count = usize::from(cli.params.is_some()) + usize::from(cli.version || cli.info);
    if mode_count > 1 {
        bail!("Use either --params or --version/--info, not both");
    }
    if cli.debug && cli.params.is_none() {
        bail!("--debug is only supported together with --params");
    }

    if cli.version || cli.info {
        println!("{}", build_info_text());
        return Ok(());
    }

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
    format!(
        "cellacdc-rs {}\nOS: {}\nARCH: {}\nProfile: {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        option_env!("PROFILE").unwrap_or("release"),
    )
}
