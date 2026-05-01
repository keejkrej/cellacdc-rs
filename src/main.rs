mod gui;

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser};
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

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
        + usize::from(cli.reset);
    if mode_count > 1 {
        bail!("Use only one of --params, --version/--info, or --reset");
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
