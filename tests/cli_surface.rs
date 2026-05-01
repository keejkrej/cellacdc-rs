use std::fs;
use std::process::Command;

fn run_bin(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .args(args)
        .output()
        .expect("run binary")
}

#[test]
fn help_shows_flat_cli_without_subcommands() {
    let output = run_bin(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cellacdc-rs [OPTIONS]"));
    assert!(stdout.contains("--params <PATH_TO_PARAMS>"));
    assert!(stdout.contains("--info"));
    assert!(!stdout.contains("Commands:"));
    assert!(!stdout.contains("run-position"));
}

#[test]
fn version_flag_succeeds() {
    let output = run_bin(&["-v"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cellacdc-rs"));
    assert!(stdout.contains("OS:"));
    assert!(stdout.contains("User profile folder:"));
    assert!(stdout.contains("Settings folder:"));
    assert!(stdout.contains("Working directory:"));
}

#[test]
fn yes_flag_is_accepted_for_python_cli_compatibility() {
    let output = run_bin(&["-y", "-v"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cellacdc-rs"));
}

#[test]
fn info_uses_python_profile_pointer_file() {
    let temp = tempfile::tempdir().expect("temp dir");
    let xdg = temp.path().join("xdg");
    let profile = temp.path().join("custom-profile");
    fs::create_dir_all(xdg.join("Cell_ACDC")).expect("user data dir");
    fs::write(
        xdg.join("Cell_ACDC").join("acdc_user_profile_location.txt"),
        profile.display().to_string(),
    )
    .expect("profile pointer");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--info")
        .env("HOME", temp.path())
        .env("XDG_DATA_HOME", &xdg)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("User profile folder: {}", profile.display())));
    assert!(stdout.contains(&format!(
        "Settings folder: {}",
        profile.join(".acdc-settings").display()
    )));
}

#[test]
fn install_details_flag_is_accepted_for_python_cli_compatibility() {
    let temp = tempfile::tempdir().expect("temp dir");
    let details_path = temp.path().join("install_details.json");
    fs::write(
        &details_path,
        r#"{"venv_path":"venv","conda_path":"conda","target_dir":"target"}"#,
    )
    .expect("install details");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--install_details")
        .arg(&details_path)
        .arg("-v")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cellacdc-rs"));
}

#[test]
fn invalid_install_details_file_fails_cleanly() {
    let temp = tempfile::tempdir().expect("temp dir");
    let details_path = temp.path().join("install_details.json");
    fs::write(&details_path, "not-json").expect("install details");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--install_details")
        .arg(&details_path)
        .arg("-v")
        .output()
        .expect("run binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to parse install details file"));
}

#[test]
fn model_download_flags_fail_with_explicit_rust_message() {
    let output = run_bin(&["--AllModelsDownload"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Python model download flags are not supported"));
}

#[test]
fn reset_yes_removes_default_python_settings_dir() {
    let temp = tempfile::tempdir().expect("temp dir");
    let settings_dir = temp.path().join("acdc-appdata").join(".acdc-settings");
    fs::create_dir_all(&settings_dir).expect("settings dir");
    fs::write(settings_dir.join("settings.csv"), "setting,value\n").expect("settings file");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .args(["--reset", "--yes"])
        .env("HOME", temp.path())
        .env_remove("XDG_DATA_HOME")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    assert!(!settings_dir.exists());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Cell-ACDC settings have been reset"));
}

#[test]
fn reset_without_yes_cancels_on_empty_stdin() {
    let temp = tempfile::tempdir().expect("temp dir");
    let settings_dir = temp.path().join("acdc-appdata").join(".acdc-settings");
    fs::create_dir_all(&settings_dir).expect("settings dir");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--reset")
        .env("HOME", temp.path())
        .env_remove("XDG_DATA_HOME")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run binary");

    assert!(output.status.success());
    assert!(settings_dir.exists());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Resetting Cell-ACDC settings cancelled"));
}

#[test]
fn reset_uses_python_profile_pointer_file() {
    let temp = tempfile::tempdir().expect("temp dir");
    let xdg = temp.path().join("xdg");
    let profile = temp.path().join("custom-profile");
    let settings_dir = profile.join(".acdc-settings");
    fs::create_dir_all(xdg.join("Cell_ACDC")).expect("user data dir");
    fs::create_dir_all(&settings_dir).expect("settings dir");
    fs::write(
        xdg.join("Cell_ACDC").join("acdc_user_profile_location.txt"),
        profile.display().to_string(),
    )
    .expect("profile pointer");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .args(["--reset", "-y"])
        .env("HOME", temp.path())
        .env("XDG_DATA_HOME", &xdg)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    assert!(!settings_dir.exists());
}

#[test]
fn old_subcommands_are_rejected() {
    let output = run_bin(&["run-position"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unexpected argument 'run-position'"));
}

#[test]
fn missing_params_file_fails_cleanly() {
    let output = run_bin(&["-p", "does-not-exist.ini"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to read workflow file"));
}
