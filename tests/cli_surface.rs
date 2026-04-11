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
