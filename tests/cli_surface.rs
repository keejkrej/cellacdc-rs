use ndarray::{Array2, Array3};
use ndarray_npy::{NpzReader, NpzWriter};
use std::fs::{self, File};
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
    assert!(stdout.contains("--count_objects"));
    assert!(stdout.contains("--fill_holes"));
    assert!(stdout.contains("--connect_3d_segm"));
    assert!(stdout.contains("--stack_2d_segm_to_3d"));
    assert!(stdout.contains("--filter_segm_from_table"));
    assert!(stdout.contains("--apply_tracking_from_table"));
    assert!(stdout.contains("--add_lineage_tree"));
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
fn count_objects_utility_writes_counts_csv() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let output_path = temp.path().join("objects_count.csv");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (3, 3),
        vec![
            0u32, 1, 1, //
            0, 2, 2, //
            0, 0, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--count_objects")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved object counts table"));
    assert!(stdout.contains("In current position: 2"));
    let csv = fs::read_to_string(output_path).expect("counts csv");
    assert!(csv.contains("In current position"));
    assert!(csv.contains("\n2\n"));
}

#[test]
fn fill_holes_utility_writes_corrected_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let output_path = temp.path().join("segm_filled.npz");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (3, 3),
        vec![
            1u32, 1, 1, //
            1, 0, 1, //
            1, 1, 1,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--fill_holes")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved hole-filled segmentation mask"));
    let mut npz = NpzReader::new(File::open(output_path).expect("filled npz")).expect("read npz");
    let filled: Array2<u32> = npz.by_name("arr_0.npy").expect("filled array");
    assert_eq!(filled[[1, 1]], 1);
}

#[test]
fn connect_3d_segm_utility_writes_connected_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let output_path = temp.path().join("segm_connected.npz");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            0u32, 1, //
            0, 0, //
            0, 0, //
            0, 1,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--connect_3d_segm")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--segm_layout")
        .arg("ZYX")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved 3D-connected segmentation mask"));
    let mut npz =
        NpzReader::new(File::open(output_path).expect("connected npz")).expect("read npz");
    let connected: Array3<u32> = npz.by_name("arr_0.npy").expect("connected array");
    assert_eq!(
        connected.iter().copied().collect::<Vec<_>>(),
        vec![
            0, 0, //
            0, 1, //
            0, 0, //
            0, 1,
        ]
    );
}

#[test]
fn stack_2d_segm_to_3d_utility_writes_stacked_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let output_path = temp.path().join("segm_3d.npz");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (2, 2),
        vec![
            0u32, 1, //
            2, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--stack_2d_segm_to_3d")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--size_z")
        .arg("3")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved 2D segmentation mask stacked to 3D"));
    let mut npz = NpzReader::new(File::open(output_path).expect("stacked npz")).expect("read npz");
    let stacked: Array3<u32> = npz.by_name("arr_0.npy").expect("stacked array");
    assert_eq!(stacked.shape(), &[3, 2, 2]);
    for z in 0..3 {
        assert_eq!(stacked[[z, 0, 1]], 1);
        assert_eq!(stacked[[z, 1, 0]], 2);
    }
}

#[test]
fn filter_segm_from_table_utility_writes_filtered_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let coords_path = temp.path().join("coords.csv");
    let output_path = temp.path().join("filtered.npz");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (2, 3),
        vec![
            1u32, 1, 0, //
            2, 2, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");
    fs::write(&coords_path, "x,y\n0,0\n").expect("coords csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--filter_segm_from_table")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--coords_table_path")
        .arg(&coords_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved coordinate-filtered segmentation mask"));
    let mut npz = NpzReader::new(File::open(output_path).expect("filtered npz")).expect("read npz");
    let filtered: Array2<u32> = npz.by_name("arr_0.npy").expect("filtered array");
    assert!(filtered.iter().all(|value| *value == 0 || *value == 1));
    assert!(filtered.iter().any(|value| *value == 1));
}

#[test]
fn apply_tracking_from_table_utility_writes_tracked_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let tracking_path = temp.path().join("tracking.csv");
    let output_path = temp.path().join("tracked.npz");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            0u32, 2, //
            2, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");
    fs::write(&tracking_path, "frame_i,track_id,mask_id\n0,5,2\n").expect("tracking csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--apply_tracking_from_table")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--tracking_table_path")
        .arg(&tracking_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--segm_layout")
        .arg("TYX")
        .arg("--mask_ids_col")
        .arg("mask_id")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved tracked segmentation mask"));
    let mut npz = NpzReader::new(File::open(output_path).expect("tracked npz")).expect("read npz");
    let tracked: Array3<u32> = npz.by_name("arr_0.npy").expect("tracked array");
    assert!(tracked.iter().all(|value| *value == 0 || *value == 5));
    assert!(tracked.iter().any(|value| *value == 5));
}

#[test]
fn add_lineage_tree_utility_writes_tree_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("acdc_output.csv");
    let output_path = temp.path().join("lineage_tree.csv");
    fs::write(
        &input_path,
        concat!(
            "frame_i,Cell_ID,cell_cycle_stage,generation_num,relative_ID,relationship,is_history_known\n",
            "0,1,G1,1,-1,mother,1\n",
            "1,1,S,1,-1,mother,1\n",
        ),
    )
    .expect("acdc output csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--add_lineage_tree")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved lineage-tree table"));
    let csv = fs::read_to_string(output_path).expect("lineage tree csv");
    assert!(csv.contains("Cell_ID_tree"));
    assert!(csv.contains("root_ID_tree"));
    assert!(csv.contains("sister_ID_tree"));
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
