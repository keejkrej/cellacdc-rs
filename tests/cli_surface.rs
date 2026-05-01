use cellacdc_rs::read_table;
use ndarray::{Array2, Array3, ArrayD};
use ndarray_npy::{read_npy, NpzReader, NpzWriter};
use std::fs::{self, File};
use std::process::Command;
use tiff::encoder::{colortype, TiffEncoder};

fn run_bin(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .args(args)
        .output()
        .expect("run binary")
}

fn write_test_tiff(path: &std::path::Path, values: &[u16], width: u32, height: u32) {
    let file = File::create(path).expect("tiff file");
    let mut encoder = TiffEncoder::new(file).expect("tiff encoder");
    encoder
        .write_image::<colortype::Gray16>(width, height, values)
        .expect("write tiff image");
}

fn write_test_tiff_stack(path: &std::path::Path, frames: &[Vec<u16>], width: u32, height: u32) {
    let file = File::create(path).expect("tiff file");
    let mut encoder = TiffEncoder::new(file).expect("tiff encoder");
    for frame in frames {
        encoder
            .write_image::<colortype::Gray16>(width, height, frame)
            .expect("write tiff image");
    }
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
    assert!(stdout.contains("--to_obj_coords"));
    assert!(stdout.contains("--fill_holes"));
    assert!(stdout.contains("--connect_3d_segm"));
    assert!(stdout.contains("--stack_2d_segm_to_3d"));
    assert!(stdout.contains("--filter_segm_from_table"));
    assert!(stdout.contains("--align_frames"));
    assert!(stdout.contains("--measure"));
    assert!(stdout.contains("--prepare_zstack_segm_info"));
    assert!(stdout.contains("--compute_background_roi_data"));
    assert!(stdout.contains("--inspect_frame"));
    assert!(stdout.contains("--export_frame_image"));
    assert!(stdout.contains("--export_frame_sequence"));
    assert!(stdout.contains("--apply_tracking_from_table"));
    assert!(stdout.contains("--apply_tracking_from_trackmate_xml"));
    assert!(stdout.contains("--add_lineage_tree"));
    assert!(stdout.contains("--build_lineage_state"));
    assert!(stdout.contains("--export_lineage_info"));
    assert!(stdout.contains("--propagate_lineage"));
    assert!(stdout.contains("--update_lineage_frame"));
    assert!(stdout.contains("--generate_mother_bud_total"));
    assert!(stdout.contains("--combine_metrics"));
    assert!(stdout.contains("--compute_multi_channel"));
    assert!(stdout.contains("--concat_acdc_outputs"));
    assert!(stdout.contains("--combine_channels"));
    assert!(stdout.contains("--convert_file_format"));
    assert!(stdout.contains("--rename_files"));
    assert!(stdout.contains("--import_experiment"));
    assert!(stdout.contains("--images_to_positions"));
    assert!(stdout.contains("--move_channel_tiffs_to_positions"));
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
fn count_objects_utility_counts_experiment_positions() {
    let temp = tempfile::tempdir().expect("temp dir");
    for (pos, object_id) in [("Position_1", 1u32), ("Position_2", 2u32)] {
        let images = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images).expect("images dir");
        let segm_path = images.join("demo_segm.npz");
        let file = File::create(&segm_path).expect("segm npz");
        let mut writer = NpzWriter::new(file);
        let masks = Array2::from_shape_vec(
            (2, 2),
            vec![
                0u32, object_id, //
                object_id, 0,
            ],
        )
        .expect("mask shape");
        writer.add_array("arr_0", &masks).expect("write mask");
        writer.finish().expect("finish npz");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--count_objects")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved object counts for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let csv = fs::read_to_string(
            temp.path()
                .join(pos)
                .join("Images")
                .join("demo_acdc_objects_count.csv"),
        )
        .expect("counts csv");
        assert!(csv.contains("In current position"));
        assert!(csv.contains("\n1\n"));
    }
}

#[test]
fn to_obj_coords_utility_writes_coordinate_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let segm_path = temp.path().join("segm.npz");
    let output_path = temp.path().join("objects_coordinates.csv");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            0u32, 1, //
            2, 2, //
            3, 0, //
            3, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--to_obj_coords")
        .arg("--segmentation_path")
        .arg(&segm_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--segm_layout")
        .arg("TYX")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved object-coordinate table"));
    let csv = fs::read_to_string(output_path).expect("object coordinate csv");
    assert!(csv.contains("frame_i,Cell_ID,y,x"));
    assert!(csv.contains("1,3,1,0"));
}

#[test]
fn to_obj_coords_utility_writes_experiment_position_tables() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        let segm_path = images_dir.join("demo_segm.npz");
        let file = File::create(segm_path).expect("segm npz");
        let mut writer = NpzWriter::new(file);
        let masks = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                0u32, 1, //
                2, 2, //
                3, 0, //
                3, 0,
            ],
        )
        .expect("mask shape");
        writer.add_array("arr_0", &masks).expect("write mask");
        writer.finish().expect("finish npz");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--to_obj_coords")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--segm_layout")
        .arg("TYX")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved object-coordinate tables for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let output_path = temp
            .path()
            .join(pos)
            .join("Images")
            .join("demo_objects_coordinates.csv");
        let csv = fs::read_to_string(output_path).expect("object coordinate csv");
        assert!(csv.contains("frame_i,Cell_ID,y,x"));
        assert!(csv.contains("1,3,1,0"));
    }
}

#[test]
fn convert_file_format_utility_converts_npz_to_npy_with_segm_cast() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("segm_float.npz");
    let output_path = temp.path().join("segm_uint32.npy");
    let file = File::create(&input_path).expect("input npz");
    let mut writer = NpzWriter::new(file);
    let values = Array2::from_shape_vec((2, 2), vec![0.0f32, 1.2, 2.0, 0.0]).expect("shape");
    writer.add_array("arr_0", &values).expect("write input");
    writer.finish().expect("finish input");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--convert_file_format")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--cast_segm_uint32")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved converted file"));
    let converted: ndarray::ArrayD<u32> = read_npy(&output_path).expect("converted npy");
    assert_eq!(converted.shape(), &[2, 2]);
    assert_eq!(
        converted.iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 0]
    );
}

#[test]
fn rename_files_utility_appends_text_to_selected_files() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("demo_phase.tif");
    fs::write(&input_path, b"not really a tiff").expect("input file");
    let output_path = temp.path().join("demo_phase_aligned.tif");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--rename_files")
        .arg("--file_path")
        .arg(&input_path)
        .arg("--rename_append_text")
        .arg("aligned")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Renamed 1 file(s)"));
    assert!(!input_path.exists());
    assert!(output_path.exists());
    assert_eq!(
        fs::read(&output_path).expect("renamed content"),
        b"not really a tiff"
    );
}

#[test]
fn images_to_positions_utility_creates_position_folders() {
    let temp = tempfile::tempdir().expect("temp dir");
    let source_dir = temp.path().join("raw");
    let target_dir = temp.path().join("experiment");
    fs::create_dir_all(&source_dir).expect("source dir");
    write_test_tiff(&source_dir.join("first.tif"), &[1, 2, 3, 4], 2, 2);
    write_test_tiff(&source_dir.join("second.tif"), &[5, 6, 7, 8], 2, 2);
    fs::write(source_dir.join("notes.txt"), b"skip me").expect("invalid file");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--images_to_positions")
        .arg("--source_dir")
        .arg(&source_dir)
        .arg("--target_dir")
        .arg(&target_dir)
        .arg("--images_append_text")
        .arg("GFP")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Converted 2 image file(s) to Position folders"));
    assert!(target_dir
        .join("Position_1")
        .join("Images")
        .join("s01_first_GFP.tif")
        .exists());
    assert!(target_dir
        .join("Position_2")
        .join("Images")
        .join("s02_second_GFP.tif")
        .exists());
}

#[test]
fn move_channel_tiffs_to_positions_utility_groups_flat_channel_files() {
    let temp = tempfile::tempdir().expect("temp dir");
    let source_dir = temp.path();
    write_test_tiff(&source_dir.join("pos2_GFP.tif"), &[1, 2, 3, 4], 2, 2);
    write_test_tiff(&source_dir.join("pos2_RFP.tif"), &[5, 6, 7, 8], 2, 2);
    write_test_tiff(&source_dir.join("pos10_GFP.tif"), &[9, 10, 11, 12], 2, 2);
    write_test_tiff(&source_dir.join("pos10_RFP.tif"), &[13, 14, 15, 16], 2, 2);
    fs::write(
        source_dir.join("pos2_metadata.csv"),
        "Description,values\nbasename,old_\nSizeT,1\n",
    )
    .expect("metadata");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--move_channel_tiffs_to_positions")
        .arg("--source_dir")
        .arg(source_dir)
        .arg("--channel_name")
        .arg("GFP")
        .arg("--channel_name")
        .arg("RFP")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Moved channel TIFFs into 2 position folder(s)"));
    assert!(source_dir
        .join("Position_1")
        .join("Images")
        .join("pos2_GFP.tif")
        .exists());
    assert!(source_dir
        .join("Position_1")
        .join("Images")
        .join("pos2_RFP.tif")
        .exists());
    assert!(source_dir
        .join("Position_2")
        .join("Images")
        .join("pos10_GFP.tif")
        .exists());
    let metadata = fs::read_to_string(
        source_dir
            .join("Position_1")
            .join("Images")
            .join("pos2_metadata.csv"),
    )
    .expect("moved metadata");
    assert!(metadata.contains("basename,pos2_"));
    assert!(!source_dir.join("pos2_GFP.tif").exists());
}

#[test]
fn import_experiment_utility_creates_position_structure() {
    let temp = tempfile::tempdir().expect("temp dir");
    let source_dir = temp.path().join("raw");
    let target_dir = temp.path().join("imported_experiment");
    fs::create_dir_all(&source_dir).expect("source dir");
    write_test_tiff(&source_dir.join("demo_phase.tif"), &[1, 2, 3, 4], 2, 2);

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--import_experiment")
        .arg("--import_source")
        .arg(&source_dir)
        .arg("--target_dir")
        .arg(&target_dir)
        .arg("--import_layout")
        .arg("file_per_position")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Imported 1 position(s)"));
    assert!(stdout.contains("Discovered 1 import source(s)"));
    let images_dir = target_dir.join("Position_1").join("Images");
    assert!(images_dir.join("s01_phase.tif").exists());
    assert!(images_dir.join("s01_metadata.csv").exists());
    assert!(images_dir.join("s01_metadataXML.txt").exists());
    let metadata = fs::read_to_string(images_dir.join("s01_metadata.csv")).expect("metadata csv");
    assert!(metadata.contains("basename,s01_"));
    assert!(metadata.contains("SizeT,1"));
    assert!(metadata.contains("channel_0_name,phase"));
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
fn fill_holes_utility_fills_experiment_positions() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images).expect("images dir");
        let segm_path = images.join("demo_segm.npz");
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
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--fill_holes")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--segm_append_name")
        .arg("filled")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved hole-filled segmentation masks for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let output_path = temp
            .path()
            .join(pos)
            .join("Images")
            .join("demo_segm_filled.npz");
        let mut npz =
            NpzReader::new(File::open(output_path).expect("filled npz")).expect("read npz");
        let filled: Array2<u32> = npz.by_name("arr_0.npy").expect("filled array");
        assert_eq!(filled[[1, 1]], 1);
    }
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
fn connect_3d_segm_utility_connects_experiment_positions() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        let segm_path = images_dir.join("demo_segm.npz");
        let file = File::create(segm_path).expect("segm npz");
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
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--connect_3d_segm")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--segm_append_name")
        .arg("connected3d")
        .arg("--segm_layout")
        .arg("ZYX")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved 3D-connected segmentation masks for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let output_path = temp
            .path()
            .join(pos)
            .join("Images")
            .join("demo_segm_connected3d.npz");
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
fn stack_2d_segm_to_3d_utility_stacks_experiment_positions() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        let segm_path = images_dir.join("demo_segm.npz");
        let file = File::create(segm_path).expect("segm npz");
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
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--stack_2d_segm_to_3d")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--segm_append_name")
        .arg("stacked3d")
        .arg("--size_z")
        .arg("3")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved 2D segmentation masks stacked to 3D for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let output_path = temp
            .path()
            .join(pos)
            .join("Images")
            .join("demo_segm_stacked3d.npz");
        let mut npz = NpzReader::new(File::open(output_path).expect("stacked npz"))
            .expect("read stacked npz");
        let stacked: Array3<u32> = npz.by_name("arr_0.npy").expect("stacked array");
        assert_eq!(stacked.shape(), &[3, 2, 2]);
        for z in 0..3 {
            assert_eq!(stacked[[z, 0, 1]], 1);
            assert_eq!(stacked[[z, 1, 0]], 2);
        }
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
fn filter_segm_from_table_utility_filters_experiment_positions() {
    let temp = tempfile::tempdir().expect("temp dir");
    let coords_path = temp.path().join("coords.csv");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        let segm_path = images_dir.join("demo_segm.npz");
        let file = File::create(segm_path).expect("segm npz");
        let mut writer = NpzWriter::new(file);
        let masks = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1u32, 1, 0, //
                2, 2, 0, //
                3, 3, 0, //
                4, 4, 0,
            ],
        )
        .expect("mask shape");
        writer.add_array("arr_0", &masks).expect("write mask");
        writer.finish().expect("finish npz");
    }
    fs::write(
        &coords_path,
        "Position_n,frame_i,x,y\nPosition_1,0,0,0\nPosition_1,1,0,0\nPosition_2,0,0,1\nPosition_2,1,0,1\n",
    )
    .expect("coords csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--filter_segm_from_table")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--coords_table_path")
        .arg(&coords_path)
        .arg("--segm_append_name")
        .arg("filtered")
        .arg("--segm_layout")
        .arg("TYX")
        .arg("--frame_col")
        .arg("frame_i")
        .arg("--position_col")
        .arg("Position_n")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved coordinate-filtered segmentation masks for 2 position(s)"));
    let expected = [
        ("Position_1", vec![1u32, 1, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0]),
        ("Position_2", vec![0u32, 0, 0, 2, 2, 0, 0, 0, 0, 4, 4, 0]),
    ];
    for (pos, values) in expected {
        let output_path = temp
            .path()
            .join(pos)
            .join("Images")
            .join("demo_segm_filtered.npz");
        let mut npz =
            NpzReader::new(File::open(output_path).expect("filtered npz")).expect("read npz");
        let filtered: Array3<u32> = npz.by_name("arr_0.npy").expect("filtered array");
        assert_eq!(filtered.iter().copied().collect::<Vec<_>>(), values);
    }
}

#[test]
fn align_frames_utility_writes_experiment_aligned_outputs() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        fs::write(
            images_dir.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )
        .expect("metadata csv");
        for channel in ["phase", "GFP"] {
            let path = images_dir.join(format!("demo_{channel}.tif"));
            write_test_tiff_stack(
                &path,
                &[
                    vec![
                        0, 1, //
                        0, 0,
                    ],
                    vec![
                        1, 0, //
                        0, 0,
                    ],
                ],
                2,
                2,
            );
        }
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--align_frames")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--reference_channel")
        .arg("phase")
        .arg("--channel_name")
        .arg("phase")
        .arg("--channel_name")
        .arg("GFP")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Aligned 4 channel output(s) across selected position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        assert!(images_dir.join("demo_phase_aligned.npz").exists());
        assert!(images_dir.join("demo_GFP_aligned.npz").exists());
        let shifts: Array2<i32> =
            read_npy(images_dir.join("demo_align_shift.npy")).expect("alignment shifts");
        assert_eq!(shifts.shape(), &[2, 2]);
    }
}

#[test]
fn measure_utility_writes_experiment_acdc_outputs() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        fs::write(
            images_dir.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\nTimeIncrement,15\nPhysicalSizeX,1\nPhysicalSizeY,1\n",
        )
        .expect("metadata csv");
        write_test_tiff_stack(
            &images_dir.join("demo_phase.tif"),
            &[vec![10, 10, 0, 0], vec![20, 20, 0, 0]],
            2,
            2,
        );
        write_test_tiff_stack(
            &images_dir.join("demo_gfp.tif"),
            &[vec![30, 30, 0, 0], vec![40, 40, 0, 0]],
            2,
            2,
        );
        let file = File::create(images_dir.join("demo_segm.npz")).expect("segm npz");
        let mut writer = NpzWriter::new(file);
        let masks = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1u32, 1, //
                0, 0, //
                1, 1, //
                0, 0,
            ],
        )
        .expect("mask shape");
        writer.add_array("arr_0", &masks).expect("write mask");
        writer.finish().expect("finish npz");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--measure")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--segm_endname")
        .arg("segm")
        .arg("--channel_name")
        .arg("phase")
        .arg("--save_object_counts")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Computed measurements for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        let csv =
            fs::read_to_string(images_dir.join("demo_acdc_output.csv")).expect("acdc output csv");
        assert!(csv.contains("phase_mean"));
        assert!(!csv.contains("gfp_mean"));
        assert!(images_dir.join("demo_acdc_objects_count.csv").exists());
    }
}

#[test]
fn prepare_zstack_segm_info_utility_writes_experiment_tables() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        fs::write(
            images_dir.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,5\n",
        )
        .expect("metadata csv");
        write_test_tiff_stack(
            &images_dir.join("demo_phase.tif"),
            &vec![vec![0, 1, 0, 0]; 10],
            2,
            2,
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--prepare_zstack_segm_info")
        .arg("--experiment_dir")
        .arg(temp.path())
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Prepared z-stack segmInfo files for 2 position(s)"));
    for pos in ["Position_1", "Position_2"] {
        let csv = fs::read_to_string(
            temp.path()
                .join(pos)
                .join("Images")
                .join("demo_segmInfo.csv"),
        )
        .expect("segm info csv");
        assert!(csv.contains("filename,frame_i,z_slice_used_dataPrep"));
        assert!(csv.contains("demo_phase.tif,0,2,single z-slice,1,2"));
        assert!(csv.contains("demo_phase.tif,1,2,single z-slice,1,2"));
    }
}

#[test]
fn compute_background_roi_data_utility_writes_npz_archive() {
    let temp = tempfile::tempdir().expect("temp dir");
    let images_dir = temp.path().join("Position_1").join("Images");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_metadata.csv"),
        "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
    )
    .expect("metadata csv");
    write_test_tiff_stack(
        &images_dir.join("demo_phase.tif"),
        &[vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10, 11, 12]],
        2,
        3,
    );
    fs::write(
        images_dir.join("demo_dataPrep_bkgrROIs.json"),
        r#"[{"pos":[1,0],"size":[2,2]}]"#,
    )
    .expect("background roi json");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--compute_background_roi_data")
        .arg("--position_dir")
        .arg(temp.path().join("Position_1"))
        .arg("--channel_name")
        .arg("phase")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Computed background ROI data for 1 channel output(s)"));
    let output_path = images_dir.join("demo_phase_bkgrRoiData.npz");
    assert!(output_path.exists());
    let file = File::open(output_path).expect("background roi npz");
    let mut npz = NpzReader::new(file).expect("read background roi npz");
    let roi_data: ArrayD<f32> = npz.by_name("roi0_data.npy").expect("roi0 data");
    assert_eq!(roi_data.shape(), &[2, 2, 1]);
}

#[test]
fn inspect_frame_utility_prints_frame_json() {
    let temp = tempfile::tempdir().expect("temp dir");
    let position_dir = temp.path().join("Position_1");
    let images_dir = position_dir.join("Images");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_metadata.csv"),
        "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\nPhysicalSizeX,0.5\nPhysicalSizeY,0.25\nTimeIncrement,12\n",
    )
    .expect("metadata csv");
    write_test_tiff(
        &images_dir.join("demo_phase.tif"),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        3,
        3,
    );
    let file = File::create(images_dir.join("demo_segm.npz")).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (3, 3),
        vec![
            0u32, 1, 1, //
            0, 2, 2, //
            0, 2, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--inspect_frame")
        .arg("--position_dir")
        .arg(&position_dir)
        .arg("--frame_i")
        .arg("0")
        .arg("--selected_label")
        .arg("2")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("frame inspection json");
    assert_eq!(payload["frame_index"].as_u64(), Some(0));
    assert_eq!(payload["time_seconds"].as_f64(), Some(0.0));
    assert_eq!(payload["object_count"].as_u64(), Some(2));
    assert_eq!(payload["available_labels"], serde_json::json!([1, 2]));
    let object = &payload["selected_object"];
    assert_eq!(object["label"].as_u64(), Some(2));
    assert_eq!(object["area_pixels"].as_u64(), Some(3));
    assert_eq!(object["area_um2"].as_f64(), Some(0.375));
    assert_eq!(object["bbox_min_x"].as_u64(), Some(1));
    assert_eq!(object["bbox_min_y"].as_u64(), Some(1));
    assert_eq!(object["bbox_max_x"].as_u64(), Some(2));
    assert_eq!(object["bbox_max_y"].as_u64(), Some(2));
    assert_eq!(object["channel_sum"]["phase"].as_f64(), Some(19.0));
    assert_eq!(object["channel_mean"]["phase"].as_f64(), Some(19.0 / 3.0));
}

#[test]
fn export_frame_image_utility_writes_rendered_png() {
    let temp = tempfile::tempdir().expect("temp dir");
    let position_dir = temp.path().join("Position_1");
    let images_dir = position_dir.join("Images");
    let output_path = temp.path().join("frame.png");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_metadata.csv"),
        "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\nPhysicalSizeX,0.5\nPhysicalSizeY,0.25\nTimeIncrement,12\n",
    )
    .expect("metadata csv");
    write_test_tiff(
        &images_dir.join("demo_phase.tif"),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        3,
        3,
    );
    let file = File::create(images_dir.join("demo_segm.npz")).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array2::from_shape_vec(
        (3, 3),
        vec![
            0u32, 1, 1, //
            0, 2, 2, //
            0, 2, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--export_frame_image")
        .arg("--position_dir")
        .arg(&position_dir)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--frame_i")
        .arg("0")
        .arg("--channel_name")
        .arg("phase")
        .arg("--selected_label")
        .arg("2")
        .arg("--show_labels")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Exported frame image to"));
    let image = image::open(&output_path).expect("exported png").to_rgba8();
    assert_eq!(image.dimensions(), (3, 3));
    assert!(image
        .pixels()
        .any(|pixel| pixel[0] != pixel[1] || pixel[1] != pixel[2]));
}

#[test]
fn export_frame_sequence_utility_writes_png_frames() {
    let temp = tempfile::tempdir().expect("temp dir");
    let position_dir = temp.path().join("Position_1");
    let images_dir = position_dir.join("Images");
    let output_dir = temp.path().join("frames");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_metadata.csv"),
        "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\nPhysicalSizeX,1\nPhysicalSizeY,1\nTimeIncrement,3\n",
    )
    .expect("metadata csv");
    write_test_tiff_stack(
        &images_dir.join("demo_phase.tif"),
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        2,
        2,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--export_frame_sequence")
        .arg("--position_dir")
        .arg(&position_dir)
        .arg("--output_path")
        .arg(&output_dir)
        .arg("--channel_name")
        .arg("phase")
        .arg("--start_frame")
        .arg("0")
        .arg("--end_frame")
        .arg("1")
        .arg("--no_overlay")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Exported 2 frame image(s)"));
    for frame_name in ["frame_0000.png", "frame_0001.png"] {
        let image = image::open(output_dir.join(frame_name))
            .expect("exported sequence frame")
            .to_rgba8();
        assert_eq!(image.dimensions(), (2, 2));
    }
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
fn apply_tracking_from_trackmate_xml_utility_writes_tracked_mask() {
    let temp = tempfile::tempdir().expect("temp dir");
    let position_dir = temp.path().join("Position_1");
    let images_dir = position_dir.join("Images");
    fs::create_dir_all(&images_dir).expect("images dir");
    let segm_path = images_dir.join("demo_segm.npz");
    let output_path = temp.path().join("tracked.npz");
    let xml_path = temp.path().join("tracks.xml");
    let file = File::create(&segm_path).expect("segm npz");
    let mut writer = NpzWriter::new(file);
    let masks = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            1u32, 1, 0, 0, //
            2, 2, 0, 0,
        ],
    )
    .expect("mask shape");
    writer.add_array("arr_0", &masks).expect("write mask");
    writer.finish().expect("finish npz");
    fs::write(
        &xml_path,
        r#"<Tracks>
<particle>
  <detection t="0" x="0" y="0" z="0"/>
  <detection t="1" x="0" y="0" z="0"/>
</particle>
</Tracks>"#,
    )
    .expect("trackmate xml");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--apply_tracking_from_trackmate_xml")
        .arg("--position_dir")
        .arg(&position_dir)
        .arg("--segm_endname")
        .arg("segm")
        .arg("--xml_path")
        .arg(&xml_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved TrackMate-tracked segmentation mask"));
    assert!(stdout.contains("Saved tracking sidecar"));
    let mut npz = NpzReader::new(File::open(output_path).expect("tracked npz")).expect("read npz");
    let tracked: Array3<u32> = npz.by_name("arr_0.npy").expect("tracked array");
    assert!(tracked.iter().all(|value| *value == 0 || *value == 1));
    assert!(tracked.iter().any(|value| *value == 1));
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
fn build_lineage_state_utility_writes_tree_columns() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("acdc_output.csv");
    let output_path = temp.path().join("lineage_state.csv");
    fs::write(&input_path, "frame_i,Cell_ID,value\n0,1,10\n0,2,20\n").expect("acdc output csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--build_lineage_state")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved lineage-state table to"));
    let table = read_table(&output_path).expect("lineage state csv");
    for column in [
        "Cell_ID_tree",
        "generation_num_tree",
        "parent_ID_tree",
        "root_ID_tree",
        "is_history_known",
    ] {
        assert!(table.headers.iter().any(|header| header == column));
    }
    let row = table.row_map(0);
    assert_eq!(row["frame_i"].as_i64(), Some(0));
    assert_eq!(row["Cell_ID"].as_i64(), Some(1));
    assert_eq!(row["value"].as_i64(), Some(10));
    assert_eq!(row["Cell_ID_tree"].as_i64(), Some(1));
    assert_eq!(row["generation_num_tree"].as_i64(), Some(1));
    assert_eq!(row["parent_ID_tree"].as_i64(), Some(-1));
    assert_eq!(row["root_ID_tree"].as_i64(), Some(1));
    assert_eq!(row["is_history_known"].as_string_lossy(), "false");
}

#[test]
fn export_lineage_info_utility_writes_frame_json() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("acdc_output.csv");
    let output_path = temp.path().join("frame1_lineage_info.json");
    fs::write(
        &input_path,
        concat!(
            "frame_i,Cell_ID,Cell_ID_tree,generation_num_tree,parent_ID_tree,root_ID_tree,sister_ID_tree,is_history_known\n",
            "0,1,1,1,-1,1,-1,false\n",
            "0,2,2,1,-1,2,-1,false\n",
            "1,2,2,1,-1,2,-1,false\n",
            "1,3,3,2,1,1,-1,true\n",
            "1,4,4,1,-1,4,-1,false\n",
        ),
    )
    .expect("acdc output csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--export_lineage_info")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--frame_i")
        .arg("1")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved lineage frame info to"));
    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output_path).expect("lineage json"))
            .expect("parse lineage json");
    assert_eq!(
        json["cells_with_parent"],
        serde_json::json!([{"cell_id": 3, "parent_id": 1}])
    );
    assert_eq!(json["orphan_cells"], serde_json::json!([4]));
    assert_eq!(json["lost_cells"], serde_json::json!([]));
}

#[test]
fn propagate_lineage_utility_writes_updated_future_rows() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("edited_acdc_output.csv");
    let output_path = temp.path().join("propagated_acdc_output.csv");
    fs::write(
        &input_path,
        concat!(
            "frame_i,Cell_ID,Cell_ID_tree,generation_num_tree,parent_ID_tree,root_ID_tree,sister_ID_tree,is_history_known,cell_area\n",
            "0,1,10,1,-1,1,-1,false,2\n",
            "0,2,20,2,10,1,-1,true,3\n",
            "1,1,99,1,-1,1,-1,false,4\n",
            "1,2,20,5,99,8,-1,true,5\n",
        ),
    )
    .expect("lineage csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--propagate_lineage")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--frame_i")
        .arg("0")
        .arg("--cell_id")
        .arg("1")
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved propagated lineage table to"));
    let csv = fs::read_to_string(output_path).expect("propagated csv");
    assert!(csv.contains("1,1,10,1,-1,1,-1,false,4"));
    assert!(csv.contains("1,2,20,2,10,1,-1,true,5"));
}

#[test]
fn update_lineage_frame_utility_applies_json_edits() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("acdc_output.csv");
    let edits_path = temp.path().join("lineage_edits.json");
    let output_path = temp.path().join("updated_acdc_output.csv");
    fs::write(
        &input_path,
        concat!(
            "frame_i,Cell_ID,Cell_ID_tree,generation_num_tree,parent_ID_tree,root_ID_tree,sister_ID_tree,is_history_known\n",
            "0,1,1,1,-1,1,-1,false\n",
            "0,2,2,1,-1,2,-1,false\n",
        ),
    )
    .expect("lineage csv");
    fs::write(
        &edits_path,
        r#"[{"Cell_ID":2,"parent_ID_tree":1,"is_history_known":true}]"#,
    )
    .expect("lineage edits");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--update_lineage_frame")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--frame_i")
        .arg("0")
        .arg("--edits_json_path")
        .arg(&edits_path)
        .output()
        .expect("run binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved updated lineage table to"));
    let csv = fs::read_to_string(output_path).expect("updated lineage csv");
    assert!(csv.contains("0,2,2,2,1,1,-1,true"));
}

#[test]
fn add_lineage_tree_utility_updates_position_tables_in_experiment() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images).expect("images dir");
        fs::write(
            images.join("demo_acdc_output.csv"),
            concat!(
                "frame_i,Cell_ID,cell_cycle_stage,generation_num,relative_ID,relationship,is_history_known\n",
                "0,1,G1,1,-1,mother,1\n",
                "1,1,S,1,-1,mother,1\n",
            ),
        )
        .expect("acdc output csv");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--add_lineage_tree")
        .arg("--experiment_dir")
        .arg(temp.path())
        .arg("--table_endname")
        .arg("acdc_output")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Added lineage-tree columns to 2 table(s)"));
    for pos in ["Position_1", "Position_2"] {
        let csv = fs::read_to_string(
            temp.path()
                .join(pos)
                .join("Images")
                .join("demo_acdc_output.csv"),
        )
        .expect("updated acdc output");
        assert!(csv.contains("Cell_ID_tree"));
        assert!(csv.contains("root_ID_tree"));
    }
}

#[test]
fn generate_mother_bud_total_utility_writes_total_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input_path = temp.path().join("acdc_output.csv");
    let output_path = temp.path().join("mother_bud_total.csv");
    fs::write(
        &input_path,
        concat!(
            "frame_i,Cell_ID,relative_ID,cell_cycle_stage,relationship,cell_area_um2\n",
            "0,1,-1,G1,mother,10\n",
            "1,1,2,S,mother,10\n",
            "1,2,1,S,bud,5\n",
        ),
    )
    .expect("acdc output csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--generate_mother_bud_total")
        .arg("--input_path")
        .arg(&input_path)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--column_operation")
        .arg("cell_area_um2=sum")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved mother-bud-total table"));
    let csv = fs::read_to_string(output_path).expect("mother bud total csv");
    assert!(csv.contains("entity"));
    assert!(csv.contains("Total"));
    assert!(csv.contains(",15"));
}

#[test]
fn combine_metrics_utility_writes_formula_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let source1 = temp.path().join("source1.csv");
    let source2 = temp.path().join("source2.csv");
    let output_path = temp.path().join("combined.csv");
    fs::write(&source1, "frame_i,Cell_ID,signal\n0,1,2\n").expect("source1 csv");
    fs::write(&source2, "frame_i,Cell_ID,signal\n0,1,3\n").expect("source2 csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--combine_metrics")
        .arg("--source_path")
        .arg(&source1)
        .arg("--source_path")
        .arg(&source2)
        .arg("--output_path")
        .arg(&output_path)
        .arg("--formula")
        .arg("sum_signal=table1_signal + table2_signal")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved combined metrics table"));
    assert!(stdout.contains("Saved combined metrics equations"));
    let csv = fs::read_to_string(output_path).expect("combined csv");
    assert!(csv.contains("sum_signal"));
    assert!(csv.contains(",5"));
}

#[test]
fn compute_multi_channel_utility_writes_position_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let images_dir = temp.path().join("Position_1").join("Images");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_acdc_output_first.csv"),
        "frame_i,Cell_ID,signal\n0,1,2\n",
    )
    .expect("first csv");
    fs::write(
        images_dir.join("demo_acdc_output_second.csv"),
        "frame_i,Cell_ID,signal\n0,1,3\n",
    )
    .expect("second csv");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--compute_multi_channel")
        .arg("--position_dir")
        .arg(temp.path().join("Position_1"))
        .arg("--source_endname")
        .arg("acdc_output_first")
        .arg("--source_endname")
        .arg("acdc_output_second")
        .arg("--formula")
        .arg("sum_signal=signal_table1 + signal_table2")
        .arg("--append_name")
        .arg("combined_metrics")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Computed multi-channel metrics for 1 position(s)"));
    let csv = fs::read_to_string(images_dir.join("demo_acdc_output_combined_metrics.csv"))
        .expect("combined metrics csv");
    assert!(csv.contains("sum_signal"));
    assert!(csv.contains(",5"));
    assert!(images_dir
        .join("demo_equations_combined_metrics.ini")
        .exists());
}

#[test]
fn concat_acdc_outputs_utility_writes_allpos_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    for pos in ["Position_1", "Position_2"] {
        let images_dir = temp.path().join(pos).join("Images");
        fs::create_dir_all(&images_dir).expect("images dir");
        let cell_id = if pos.ends_with('1') { 1 } else { 2 };
        fs::write(
            images_dir.join("demo_acdc_output.csv"),
            format!("frame_i,Cell_ID,value\n0,{cell_id},{pos}\n"),
        )
        .expect("acdc output csv");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--concat_acdc_outputs")
        .arg("--concat_experiment_dir")
        .arg(temp.path())
        .arg("--table_endname")
        .arg("acdc_output")
        .arg("--selected_column")
        .arg("Position_n")
        .arg("--selected_column")
        .arg("Cell_ID")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Saved concatenated position table"));
    let csv = fs::read_to_string(
        temp.path()
            .join("AllPos_acdc_output")
            .join("AllPos_acdc_output.csv"),
    )
    .expect("allpos csv");
    assert!(csv.contains("Position_n,Cell_ID"));
    assert!(csv.contains("Position_1,1"));
    assert!(csv.contains("Position_2,2"));
}

#[test]
fn combine_channels_utility_writes_recipe_output() {
    let temp = tempfile::tempdir().expect("temp dir");
    let position_dir = temp.path().join("Position_1");
    let images_dir = position_dir.join("Images");
    fs::create_dir_all(&images_dir).expect("images dir");
    fs::write(
        images_dir.join("demo_metadata.csv"),
        "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
    )
    .expect("metadata");
    write_test_tiff(
        &images_dir.join("demo_ch1.tif"),
        &[0, 0, u16::MAX, u16::MAX],
        2,
        2,
    );
    let recipe_path = temp.path().join("recipe.json");
    fs::write(
        &recipe_path,
        r#"{
  "1": {
    "name": "img",
    "channel": "ch1",
    "binarize": "No",
    "min_val": 0.0,
    "max_val": 1.0
  },
  "formula": "img",
  "keep_input_data_type": true,
  "save_as_segm": false
}"#,
    )
    .expect("recipe json");

    let output = Command::new(env!("CARGO_BIN_EXE_cellacdc-rs"))
        .arg("--combine_channels")
        .arg("--position_dir")
        .arg(&position_dir)
        .arg("--recipe_path")
        .arg(&recipe_path)
        .arg("--append_name")
        .arg("combined")
        .output()
        .expect("run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Combined channels for 1 position(s)"));
    assert!(images_dir.join("demo_combined.tif").exists());
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
