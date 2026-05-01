# `cellacdc-rs` CLI

The public CLI now follows the same top-level shape as Python `acdc`.

## Usage

```bash
cellacdc-rs
cellacdc-rs -p workflow.ini
cellacdc-rs -v
cellacdc-rs -info
cellacdc-rs --count_objects --segmentation_path demo_segm.npz --output_path demo_acdc_objects_count.csv
cellacdc-rs --count_objects --experiment_dir Experiment_1 --segm_endname segm
cellacdc-rs --to_obj_coords --segmentation_path demo_segm.npz --output_path demo_objects_coordinates.csv --segm_layout TYX
cellacdc-rs --to_obj_coords --experiment_dir Experiment_1 --segm_endname segm --segm_layout TYX
cellacdc-rs --fill_holes --segmentation_path demo_segm.npz --output_path demo_segm_filled.npz
cellacdc-rs --fill_holes --experiment_dir Experiment_1 --segm_endname segm --segm_append_name filled
cellacdc-rs --connect_3d_segm --segmentation_path demo_segm3d.npz --output_path demo_segm3d_connected.npz --segm_layout ZYX
cellacdc-rs --connect_3d_segm --experiment_dir Experiment_1 --segm_endname segm --segm_append_name connected3d --segm_layout ZYX
cellacdc-rs --stack_2d_segm_to_3d --segmentation_path demo_segm2d.npz --output_path demo_segm3d.npz --size_z 5
cellacdc-rs --stack_2d_segm_to_3d --experiment_dir Experiment_1 --segm_endname segm --segm_append_name stacked3d --size_z 5
cellacdc-rs --filter_segm_from_table --segmentation_path demo_segm.npz --coords_table_path coords.csv --output_path demo_segm_filtered.npz
cellacdc-rs --filter_segm_from_table --experiment_dir Experiment_1 --segm_endname segm --coords_table_path coords.csv --segm_append_name filtered --frame_col frame_i --position_col Position_n
cellacdc-rs --align_frames --experiment_dir Experiment_1 --reference_channel phase --channel_name phase --channel_name GFP
cellacdc-rs --measure --experiment_dir Experiment_1 --segm_endname segm --channel_name phase --save_object_counts
cellacdc-rs --prepare_zstack_segm_info --experiment_dir Experiment_1
cellacdc-rs --compute_background_roi_data --position_dir Position_1 --channel_name phase
cellacdc-rs --inspect_frame --position_dir Position_1 --frame_i 0 --selected_label 2
cellacdc-rs --export_frame_image --position_dir Position_1 --frame_i 0 --channel_name phase --output_path frame.png --selected_label 2 --show_labels
cellacdc-rs --export_frame_sequence --position_dir Position_1 --channel_name phase --output_path frames --start_frame 0 --end_frame 10 --no_overlay
cellacdc-rs --apply_tracking_from_table --segmentation_path demo_segm.npz --tracking_table_path tracking.csv --output_path demo_segm_tracked.npz --segm_layout TYX --mask_ids_col mask_id
cellacdc-rs --repeat_tracking --position_dir Position_1 --segm_endname segm --start_frame 5 --ioa_threshold 0.6
cellacdc-rs --apply_tracking_from_trackmate_xml --position_dir Position_1 --segm_endname segm --xml_path tracks.xml --output_path demo_segm_tracked.npz
cellacdc-rs --add_lineage_tree --input_path demo_acdc_output.csv --output_path demo_lineage_tree.csv
cellacdc-rs --add_lineage_tree --experiment_dir Experiment_1 --table_endname acdc_output
cellacdc-rs --build_lineage_state --input_path demo_acdc_output.csv --output_path demo_lineage_state.csv
cellacdc-rs --export_lineage_info --input_path demo_acdc_output.csv --output_path frame1_lineage_info.json --frame_i 1
cellacdc-rs --propagate_lineage --input_path edited_acdc_output.csv --output_path propagated_acdc_output.csv --frame_i 0 --cell_id 1
cellacdc-rs --update_lineage_frame --input_path demo_acdc_output.csv --output_path updated_acdc_output.csv --frame_i 0 --edits_json_path lineage_edits.json
cellacdc-rs --generate_mother_bud_total --input_path demo_acdc_output.csv --output_path demo_mother_bud_total.csv --column_operation cell_area_um2=sum
cellacdc-rs --combine_metrics --source_path channel1.csv --source_path channel2.csv --output_path combined.csv --formula "sum_signal=table1_signal + table2_signal"
cellacdc-rs --compute_multi_channel --position_dir Position_1 --source_endname acdc_output_first --source_endname acdc_output_second --formula "sum_signal=signal_table1 + signal_table2"
cellacdc-rs --concat_acdc_outputs --concat_experiment_dir Experiment_1 --table_endname acdc_output --output_format csv
cellacdc-rs --combine_channels --position_dir Position_1 --recipe_path combine_channels_recipe.json --append_name combined
cellacdc-rs --convert_file_format --input_path demo_segm.npz --output_path demo_segm.npy --cast_segm_uint32
cellacdc-rs --rename_files --file_path Position_1/Images/demo_phase.tif --rename_append_text aligned
cellacdc-rs --import_experiment --import_source raw_images --target_dir Experiment_1 --import_layout file_per_position
cellacdc-rs --images_to_positions --source_dir raw_images --target_dir Experiment_1 --images_append_text GFP
cellacdc-rs --move_channel_tiffs_to_positions --source_dir exported_tiffs --channel_name GFP --channel_name RFP
```

- No arguments: launch the desktop GUI.
- `-p, --params`: run a supported workflow INI file.
- `-v, --version`: print version and environment info.
- `-info, --info`: same as `--version`.
- `-d, --debug`: enable verbose logging for `--params` runs.
- `--reset`: reset the Python-compatible Cell-ACDC settings folder.
- `-y, --yes`: accepted as a Python-compatible auto-confirm flag. It is a no-op
  for non-interactive workflows and confirms `--reset`.
- `--install_details`: accepted as a Python installer compatibility flag. The
  JSON file is parsed and path-like fields are normalized, but Rust does not run
  Python installer commands.
- `--count_objects`: run the object-count utility on a segmentation mask and
  write the counts table to `--output_path`. Use `--segmentation_path` and
  `--output_path` for one explicit mask, or pass exactly one of
  `--position_dir` and `--experiment_dir` with `--segm_endname` to count
  matching segmentation files and write Python-style
  `*_acdc_objects_count*.csv` tables next to them. Optional layout hints
  `--size_t`, `--size_z`, and `--segm_layout` resolve ambiguous 3D masks.
- `--to_obj_coords`: convert every non-zero segmentation pixel/voxel to an
  object-coordinate table. Use `--segmentation_path` and `--output_path` for one
  explicit mask, or pass exactly one of `--position_dir` and `--experiment_dir`
  with `--segm_endname` to write Python-style `*_objects_coordinates.csv`
  tables next to matching segmentation files. Output columns are `frame_i`,
  `Cell_ID`, `y`, and `x`, with `z` included for 3D masks. It uses the same
  optional layout-hint flags as `--count_objects`.
- `--fill_holes`: fill holes in a segmentation mask. Use `--segmentation_path`
  and `--output_path` for one explicit mask, or pass exactly one of
  `--position_dir` and `--experiment_dir` with `--segm_endname` to fill matching
  segmentation files. Batch mode overwrites in place by default; use
  `--segm_append_name TEXT` to save appended output files instead. It uses the
  same optional layout-hint flags as `--count_objects`.
- `--connect_3d_segm`: connect labels across z-slice boundaries in a 3D
  segmentation mask. Use `--segmentation_path` and `--output_path` for one
  explicit mask, or pass exactly one of `--position_dir` and `--experiment_dir`
  with `--segm_endname` and `--segm_append_name` to write appended outputs for
  matching segmentation files. Use `--segm_layout ZYX` or `--segm_layout TZYX`
  when metadata does not make the layout unambiguous.
- `--stack_2d_segm_to_3d`: broadcast 2D segmentation masks into a 3D z-stack
  and write the stacked mask. Use `--segmentation_path`, `--output_path`, and
  target depth `--size_z` for one explicit mask, or pass exactly one of
  `--position_dir` and `--experiment_dir` with `--segm_endname`,
  `--segm_append_name`, and `--size_z` to write appended outputs for matching
  segmentation files.
- `--filter_segm_from_table`: keep only segmentation labels touched by
  coordinates in a CSV/XLSX table. Use `--segmentation_path`,
  `--coords_table_path`, and `--output_path` for one explicit mask, or pass
  exactly one of `--position_dir` and `--experiment_dir` with `--segm_endname`,
  `--coords_table_path`, and `--segm_append_name` to write appended outputs for
  matching segmentation files. Coordinate columns default to `--x_col x` and
  `--y_col y`; time-series/3D masks can also use `--frame_col`, `--z_col`,
  `--position_col`, and `--position_value`. In batch mode, `--position_col`
  defaults to each current `Position_n` folder unless `--position_value` is
  provided.
- `--align_frames`: compute frame-alignment shifts from `--reference_channel`
  and write `*_aligned.npz` outputs plus `*align_shift.npy` for exactly one of
  `--position_dir` or `--experiment_dir`. Repeat `--channel_name` to align a
  subset of channels; when omitted, all discovered channels are aligned.
  Existing aligned outputs are protected unless `--yes` is provided.
- `--measure`: compute Cell-ACDC measurement tables for exactly one of
  `--position_dir` or `--experiment_dir`. `--segm_endname` selects the
  segmentation, repeated `--channel_name` limits measured channels,
  `--stop_frame` limits processed frames, and `--save_object_counts` writes the
  matching `acdc_objects_count` table. Existing `acdc_output` files are
  protected unless `--yes` is provided.
- `--prepare_zstack_segm_info`: write default Python-compatible
  `*segmInfo.csv` tables for z-stack positions under exactly one of
  `--position_dir` or `--experiment_dir`. Existing files are protected unless
  `--yes` is provided.
- `--compute_background_roi_data`: write Python-compatible
  `*_bkgrRoiData.npz` archives from an existing `*dataPrep_bkgrROIs.json` under
  exactly one of `--position_dir` or `--experiment_dir`. Repeat
  `--channel_name` to restrict channels; when omitted, all discovered channels
  are processed.
- `--inspect_frame`: inspect one Cell-ACDC position frame and print a JSON
  summary of segmentation labels. Required arguments are `--position_dir` and
  `--frame_i`; `--segm_endname` selects a non-default segmentation,
  `--selected_label` adds per-object area, centroid, bounding box, channel
  intensity, and cell-cycle metadata when available, and `--z_slice` selects a
  z-slice instead of the default max projection. Use `--output_path` to write
  the JSON to a file.
- `--export_frame_image`: export one rendered position frame to PNG or TIFF.
  Required arguments are `--position_dir`, `--frame_i`, and `--output_path`.
  Provide one `--channel_name` to choose the image channel; when omitted, the
  default phase-like channel is used. `--segm_endname` selects a segmentation,
  `--selected_label` highlights one label, and `--z_slice` selects a z-slice
  instead of the default max projection. The segmentation overlay is enabled by
  default when available; use `--no_overlay`, `--show_labels`, `--scale_bar`,
  and `--timestamp` to control rendered annotations.
- `--export_frame_sequence`: export a rendered position frame range as PNG
  images named `frame_0000.png`, `frame_0001.png`, and so on under
  `--output_path`. Required arguments are `--position_dir` and `--output_path`;
  `--start_frame` and `--end_frame` are inclusive and default to the full time
  range. The channel, segmentation, z-slice, overlay, label, scale bar, and
  timestamp options match `--export_frame_image`.
- `--apply_tracking_from_table`: apply tracking IDs from a CSV/XLSX table to a
  time-series segmentation mask. Required arguments are `--segmentation_path`,
  `--tracking_table_path`, and `--output_path`. Tracking columns default to
  `--frame_index_col frame_i` and `--track_ids_col track_id`; optional mapping
  inputs include `--mask_ids_col`, centroid columns, `--first_frame_one`, and
  `--delete_untracked_ids`. Optional `--source_acdc_output_path` and
  `--output_acdc_output_path` remap a matching `acdc_output` table.
- `--repeat_tracking`: repeat overlap tracking for a position segmentation and
  refresh its measurement table. Required argument is `--position_dir`;
  `--segm_endname` selects a non-default segmentation. By default tracking runs
  over the whole position; `--start_frame` repeats tracking from one frame to
  the end while preserving the previous frame as the anchor. `--ioa_threshold`
  defaults to `0.6`, `--overlap_denominator` accepts `area_prev` or `union`,
  and `--no_assign_unique_new_ids` disables unique ID assignment for new labels.
- `--apply_tracking_from_trackmate_xml`: apply tracking IDs from a TrackMate XML
  file to a position segmentation mask. Required arguments are `--position_dir`,
  `--segm_endname`, and `--xml_path`; `--output_path` optionally overrides the
  tracked segmentation output path. `--delete_untracked_ids` and the optional
  `--source_acdc_output_path`/`--output_acdc_output_path` table remapping flags
  are also supported.
- `--add_lineage_tree`: add lineage-tree columns to `acdc_output` CSV/XLSX
  tables. Use `--input_path` and `--output_path` for one explicit table, or
  pass exactly one of `--position_dir` and `--experiment_dir` to update matching
  position tables in place. `--table_endname` defaults to `acdc_output` for
  batch operation.
- `--build_lineage_state`: build or normalize lineage-tree columns in an
  `acdc_output` table. Required arguments are `--input_path` and
  `--output_path`.
- `--export_lineage_info`: export a JSON summary of cells with parents, orphan
  cells, and lost cells for one lineage frame. Required arguments are
  `--input_path`, `--output_path`, and `--frame_i`.
- `--propagate_lineage`: propagate lineage-tree values from one frame to future
  rows in an `acdc_output` table. Required arguments are `--input_path`,
  `--output_path`, and `--frame_i`. Repeat `--cell_id` to restrict propagation;
  when omitted, all cells in the source frame are propagated.
- `--update_lineage_frame`: apply lineage edits to one frame of an
  `acdc_output` table. Required arguments are `--input_path`, `--output_path`,
  `--frame_i`, and exactly one of `--edits_table_path` or `--edits_json_path`.
- `--generate_mother_bud_total`: generate G1, mother, bud, and total rows from
  an `acdc_output` CSV/XLSX table. Required arguments are `--input_path` and
  `--output_path`. Repeat `--column_operation COLUMN=OPERATION` for columns that
  should be combined; operations containing `sum` add mother and bud values.
  Repeat `--grouping_column COLUMN` when mother/bud matching needs extra keys.
  The entity label column defaults to `entity` and can be changed with
  `--entity_colname`; use `--no_copy_all_nonselected_columns` to keep only
  columns named by `--column_operation`.
- `--combine_metrics`: combine metrics from two or more CSV/XLSX tables using
  formulas. Repeat `--source_path PATH` for each table, pass `--output_path`,
  and repeat `--formula COLUMN=EXPRESSION` for each output metric. Expressions
  can use aliases such as `table1_signal` and `signal_table1`. The equations INI
  path defaults next to the output table and can be changed with
  `--equations_path`.
- `--compute_multi_channel`: compute combined metric tables for a Cell-ACDC
  position or experiment. Provide exactly one of `--position_dir` or
  `--experiment_dir`, repeat `--source_endname ENDNAME` for the source tables,
  and repeat `--formula COLUMN=EXPRESSION`. Outputs are written into each
  position `Images` folder using `--append_name`, which defaults to
  `combined_metrics`.
- `--concat_acdc_outputs`: concatenate `acdc_output`-style tables across
  Cell-ACDC positions and optionally across experiments. Repeat
  `--concat_experiment_dir PATH` for each experiment. `--table_endname` defaults
  to `acdc_output`, `--output_format` accepts `csv` or `xlsx`, and repeated
  `--selected_column COLUMN` flags keep a subset of columns. `--output_name`
  controls each experiment-level output filename; `--multi_experiment_dir`
  controls where multi-experiment outputs are written.
- `--combine_channels`: combine raw image and segmentation channels from a JSON
  recipe. Provide exactly one of `--position_dir` or `--experiment_dir`, pass
  `--recipe_path`, and use `--append_name` for the output suffix. Recipes use
  numeric step entries with `name`, `channel`, `binarize`, `min_val`, and
  `max_val`, plus top-level `formula`, `keep_input_data_type`, and
  `save_as_segm` fields.
- `--convert_file_format`: convert a single Cell-ACDC-compatible image/array
  file. Required arguments are `--input_path` and `--output_path`; formats are
  inferred from extensions. Inputs support `.npz`, `.npy`, `.tif`, `.tiff`, and
  `.h5`; outputs support `.npz`, `.npy`, `.tif`, and `.tiff`. Use
  `--cast_segm_uint32` to match the Python converter's segmentation cast.
- `--rename_files`: append text to one or more filenames. Repeat
  `--file_path PATH` for each file and pass `--rename_append_text TEXT`. The
  output filename is `stem_TEXT.ext`; existing target files are not overwritten.
- `--import_experiment`: import TIFF/NPZ/H5 sources into a Cell-ACDC experiment
  folder using the native Data Structure importer. Repeat `--import_source PATH`
  for files or folders and provide `--target_dir`. `--import_layout` accepts
  `single_file_multi_position`, `file_per_position`, or `file_per_channel` and
  is inferred when omitted. Optional controls include `--import_backend`
  (`auto`, `native`, `bioformats`), `--import_conflict_mode`
  (`create_new_positions`, `overwrite`, `add_files`), `--import_output_format`
  (`tiff`, `h5`), `--import_add_image_name`, and zero-based half-open frame
  cropping with `--import_time_start` and `--import_time_end`.
- `--images_to_positions`: convert a flat folder of image files into Cell-ACDC
  `Position_n/Images` folders. Required arguments are `--source_dir`,
  `--target_dir`, and `--images_append_text`; output filenames follow the Python
  utility pattern `sXX_STEM_TEXT.tif`. Invalid image files and directories are
  skipped.
- `--move_channel_tiffs_to_positions`: move separate channel TIFF files from a
  flat folder into `Position_n/Images` folders. Required arguments are
  `--source_dir` and one or more `--channel_name` values. Basenames are inferred
  from filename prefixes before the channel names, matching the Python utility;
  matching metadata CSV files are moved and their `basename` row is updated.

## Supported Workflow INI Subset

Supported sections:

- `[workflow]`
- `[paths_info]`
- `[paths_to_segment]` as a legacy alias for `[paths_info]`
- `[initialization]`
- `[metadata]`
- `[init_segmentation_model_params]`
- `[segmentation_model_params]`
- `[init_tracker_params]`
- `[tracker_params]`
- `[standard_postprocess_features]`
- `[custom_postprocess_features]`
- `[preprocess.stepN]`
- `[postprocess_features.<category>]`
- `[measurements]`
- `[rust_cli]`

Supported workflow types:

- `segmentation and/or tracking`
- `measurements`

Model path for segmentation/tracking workflows:

```ini
[rust_cli]
model_path = /path/to/model.onnx
```

`[rust_cli].model_path` is preferred. For Python-generated workflows that
already contain an explicit `init_segmentation_model_params.model_path` or
`segmentation_model_params.model_path`, Rust uses that path and `[rust_cli]` can
be omitted.

Optional `rust_cli` keys:

- `fluo_channel`
- `cpu`
- `overwrite`

Known Python-generated segmentation metadata, initialization, preprocessing,
and postprocessing sections are accepted for workflow compatibility. The Rust
runner currently uses the segmentation channel, segmentation suffix, stop-frame
counts, overlap-tracking threshold, data-prep crop ROI coordinates, data-prep
free-hand ROI masks, Python `second_channel_name`, and `rust_cli` options.
When tracking is enabled, Rust supports Python `tracker_name` values that map to
Cell-ACDC overlap/IoA tracking; `IoA_thresh`, `IoA_threshold`, and
`overlap_threshold` configure the same overlap threshold. The
`assign_unique_new_IDs` tracker parameter is also honored, as is
`denom_overlap_matrix` with Python's `area_prev` and `union` modes. Other Python
trackers are rejected.
Rust Cellpose-style parameters `tile`, `batch_size`, `cellprob_threshold`,
`niter`, and `min_size` are used when present and otherwise fall back to Rust
defaults. Python preprocessing recipe sections are parsed in order; the Rust
runner currently applies `method = Gaussian filter` with scalar or vector `sigma` and
`method = Remove hot pixels`, plus `method = Spot detector filter` with
`spots_zyx_radii_pxl`, `method = Ridge filter` with `sigmas`, `method =
Enhance speckles` with `radius`, and `method = Correct illumination` with `block_size`,
`approximate_object_diameter`, and `apply_gaussian_filter`. `method = FUCCI
pre-processing` is expanded into the supported illumination and speckle filters
when `do_basicpy_background_correction = false`; the BaSiC background correction
stage remains Python-only. Other Python-only
preprocessing, postprocessing, and model-specific values are parsed as known
no-op metadata. `rust_cli.fluo_channel` overrides
`initialization.second_channel_name` when both are present. If Python-style
`use_ROI = false` or
`use_freehand_ROI = false` initialization keys are present, the corresponding
saved ROI files are ignored; otherwise `*_dataPrepROIs_coords.csv` clamps output
masks to the active crop ROI and objects touching pixels outside
`*_dataPrepFreeRoi.npz` are removed before tracking, matching Python's default
CLI behavior. `do_save = false` runs segmentation without writing segmentation,
measurement, or hyperparameter files. When `do_postprocess = true`, the
standard postprocess `min_area` and `max_elongation` filters are applied before
tracking, along with `min_solidity`. The `min_obj_no_zslices` option is parsed
and preserved in the postprocess config; it is currently a no-op in Rust's 2D
segmentation runner because Python applies it only to 3D z-stack labels. Other
standard/custom postprocess options are accepted as deferred metadata.

Measurement workflows do not require `[initialization]`,
`[segmentation_model_params]`, or `[rust_cli]`. Supported `[measurements]`
keys include:

- `end_filename_segm`
- `channels`
- `channel_names_to_skip`
- `channel_names_to_process`
- `metrics_to_save_<channel>`
- `metrics_to_skip_<channel>`
- `size_metrics_to_save`
- `regionprops_to_save`
- `calc_for_each_zslice_channels`
- `calc_for_each_zslice_size`
- `save_object_counts_table`

Empty Python-generated `channel_indipendent_custom_metrics_to_save` and
`mixed_combine_metrics_to_skip` entries are accepted as no-op metadata. Non-empty
custom metric requests are still unsupported.

Python-style segmentation end filenames are accepted by segmentation and
measurement workflows with or without the `segm` prefix and optional `.npz`
extension. For example, `rust`, `segm_rust`, and `segm_rust.npz` all resolve to
`*_segm_rust.npz`.
For optional segmentation strings, the Python literal `None` is treated as
missing.

Measurement metric filters may use either metric suffixes such as `mean` or
full Python column names such as `phase_mean`. Z-stack workflow options can
emit selected z-slice columns such as `phase_mean_zSlice` and per-z columns
such as `phase_mean_zSlice0`, plus per-z size columns such as
`cell_area_pxl_zslice0`.
When a Python-style `*_manualBackground*.npz` mask exists next to the selected
segmentation, manual background metrics such as `phase_mean_manualBkgr` and
`phase_amount_manualBkgr` are emitted and can be selected with
`metrics_to_save_<channel>`.

`channel_names_to_skip` is interpreted relative to `channels` or
`channel_names_to_process`; a skip list without either base list is rejected.

The workflow parser is intentionally strict:

- Unsupported sections fail immediately.
- Unsupported keys inside supported sections fail immediately.
- Python-only workflow features outside the documented subset are not emulated.

## Explicitly Unsupported In This Milestone

- model download flags
- Python built-in model-name resolution without an explicit model path
- Python trackers other than Cell-ACDC overlap/IoA tracking
- Python custom measurement functions and mixed-channel combine metrics when
  requested
- public CLI access to remaining Rust-only utilities and lineage helpers

Those remain available through the Rust library and, where already wired, the GUI.
