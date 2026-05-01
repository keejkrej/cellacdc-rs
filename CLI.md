# `cellacdc-rs` CLI

The public CLI now follows the same top-level shape as Python `acdc`.

## Usage

```bash
cellacdc-rs
cellacdc-rs -p workflow.ini
cellacdc-rs -v
cellacdc-rs -info
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
- public CLI access to Rust-only utilities and lineage helpers

Those remain available through the Rust library and, where already wired, the GUI.
