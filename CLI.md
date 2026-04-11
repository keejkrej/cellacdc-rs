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

## Supported Workflow INI Subset

Supported sections:

- `[workflow]`
- `[paths_info]`
- `[initialization]`
- `[segmentation_model_params]`
- `[tracker_params]`
- `[measurements]`
- `[rust_cli]`

Required Rust-only section:

```ini
[rust_cli]
model_path = /path/to/model.onnx
```

Optional `rust_cli` keys:

- `fluo_channel`
- `cpu`
- `overwrite`

The workflow parser is intentionally strict:

- Unsupported sections fail immediately.
- Unsupported keys inside supported sections fail immediately.
- Python-only workflow features outside the documented subset are not emulated.

## Explicitly Unsupported In This Milestone

- `--reset`
- `-y/--yes`
- `--install_details`
- model download flags
- Python model-name resolution without an explicit `rust_cli.model_path`
- Python measurement-option parity beyond `measurements.end_filename_segm`
- public CLI access to Rust-only utilities and lineage helpers

Those remain available through the Rust library and, where already wired, the GUI.
