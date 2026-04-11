use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::layout::{resolve_workflow_targets, PositionSpec};
use crate::measure::{measure_position, MeasurementRunConfig, MeasurementRunResult};
use crate::runner::{
    run_position, OverwritePolicy, RunResult, SegmentationParams, SegmentationRunConfig,
};
use crate::tracking::TrackingConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowKind {
    SegmentationAndTracking,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowTarget {
    pub input_path: PathBuf,
    pub position: PositionSpec,
    pub stop_frame: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationWorkflowConfig {
    pub targets: Vec<WorkflowTarget>,
    pub phase_channel: String,
    pub fluo_channel: String,
    pub model_path: PathBuf,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub cpu: bool,
    pub params: SegmentationParams,
    pub tracking: Option<TrackingConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementWorkflowConfig {
    pub targets: Vec<WorkflowTarget>,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowFile {
    pub path: PathBuf,
    pub kind: WorkflowKind,
    pub segmentation: SegmentationWorkflowConfig,
    pub measurement: Option<MeasurementWorkflowConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkflowRunOptions {
    pub debug: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkflowRunReport {
    pub segmentation_results: Vec<RunResult>,
    pub measurement_results: Vec<MeasurementRunResult>,
}

#[derive(Debug, Clone)]
struct IniFile {
    sections: Vec<IniSection>,
}

#[derive(Debug, Clone)]
struct IniSection {
    original_name: String,
    normalized_name: String,
    entries: Vec<IniEntry>,
}

#[derive(Debug, Clone)]
struct IniEntry {
    original_key: String,
    normalized_key: String,
    value: String,
}

pub fn parse_workflow_file(path: impl AsRef<Path>) -> Result<WorkflowFile> {
    let path = path.as_ref().to_path_buf();
    let text = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read workflow file {}", path.display()))?;
    let ini = parse_ini(&text, &path)?;
    validate_supported_sections(&ini)?;

    let workflow = required_section(&ini, "workflow")?;
    let workflow_type = required_key(workflow, "type")?;
    let kind = match workflow_type.value.trim().to_ascii_lowercase().as_str() {
        "segmentation and/or tracking" => WorkflowKind::SegmentationAndTracking,
        other => bail!(
            "Unsupported workflow.type {:?} in {}. Only \"segmentation and/or tracking\" is supported.",
            other,
            path.display()
        ),
    };

    let paths_info = required_section(&ini, "paths_info")?;
    let initialization = required_section(&ini, "initialization")?;
    let segm_params = required_section(&ini, "segmentation_model_params")?;
    let rust_cli = required_section(&ini, "rust_cli")?;
    let tracker_params = find_section(&ini, "tracker_params");
    let measurements = find_section(&ini, "measurements");
    let workflow_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let phase_channel = required_key(initialization, "user_ch_name")?
        .value
        .trim()
        .to_string();
    if phase_channel.is_empty() {
        bail!(
            "initialization.user_ch_name cannot be empty in {}",
            path.display()
        );
    }

    let fluo_channel = optional_key(rust_cli, "fluo_channel")
        .map(|entry| entry.value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| phase_channel.clone());
    let segm_endname = optional_key(initialization, "segm_endname")
        .map(|entry| entry.value.trim().to_string())
        .filter(|value| !value.is_empty());
    let do_tracking = parse_bool(required_key(initialization, "do_tracking")?, "do_tracking")?;

    let model_path = resolve_path_from_workflow_dir(
        workflow_dir,
        Path::new(required_key(rust_cli, "model_path")?.value.trim()),
    );
    let overwrite_policy = overwrite_policy(
        parse_optional_bool(optional_key(rust_cli, "overwrite"))?.unwrap_or(false),
    );
    let cpu = parse_optional_bool(optional_key(rust_cli, "cpu"))?.unwrap_or(false);

    let params = SegmentationParams {
        tile: parse_usize(required_key(segm_params, "tile")?, "tile")?,
        batch_size: parse_usize(required_key(segm_params, "batch_size")?, "batch_size")?,
        cellprob_threshold: parse_f32(
            required_key(segm_params, "cellprob_threshold")?,
            "cellprob_threshold",
        )?,
        niter: parse_usize(required_key(segm_params, "niter")?, "niter")?,
        min_size: parse_usize(required_key(segm_params, "min_size")?, "min_size")?,
    };

    let tracking = if do_tracking {
        let ioa_threshold =
            match tracker_params.and_then(|section| optional_key(section, "ioa_thresh")) {
                Some(entry) => parse_f32(entry, "IoA_thresh")?,
                None => 0.4,
            };
        Some(TrackingConfig { ioa_threshold })
    } else {
        None
    };

    let input_paths = parse_path_list(required_key(paths_info, "paths")?, workflow_dir)?;
    if input_paths.is_empty() {
        bail!(
            "paths_info.paths must contain at least one path in {}",
            path.display()
        );
    }
    let stop_frames = parse_stop_frame_numbers(required_key(paths_info, "stop_frame_numbers")?)?;

    let mut expanded = Vec::<(PathBuf, PositionSpec)>::new();
    for input_path in &input_paths {
        for position in resolve_workflow_targets(input_path, &phase_channel, &fluo_channel)? {
            expanded.push((input_path.clone(), position));
        }
    }
    if expanded.is_empty() {
        bail!(
            "Workflow {} did not resolve any Cell-ACDC targets",
            path.display()
        );
    }

    let stop_frames = broadcast_stop_frames(&stop_frames, expanded.len(), &path)?;
    let targets = expanded
        .into_iter()
        .zip(stop_frames)
        .map(|((input_path, position), stop_frame)| WorkflowTarget {
            input_path,
            position,
            stop_frame,
        })
        .collect::<Vec<_>>();

    let segmentation = SegmentationWorkflowConfig {
        targets: targets.clone(),
        phase_channel,
        fluo_channel,
        model_path,
        segm_endname,
        overwrite_policy,
        cpu,
        params,
        tracking,
    };

    let measurement = measurements
        .map(|section| -> Result<MeasurementWorkflowConfig> {
            let segm_endname = optional_key(section, "end_filename_segm")
                .map(|entry| entry.value.trim().to_string())
                .filter(|value| !value.is_empty());
            if segm_endname.is_none() {
                bail!(
                    "measurements.end_filename_segm cannot be empty in {}",
                    path.display()
                );
            }
            Ok(MeasurementWorkflowConfig {
                targets: targets.clone(),
                segm_endname,
                overwrite_policy,
            })
        })
        .transpose()?;

    Ok(WorkflowFile {
        path,
        kind,
        segmentation,
        measurement,
    })
}

pub fn run_workflow_file(
    path: impl AsRef<Path>,
    opts: WorkflowRunOptions,
) -> Result<WorkflowRunReport> {
    let workflow = parse_workflow_file(path)?;
    let mut report = WorkflowRunReport::default();

    if opts.debug {
        eprintln!(
            "Running workflow {} for {} target(s)",
            workflow.path.display(),
            workflow.segmentation.targets.len()
        );
    }

    match workflow.kind {
        WorkflowKind::SegmentationAndTracking => {
            for target in &workflow.segmentation.targets {
                if opts.debug {
                    eprintln!(
                        "Segmentation target {} -> {}",
                        target.input_path.display(),
                        target.position.position_dir.display()
                    );
                }
                report
                    .segmentation_results
                    .push(run_position(SegmentationRunConfig {
                        position: target.position.clone(),
                        model_path: workflow.segmentation.model_path.clone(),
                        segm_endname: workflow.segmentation.segm_endname.clone(),
                        overwrite_policy: workflow.segmentation.overwrite_policy,
                        cpu: workflow.segmentation.cpu,
                        params: workflow.segmentation.params.clone(),
                        tracking: workflow.segmentation.tracking.clone(),
                        stop_frame: target.stop_frame,
                    })?);
            }
        }
    }

    if let Some(measurement) = &workflow.measurement {
        for target in &measurement.targets {
            if opts.debug {
                eprintln!(
                    "Measurement target {} -> {}",
                    target.input_path.display(),
                    target.position.position_dir.display()
                );
            }
            report
                .measurement_results
                .push(measure_position(MeasurementRunConfig {
                    position_path: target.position.position_dir.clone(),
                    segm_endname: measurement.segm_endname.clone(),
                    overwrite_policy: measurement.overwrite_policy,
                    stop_frame: target.stop_frame,
                })?);
        }
    }

    Ok(report)
}

fn parse_ini(text: &str, path: &Path) -> Result<IniFile> {
    let mut sections = Vec::<IniSection>::new();
    let mut current_section = None::<usize>;
    let mut current_entry = None::<usize>;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let name = trimmed[1..trimmed.len() - 1].trim();
            if name.is_empty() {
                bail!("Empty INI section header at {}:{}", path.display(), line_no);
            }
            let normalized_name = normalize_name(name);
            if sections
                .iter()
                .any(|section| section.normalized_name == normalized_name)
            {
                bail!(
                    "Duplicate INI section [{}] at {}:{}",
                    name,
                    path.display(),
                    line_no
                );
            }
            sections.push(IniSection {
                original_name: name.to_string(),
                normalized_name,
                entries: Vec::new(),
            });
            current_section = Some(sections.len() - 1);
            current_entry = None;
            continue;
        }

        if raw_line.starts_with(char::is_whitespace) {
            let Some(section_idx) = current_section else {
                bail!(
                    "Found multi-line continuation outside any section at {}:{}",
                    path.display(),
                    line_no
                );
            };
            let Some(entry_idx) = current_entry else {
                bail!(
                    "Found multi-line continuation without a key at {}:{}",
                    path.display(),
                    line_no
                );
            };
            let continuation = trimmed;
            let value = &mut sections[section_idx].entries[entry_idx].value;
            if !value.is_empty() && !continuation.is_empty() {
                value.push('\n');
            }
            value.push_str(continuation);
            continue;
        }

        let Some(section_idx) = current_section else {
            bail!(
                "Found key/value outside any section at {}:{}",
                path.display(),
                line_no
            );
        };
        let Some((raw_key, raw_value)) = raw_line.split_once('=') else {
            bail!(
                "Invalid INI assignment at {}:{}: expected key = value",
                path.display(),
                line_no
            );
        };
        let key = raw_key.trim();
        if key.is_empty() {
            bail!("Empty INI key at {}:{}", path.display(), line_no);
        }
        let normalized_key = normalize_name(key);
        if sections[section_idx]
            .entries
            .iter()
            .any(|entry| entry.normalized_key == normalized_key)
        {
            bail!(
                "Duplicate INI key {} in section [{}] at {}:{}",
                key,
                sections[section_idx].original_name,
                path.display(),
                line_no
            );
        }
        sections[section_idx].entries.push(IniEntry {
            original_key: key.to_string(),
            normalized_key,
            value: raw_value.trim().to_string(),
        });
        current_entry = Some(sections[section_idx].entries.len() - 1);
    }

    Ok(IniFile { sections })
}

fn validate_supported_sections(ini: &IniFile) -> Result<()> {
    let allowed = BTreeMap::from([
        ("workflow", BTreeSet::from(["type"])),
        (
            "paths_info",
            BTreeSet::from(["paths", "stop_frame_numbers"]),
        ),
        (
            "initialization",
            BTreeSet::from(["user_ch_name", "segm_endname", "do_tracking"]),
        ),
        (
            "segmentation_model_params",
            BTreeSet::from([
                "tile",
                "batch_size",
                "cellprob_threshold",
                "niter",
                "min_size",
            ]),
        ),
        ("tracker_params", BTreeSet::from(["ioa_thresh"])),
        ("measurements", BTreeSet::from(["end_filename_segm"])),
        (
            "rust_cli",
            BTreeSet::from(["model_path", "fluo_channel", "cpu", "overwrite"]),
        ),
    ]);
    let mut unsupported_sections = Vec::new();
    let mut unsupported_keys = Vec::new();

    for section in &ini.sections {
        let Some(allowed_keys) = allowed.get(section.normalized_name.as_str()) else {
            unsupported_sections.push(section.original_name.clone());
            continue;
        };
        for entry in &section.entries {
            if !allowed_keys.contains(entry.normalized_key.as_str()) {
                unsupported_keys.push(format!(
                    "[{}].{}",
                    section.original_name, entry.original_key
                ));
            }
        }
    }

    if unsupported_sections.is_empty() && unsupported_keys.is_empty() {
        return Ok(());
    }

    let mut message = String::new();
    if !unsupported_sections.is_empty() {
        unsupported_sections.sort();
        message.push_str("Unsupported workflow sections: ");
        message.push_str(&unsupported_sections.join(", "));
    }
    if !unsupported_keys.is_empty() {
        unsupported_keys.sort();
        if !message.is_empty() {
            message.push('\n');
        }
        message.push_str("Unsupported workflow keys: ");
        message.push_str(&unsupported_keys.join(", "));
    }
    bail!(message)
}

fn required_section<'a>(ini: &'a IniFile, name: &str) -> Result<&'a IniSection> {
    find_section(ini, name).ok_or_else(|| anyhow::anyhow!("Missing required INI section [{name}]"))
}

fn find_section<'a>(ini: &'a IniFile, name: &str) -> Option<&'a IniSection> {
    let normalized = normalize_name(name);
    ini.sections
        .iter()
        .find(|section| section.normalized_name == normalized)
}

fn required_key<'a>(section: &'a IniSection, key: &str) -> Result<&'a IniEntry> {
    optional_key(section, key).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing required INI key [{}].{}",
            section.original_name,
            key
        )
    })
}

fn optional_key<'a>(section: &'a IniSection, key: &str) -> Option<&'a IniEntry> {
    let normalized = normalize_name(key);
    section
        .entries
        .iter()
        .find(|entry| entry.normalized_key == normalized)
}

fn normalize_name(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn parse_bool(entry: &IniEntry, label: &str) -> Result<bool> {
    parse_bool_value(&entry.value).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid boolean for {} (key {}): {:?}",
            label,
            entry.original_key,
            entry.value
        )
    })
}

fn parse_optional_bool(entry: Option<&IniEntry>) -> Result<Option<bool>> {
    entry
        .map(|entry| parse_bool(entry, &entry.original_key))
        .transpose()
}

fn parse_bool_value(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn parse_usize(entry: &IniEntry, label: &str) -> Result<usize> {
    entry.value.trim().parse::<usize>().with_context(|| {
        format!(
            "Failed to parse [{}].{} as usize for {}",
            entry.original_key, entry.original_key, label
        )
    })
}

fn parse_f32(entry: &IniEntry, label: &str) -> Result<f32> {
    entry.value.trim().parse::<f32>().with_context(|| {
        format!(
            "Failed to parse [{}].{} as number for {}",
            entry.original_key, entry.original_key, label
        )
    })
}

fn parse_path_list(entry: &IniEntry, base_dir: &Path) -> Result<Vec<PathBuf>> {
    split_multiline_value(&entry.value)
        .into_iter()
        .map(|value| {
            let path = Path::new(value);
            let resolved = resolve_path_from_workflow_dir(base_dir, path);
            if resolved.exists() {
                Ok(resolved)
            } else {
                bail!(
                    "Workflow path {:?} from key {} does not exist",
                    value,
                    entry.original_key
                )
            }
        })
        .collect()
}

fn parse_stop_frame_numbers(entry: &IniEntry) -> Result<Vec<Option<usize>>> {
    let values = split_multiline_value(&entry.value);
    if values.is_empty() {
        bail!("paths_info.stop_frame_numbers must not be empty");
    }
    values
        .into_iter()
        .map(|value| {
            let parsed = value.parse::<usize>().with_context(|| {
                format!(
                    "Failed to parse stop_frame_numbers value {:?} as usize",
                    value
                )
            })?;
            Ok(Some(parsed))
        })
        .collect()
}

fn split_multiline_value(value: &str) -> Vec<&str> {
    value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

fn resolve_path_from_workflow_dir(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let joined = base_dir.join(path);
        if joined.exists() {
            joined
        } else {
            path.to_path_buf()
        }
    }
}

fn broadcast_stop_frames(
    stop_frames: &[Option<usize>],
    target_count: usize,
    path: &Path,
) -> Result<Vec<Option<usize>>> {
    match stop_frames.len() {
        1 => Ok(vec![stop_frames[0]; target_count]),
        len if len == target_count => Ok(stop_frames.to_vec()),
        len => bail!(
            "paths_info.stop_frame_numbers in {} resolved to {} value(s) but workflow expansion produced {} target(s)",
            path.display(),
            len,
            target_count
        ),
    }
}

fn overwrite_policy(overwrite: bool) -> OverwritePolicy {
    if overwrite {
        OverwritePolicy::Overwrite
    } else {
        OverwritePolicy::Refuse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn parses_segmentation_only_workflow() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[workflow]\n\
type = segmentation and/or tracking\n\n\
[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 2\n\n\
[initialization]\n\
user_ch_name = phase\n\
segm_endname = rust\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\
tile = 256\n\
batch_size = 1\n\
cellprob_threshold = 0.0\n\
niter = 200\n\
min_size = 15\n\n\
[tracker_params]\n\
IoA_thresh = 0.6\n\n\
[rust_cli]\n\
model_path = models/demo.onnx\n\
overwrite = true\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        assert_eq!(workflow.kind, WorkflowKind::SegmentationAndTracking);
        assert_eq!(workflow.segmentation.targets.len(), 1);
        assert_eq!(workflow.segmentation.targets[0].stop_frame, Some(2));
        assert_eq!(workflow.segmentation.phase_channel, "phase");
        assert_eq!(workflow.segmentation.fluo_channel, "phase");
        assert_eq!(
            workflow.segmentation.tracking,
            Some(TrackingConfig { ioa_threshold: 0.6 })
        );
        assert_eq!(workflow.segmentation.segm_endname.as_deref(), Some("rust"));
        assert_eq!(workflow.measurement, None);
        Ok(())
    }

    #[test]
    fn parses_measurement_workflow() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[workflow]\n\
type = segmentation and/or tracking\n\n\
[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[initialization]\n\
user_ch_name = phase\n\
segm_endname = segm_rust\n\
do_tracking = false\n\n\
[segmentation_model_params]\n\
tile = 256\n\
batch_size = 1\n\
cellprob_threshold = 0.0\n\
niter = 200\n\
min_size = 15\n\n\
[measurements]\n\
end_filename_segm = segm_rust\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let measurement = workflow.measurement.expect("measurement workflow");
        assert_eq!(measurement.targets.len(), 1);
        assert_eq!(measurement.segm_endname.as_deref(), Some("segm_rust"));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_keys() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[workflow]\n\
type = segmentation and/or tracking\n\n\
[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[initialization]\n\
user_ch_name = phase\n\
segm_endname = rust\n\
do_tracking = false\n\
model_name = cyto3\n\n\
[segmentation_model_params]\n\
tile = 256\n\
batch_size = 1\n\
cellprob_threshold = 0.0\n\
niter = 200\n\
min_size = 15\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).unwrap_err();
        assert!(err
            .to_string()
            .contains("Unsupported workflow keys: [initialization].model_name"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_stop_frame_counts() -> Result<()> {
        let temp = tempdir()?;
        let pos1 = write_test_position(temp.path(), "Position_1")?;
        let pos2 = write_test_position(temp.path(), "Position_2")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[workflow]\n\
type = segmentation and/or tracking\n\n\
[paths_info]\n\
paths =\n  {}\n  {}\n\
stop_frame_numbers = 1\n  2\n  3\n\n\
[initialization]\n\
user_ch_name = phase\n\
segm_endname = rust\n\
do_tracking = false\n\n\
[segmentation_model_params]\n\
tile = 256\n\
batch_size = 1\n\
cellprob_threshold = 0.0\n\
niter = 200\n\
min_size = 15\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                pos1.display(),
                pos2.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).unwrap_err();
        assert!(err.to_string().contains("stop_frame_numbers"));
        Ok(())
    }

    fn write_test_position(root: &Path, name: &str) -> Result<PathBuf> {
        let position = root.join(name);
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack(&images.join("demo_phase.tif"), &[1, 2])?;
        let mut metadata = fs::File::create(images.join("demo_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,demo_")?;
        writeln!(metadata, "SizeT,2")?;
        Ok(position)
    }

    fn write_test_stack(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = fs::File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frame_values {
            let data = vec![*value; 4];
            encoder.write_image::<colortype::Gray16>(2, 2, &data)?;
        }
        Ok(())
    }
}
