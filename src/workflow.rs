use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::layout::{
    discover_measurement_experiment, resolve_measurement_position, resolve_workflow_targets,
    PositionSpec,
};
use crate::measure::{
    measure_position, normalize_metric_key, MeasurementMetricOptions, MeasurementRunConfig,
    MeasurementRunResult,
};
use crate::runner::{
    run_position, OverwritePolicy, PostprocessConfig, PreprocessStep, RunResult,
    SegmentationParams, SegmentationRunConfig,
};
use crate::tracking::{OverlapDenominator, TrackingConfig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowKind {
    SegmentationAndTracking,
    Measurements,
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
    pub preprocess_steps: Vec<PreprocessStep>,
    pub tracking: Option<TrackingConfig>,
    pub postprocess: Option<PostprocessConfig>,
    pub save_outputs: bool,
    pub use_data_prep_roi: bool,
    pub use_data_prep_free_roi: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementWorkflowTarget {
    pub input_path: PathBuf,
    pub position_path: PathBuf,
    pub stop_frame: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementWorkflowConfig {
    pub targets: Vec<MeasurementWorkflowTarget>,
    pub segm_endname: Option<String>,
    pub overwrite_policy: OverwritePolicy,
    pub channel_names: Option<Vec<String>>,
    pub metric_options: Option<MeasurementMetricOptions>,
    pub save_object_counts_table: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowFile {
    pub path: PathBuf,
    pub kind: WorkflowKind,
    pub segmentation: Option<SegmentationWorkflowConfig>,
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
        "measurements" => WorkflowKind::Measurements,
        other => bail!(
            "Unsupported workflow.type {:?} in {}. Only \"segmentation and/or tracking\" and \"measurements\" are supported.",
            other,
            path.display()
        ),
    };

    let paths_info = workflow_paths_section(&ini)?;
    let measurements = find_section(&ini, "measurements");
    let workflow_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let input_paths = parse_path_list(required_key(paths_info, "paths")?, workflow_dir)?;
    if input_paths.is_empty() {
        bail!(
            "paths_info.paths must contain at least one path in {}",
            path.display()
        );
    }
    let stop_frames = parse_stop_frame_numbers(required_key(paths_info, "stop_frame_numbers")?)?;

    let (segmentation, measurement) = match kind {
        WorkflowKind::SegmentationAndTracking => {
            let initialization = required_section(&ini, "initialization")?;
            let segm_params = required_section(&ini, "segmentation_model_params")?;
            let rust_cli = find_section(&ini, "rust_cli");
            let init_model_params = find_section(&ini, "init_segmentation_model_params");
            let tracker_params = find_section(&ini, "tracker_params");
            let standard_postprocess = find_section(&ini, "standard_postprocess_features");

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

            let fluo_channel = rust_cli
                .and_then(|section| optional_python_string(section, "fluo_channel"))
                .or_else(|| optional_python_string(initialization, "second_channel_name"))
                .unwrap_or_else(|| phase_channel.clone());
            let segm_endname = optional_python_string(initialization, "segm_endname");
            let do_tracking =
                parse_bool(required_key(initialization, "do_tracking")?, "do_tracking")?;
            let do_postprocess =
                parse_optional_bool(optional_key(initialization, "do_postprocess"))?
                    .unwrap_or(false);

            let model_path = resolve_segmentation_model_path(
                workflow_dir,
                rust_cli,
                initialization,
                init_model_params,
                segm_params,
                &path,
            )?;
            let overwrite_policy = overwrite_policy(
                parse_optional_bool(
                    rust_cli.and_then(|section| optional_key(section, "overwrite")),
                )?
                .unwrap_or(false),
            );
            let cpu =
                parse_optional_bool(rust_cli.and_then(|section| optional_key(section, "cpu")))?
                    .unwrap_or(false);
            let save_outputs =
                parse_optional_bool(optional_key(initialization, "do_save"))?.unwrap_or(true);
            let use_data_prep_roi =
                parse_optional_bool(optional_key(initialization, "use_ROI"))?.unwrap_or(true);
            let use_data_prep_free_roi =
                parse_optional_bool(optional_key(initialization, "use_freehand_ROI"))?
                    .unwrap_or(true);

            let params = parse_segmentation_params(segm_params)?;
            let preprocess_steps = parse_preprocess_steps(&ini)?;
            let postprocess = if do_postprocess {
                standard_postprocess
                    .map(parse_standard_postprocess_config)
                    .transpose()?
                    .or_else(|| Some(PostprocessConfig::default()))
            } else {
                None
            };

            let tracking = if do_tracking {
                validate_supported_tracker(initialization, &path)?;
                Some(parse_tracking_config(tracker_params)?)
            } else {
                None
            };

            let mut expanded = Vec::<(PathBuf, PositionSpec)>::new();
            for input_path in &input_paths {
                for position in resolve_workflow_targets(input_path, &phase_channel, &fluo_channel)?
                {
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
                preprocess_steps,
                tracking,
                postprocess,
                save_outputs,
                use_data_prep_roi,
                use_data_prep_free_roi,
            };

            let measurement = measurements
                .map(|section| -> Result<MeasurementWorkflowConfig> {
                    parse_measurement_workflow_config(
                        section,
                        targets
                            .iter()
                            .map(|target| MeasurementWorkflowTarget {
                                input_path: target.input_path.clone(),
                                position_path: target.position.position_dir.clone(),
                                stop_frame: target.stop_frame,
                            })
                            .collect(),
                        overwrite_policy,
                        &path,
                    )
                })
                .transpose()?;
            (Some(segmentation), measurement)
        }
        WorkflowKind::Measurements => {
            let measurements = measurements.ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing required INI section [measurements] for measurements workflow"
                )
            })?;
            let targets = resolve_measurement_workflow_targets(&input_paths, &stop_frames, &path)?;
            let measurement = parse_measurement_workflow_config(
                measurements,
                targets,
                OverwritePolicy::Overwrite,
                &path,
            )?;
            (None, Some(measurement))
        }
    };

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
        let target_count = workflow
            .segmentation
            .as_ref()
            .map(|config| config.targets.len())
            .or_else(|| {
                workflow
                    .measurement
                    .as_ref()
                    .map(|config| config.targets.len())
            })
            .unwrap_or(0);
        eprintln!(
            "Running workflow {} for {} target(s)",
            workflow.path.display(),
            target_count
        );
    }

    match workflow.kind {
        WorkflowKind::SegmentationAndTracking => {
            let segmentation = workflow
                .segmentation
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Missing segmentation workflow configuration"))?;
            for target in &segmentation.targets {
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
                        model_path: segmentation.model_path.clone(),
                        segm_endname: segmentation.segm_endname.clone(),
                        overwrite_policy: segmentation.overwrite_policy,
                        cpu: segmentation.cpu,
                        params: segmentation.params.clone(),
                        preprocess_steps: segmentation.preprocess_steps.clone(),
                        tracking: segmentation.tracking.clone(),
                        postprocess: segmentation.postprocess.clone(),
                        stop_frame: target.stop_frame,
                        save_outputs: segmentation.save_outputs,
                        use_data_prep_roi: segmentation.use_data_prep_roi,
                        use_data_prep_free_roi: segmentation.use_data_prep_free_roi,
                    })?);
            }
        }
        WorkflowKind::Measurements => {}
    }

    if let Some(measurement) = &workflow.measurement {
        for target in &measurement.targets {
            if opts.debug {
                eprintln!(
                    "Measurement target {} -> {}",
                    target.input_path.display(),
                    target.position_path.display()
                );
            }
            report
                .measurement_results
                .push(measure_position(MeasurementRunConfig {
                    position_path: target.position_path.clone(),
                    segm_endname: measurement.segm_endname.clone(),
                    overwrite_policy: measurement.overwrite_policy,
                    stop_frame: target.stop_frame,
                    channel_names: measurement.channel_names.clone(),
                    metric_options: measurement.metric_options.clone(),
                    save_object_counts_table: measurement.save_object_counts_table,
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
            "paths_to_segment",
            BTreeSet::from(["paths", "stop_frame_numbers"]),
        ),
        (
            "initialization",
            BTreeSet::from([
                "user_ch_name",
                "segm_endname",
                "model_name",
                "tracker_name",
                "do_tracking",
                "do_postprocess",
                "do_save",
                "image_channel_tracker",
                "issegm3d",
                "use_roi",
                "use_freehand_roi",
                "second_channel_name",
                "use3ddatafor2dsegm",
                "reduce_memory_usage",
            ]),
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
        ("metadata", BTreeSet::from(["sizet", "sizez"])),
        ("init_segmentation_model_params", BTreeSet::new()),
        ("init_tracker_params", BTreeSet::new()),
        ("tracker_params", BTreeSet::new()),
        ("standard_postprocess_features", BTreeSet::new()),
        ("custom_postprocess_features", BTreeSet::new()),
        (
            "measurements",
            BTreeSet::from([
                "channels",
                "end_filename_segm",
                "channel_names_to_skip",
                "channel_names_to_process",
                "calc_for_each_zslice_channels",
                "calc_for_each_zslice_size",
                "size_metrics_to_save",
                "regionprops_to_save",
                "save_object_counts_table",
            ]),
        ),
        (
            "rust_cli",
            BTreeSet::from(["model_path", "fluo_channel", "cpu", "overwrite"]),
        ),
    ]);
    let mut unsupported_sections = Vec::new();
    let mut unsupported_keys = Vec::new();

    for section in &ini.sections {
        let Some(allowed_keys) = allowed.get(section.normalized_name.as_str()) else {
            if !is_supported_dynamic_section(section) {
                unsupported_sections.push(section.original_name.clone());
            }
            continue;
        };
        if allows_arbitrary_workflow_keys(section) {
            continue;
        }
        for entry in &section.entries {
            if !allowed_keys.contains(entry.normalized_key.as_str())
                && !is_supported_measurement_dynamic_key(section, entry)
            {
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

fn is_supported_dynamic_section(section: &IniSection) -> bool {
    section.normalized_name.starts_with("preprocess.step")
        || section.normalized_name.starts_with("postprocess_features.")
}

fn allows_arbitrary_workflow_keys(section: &IniSection) -> bool {
    matches!(
        section.normalized_name.as_str(),
        "init_segmentation_model_params"
            | "init_tracker_params"
            | "tracker_params"
            | "segmentation_model_params"
            | "standard_postprocess_features"
            | "custom_postprocess_features"
    ) || is_supported_dynamic_section(section)
}

fn is_supported_measurement_dynamic_key(section: &IniSection, entry: &IniEntry) -> bool {
    section.normalized_name == "measurements"
        && (entry.normalized_key.starts_with("metrics_to_skip_")
            || entry.normalized_key.starts_with("metrics_to_save_")
            || is_empty_python_custom_measurement_key(entry))
}

fn is_empty_python_custom_measurement_key(entry: &IniEntry) -> bool {
    matches!(
        entry.normalized_key.as_str(),
        "channel_indipendent_custom_metrics_to_save" | "mixed_combine_metrics_to_skip"
    ) && split_multiline_value(&entry.value).is_empty()
}

fn required_section<'a>(ini: &'a IniFile, name: &str) -> Result<&'a IniSection> {
    find_section(ini, name).ok_or_else(|| anyhow::anyhow!("Missing required INI section [{name}]"))
}

fn workflow_paths_section(ini: &IniFile) -> Result<&IniSection> {
    find_section(ini, "paths_info")
        .or_else(|| find_section(ini, "paths_to_segment"))
        .ok_or_else(|| anyhow::anyhow!("Missing required INI section [paths_info]"))
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

fn optional_python_string(section: &IniSection, key: &str) -> Option<String> {
    optional_key(section, key)
        .map(|entry| entry.value.trim())
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("none"))
        .map(str::to_string)
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

fn parse_segmentation_params(section: &IniSection) -> Result<SegmentationParams> {
    let defaults = SegmentationParams::default();
    Ok(SegmentationParams {
        tile: optional_usize(section, "tile", defaults.tile)?,
        batch_size: optional_usize(section, "batch_size", defaults.batch_size)?,
        cellprob_threshold: optional_f32(
            section,
            "cellprob_threshold",
            defaults.cellprob_threshold,
        )?,
        niter: optional_usize(section, "niter", defaults.niter)?,
        min_size: optional_usize(section, "min_size", defaults.min_size)?,
    })
}

fn validate_supported_tracker(initialization: &IniSection, workflow_path: &Path) -> Result<()> {
    let Some(tracker_name) = optional_python_string(initialization, "tracker_name") else {
        return Ok(());
    };
    let normalized = tracker_name.trim().to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        "overlap" | "cellacdc" | "cellacdc_tracker"
    ) {
        return Ok(());
    }
    bail!(
        "Workflow {} requests tracker_name {:?}, but cellacdc-rs currently supports only overlap/CellACDC IoA tracking.",
        workflow_path.display(),
        tracker_name
    )
}

fn parse_tracking_config(tracker_params: Option<&IniSection>) -> Result<TrackingConfig> {
    let ioa_threshold = tracker_params
        .and_then(|section| {
            optional_key(section, "ioa_thresh")
                .or_else(|| optional_key(section, "ioa_threshold"))
                .or_else(|| optional_key(section, "overlap_threshold"))
        })
        .map(|entry| parse_f32(entry, &entry.original_key))
        .transpose()?
        .unwrap_or(0.4);
    let assign_unique_new_ids = parse_optional_bool(
        tracker_params.and_then(|section| optional_key(section, "assign_unique_new_IDs")),
    )?
    .unwrap_or(true);
    let overlap_denominator = tracker_params
        .and_then(|section| optional_python_string(section, "denom_overlap_matrix"))
        .map(|value| parse_overlap_denominator(&value))
        .transpose()?
        .unwrap_or(OverlapDenominator::AreaPrev);
    Ok(TrackingConfig {
        ioa_threshold,
        assign_unique_new_ids,
        overlap_denominator,
    })
}

fn parse_overlap_denominator(value: &str) -> Result<OverlapDenominator> {
    match value.trim().to_ascii_lowercase().as_str() {
        "area_prev" => Ok(OverlapDenominator::AreaPrev),
        "union" => Ok(OverlapDenominator::Union),
        _ => bail!(
            "Unsupported tracker_params.denom_overlap_matrix {:?}; expected 'area_prev' or 'union'",
            value
        ),
    }
}

fn resolve_segmentation_model_path(
    workflow_dir: &Path,
    rust_cli: Option<&IniSection>,
    initialization: &IniSection,
    init_model_params: Option<&IniSection>,
    segm_params: &IniSection,
    workflow_path: &Path,
) -> Result<PathBuf> {
    for (section, key) in [
        (rust_cli, "model_path"),
        (init_model_params, "model_path"),
        (Some(segm_params), "model_path"),
    ] {
        if let Some(value) = section.and_then(|section| optional_python_string(section, key)) {
            return Ok(resolve_path_from_workflow_dir(
                workflow_dir,
                Path::new(&value),
            ));
        }
    }

    if let Some(model_name) = optional_python_string(initialization, "model_name") {
        if looks_like_model_path(&model_name) {
            return Ok(resolve_path_from_workflow_dir(
                workflow_dir,
                Path::new(&model_name),
            ));
        }
        bail!(
            "Workflow {} uses initialization.model_name {:?}, but cellacdc-rs can only resolve explicit model paths. Add [rust_cli].model_path or init_segmentation_model_params.model_path.",
            workflow_path.display(),
            model_name
        );
    }

    bail!(
        "Workflow {} is missing a model path. Add [rust_cli].model_path or init_segmentation_model_params.model_path.",
        workflow_path.display()
    )
}

fn looks_like_model_path(value: &str) -> bool {
    value.contains('/')
        || value.contains('\\')
        || Path::new(value).is_absolute()
        || Path::new(value).extension().is_some()
}

fn parse_standard_postprocess_config(section: &IniSection) -> Result<PostprocessConfig> {
    Ok(PostprocessConfig {
        min_area: optional_python_usize(section, "min_area")?,
        min_solidity: optional_python_f64(section, "min_solidity")?,
        max_elongation: optional_python_f64(section, "max_elongation")?,
        min_obj_no_zslices: optional_python_usize(section, "min_obj_no_zslices")?,
    })
}

fn parse_preprocess_steps(ini: &IniFile) -> Result<Vec<PreprocessStep>> {
    let mut steps = Vec::<(usize, PreprocessStep)>::new();
    for section in &ini.sections {
        let Some(step_text) = section.normalized_name.strip_prefix("preprocess.step") else {
            continue;
        };
        let step_index = step_text.parse::<usize>().unwrap_or(usize::MAX);
        let Some(method) = optional_python_string(section, "method") else {
            continue;
        };
        let normalized_method = normalize_name(&method);
        if normalized_method == "gaussian filter" {
            let sigma = optional_key(section, "sigma")
                .map(parse_f32_vector)
                .transpose()?
                .unwrap_or_else(|| vec![0.75]);
            let (sigma_y, sigma_x) = preprocess_2d_radii(&sigma, "sigma")?;
            steps.push((
                step_index,
                PreprocessStep::GaussianFilter { sigma_y, sigma_x },
            ));
        } else if normalized_method == "remove hot pixels" {
            steps.push((step_index, PreprocessStep::RemoveHotPixels));
        } else if normalized_method == "spot detector filter" {
            let radii = optional_key(section, "spots_zyx_radii_pxl")
                .map(parse_f32_vector)
                .transpose()?
                .unwrap_or_else(|| vec![3.0, 5.0, 5.0]);
            let (radius_y, radius_x) = preprocess_2d_radii(&radii, "spots_zyx_radii_pxl")?;
            steps.push((
                step_index,
                PreprocessStep::SpotDetectorFilter { radius_y, radius_x },
            ));
        } else if normalized_method == "ridge filter" {
            let sigmas = optional_key(section, "sigmas")
                .map(parse_f32_vector)
                .transpose()?
                .unwrap_or_else(|| vec![1.0, 2.0]);
            steps.push((step_index, PreprocessStep::RidgeFilter { sigmas }));
        } else if normalized_method == "enhance speckles" {
            let radius = optional_usize(section, "radius", 15)?;
            steps.push((step_index, PreprocessStep::EnhanceSpeckles { radius }));
        } else if normalized_method == "correct illumination" {
            let block_size = optional_usize(section, "block_size", 45)?;
            let approximate_object_diameter = optional_key(section, "approximate_object_diameter")
                .map(|entry| parse_f32(entry, "approximate_object_diameter"))
                .transpose()?
                .unwrap_or(15.0);
            let apply_gaussian_filter =
                parse_optional_bool(optional_key(section, "apply_gaussian_filter"))?
                    .unwrap_or(true);
            steps.push((
                step_index,
                PreprocessStep::CorrectIllumination {
                    block_size,
                    approximate_object_diameter,
                    apply_gaussian_filter,
                },
            ));
        } else if is_fucci_preprocess_method(&normalized_method) {
            let do_basicpy_background_correction =
                parse_optional_bool(optional_key(section, "do_basicpy_background_correction"))?
                    .unwrap_or(true);
            if do_basicpy_background_correction {
                continue;
            }
            let correct_illumination_toggle =
                parse_optional_bool(optional_key(section, "correct_illumination_toggle"))?
                    .unwrap_or(false);
            if correct_illumination_toggle {
                let block_size = optional_usize(section, "block_size", 120)?;
                let approximate_object_diameter =
                    optional_key(section, "approximate_object_diameter")
                        .map(|entry| parse_f32(entry, "approximate_object_diameter"))
                        .transpose()?
                        .unwrap_or(25.0);
                let apply_gaussian_filter =
                    parse_optional_bool(optional_key(section, "apply_gaussian_filter"))?
                        .unwrap_or(true);
                steps.push((
                    step_index,
                    PreprocessStep::CorrectIllumination {
                        block_size,
                        approximate_object_diameter,
                        apply_gaussian_filter,
                    },
                ));
            }
            let enhance_speckles_toggle =
                parse_optional_bool(optional_key(section, "enhance_speckles_toggle"))?
                    .unwrap_or(true);
            if enhance_speckles_toggle {
                let radius = optional_usize(section, "speckle_radius", 25)?;
                steps.push((step_index, PreprocessStep::EnhanceSpeckles { radius }));
            }
        }
    }
    steps.sort_by_key(|(step_index, _)| *step_index);
    Ok(steps.into_iter().map(|(_, step)| step).collect())
}

fn is_fucci_preprocess_method(normalized_method: &str) -> bool {
    matches!(
        normalized_method,
        "fucci pre-processing" | "fucci preprocessing" | "fucci_filter" | "fucci filter"
    )
}

fn preprocess_2d_radii(values: &[f32], label: &str) -> Result<(f32, f32)> {
    match values {
        [radius] => Ok((*radius, *radius)),
        [radius_y, radius_x] => Ok((*radius_y, *radius_x)),
        [_, radius_y, radius_x] => Ok((*radius_y, *radius_x)),
        _ => bail!(
            "Invalid preprocess vector for {}; expected one, two, or three numbers",
            label
        ),
    }
}

fn parse_f32_vector(entry: &IniEntry) -> Result<Vec<f32>> {
    let cleaned = entry
        .value
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim_start_matches('[')
        .trim_end_matches(']');
    let mut values = Vec::new();
    for value in cleaned
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        values.push(value.parse::<f32>().with_context(|| {
            format!(
                "Failed to parse [{}].{} vector value {:?} as number",
                entry.original_key, entry.original_key, value
            )
        })?);
    }
    if values.is_empty() {
        bail!(
            "Invalid vector for key {}; expected comma-separated numbers",
            entry.original_key
        );
    }
    Ok(values)
}

fn optional_usize(section: &IniSection, key: &str, default: usize) -> Result<usize> {
    optional_key(section, key)
        .map(|entry| parse_usize(entry, key))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn optional_python_usize(section: &IniSection, key: &str) -> Result<Option<usize>> {
    optional_key(section, key)
        .map(|entry| {
            let value = entry.value.trim();
            if value.is_empty() || value.eq_ignore_ascii_case("none") {
                Ok(None)
            } else {
                parse_usize(entry, key).map(Some)
            }
        })
        .transpose()
        .map(Option::flatten)
}

fn optional_python_f64(section: &IniSection, key: &str) -> Result<Option<f64>> {
    optional_key(section, key)
        .map(|entry| {
            let value = entry.value.trim();
            if value.is_empty() || value.eq_ignore_ascii_case("none") {
                Ok(None)
            } else {
                parse_f64(entry, key).map(Some)
            }
        })
        .transpose()
        .map(Option::flatten)
}

fn parse_f32(entry: &IniEntry, label: &str) -> Result<f32> {
    entry.value.trim().parse::<f32>().with_context(|| {
        format!(
            "Failed to parse [{}].{} as number for {}",
            entry.original_key, entry.original_key, label
        )
    })
}

fn parse_f64(entry: &IniEntry, label: &str) -> Result<f64> {
    entry.value.trim().parse::<f64>().with_context(|| {
        format!(
            "Failed to parse [{}].{} as number for {}",
            entry.original_key, entry.original_key, label
        )
    })
}

fn optional_f32(section: &IniSection, key: &str, default: f32) -> Result<f32> {
    optional_key(section, key)
        .map(|entry| parse_f32(entry, key))
        .transpose()
        .map(|value| value.unwrap_or(default))
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

fn resolve_measurement_workflow_targets(
    input_paths: &[PathBuf],
    stop_frames: &[Option<usize>],
    path: &Path,
) -> Result<Vec<MeasurementWorkflowTarget>> {
    let mut expanded = Vec::<(PathBuf, PathBuf)>::new();
    for input_path in input_paths {
        for position_path in resolve_measurement_input_path(input_path)? {
            expanded.push((input_path.clone(), position_path));
        }
    }
    if expanded.is_empty() {
        bail!(
            "Workflow {} did not resolve any Cell-ACDC measurement targets",
            path.display()
        );
    }

    let stop_frames = broadcast_stop_frames(stop_frames, expanded.len(), path)?;
    Ok(expanded
        .into_iter()
        .zip(stop_frames)
        .map(
            |((input_path, position_path), stop_frame)| MeasurementWorkflowTarget {
                input_path,
                position_path,
                stop_frame,
            },
        )
        .collect())
}

fn resolve_measurement_input_path(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        let parent = path.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "Workflow target file has no parent directory: {}",
                path.display()
            )
        })?;
        return Ok(vec![resolve_measurement_position(parent)?.position_dir]);
    }

    if path.file_name().and_then(|name| name.to_str()) == Some("Images")
        || path.join("Images").is_dir()
    {
        return Ok(vec![resolve_measurement_position(path)?.position_dir]);
    }

    Ok(discover_measurement_experiment(path)?
        .positions
        .into_iter()
        .map(|position| position.position_dir)
        .collect())
}

fn parse_measurement_workflow_config(
    section: &IniSection,
    targets: Vec<MeasurementWorkflowTarget>,
    overwrite_policy: OverwritePolicy,
    path: &Path,
) -> Result<MeasurementWorkflowConfig> {
    let segm_endname = optional_python_string(section, "end_filename_segm");
    if segm_endname.is_none() {
        bail!(
            "measurements.end_filename_segm cannot be empty in {}",
            path.display()
        );
    }
    Ok(MeasurementWorkflowConfig {
        targets,
        segm_endname,
        overwrite_policy,
        channel_names: measurement_channel_names(section)?,
        metric_options: measurement_metric_options(section)?,
        save_object_counts_table: parse_optional_bool(optional_key(
            section,
            "save_object_counts_table",
        ))?
        .unwrap_or(false),
    })
}

fn measurement_channel_names(section: &IniSection) -> Result<Option<Vec<String>>> {
    let channels = optional_key(section, "channels")
        .map(|entry| {
            split_multiline_value(&entry.value)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let skip = optional_key(section, "channel_names_to_skip")
        .map(|entry| {
            split_multiline_value(&entry.value)
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let process = optional_key(section, "channel_names_to_process").map(|entry| {
        split_multiline_value(&entry.value)
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
    });

    if channels.is_empty() && process.is_none() && skip.is_empty() {
        return Ok(None);
    }

    if !skip.is_empty()
        && channels.is_empty()
        && process
            .as_ref()
            .map(|values| values.is_empty())
            .unwrap_or(true)
    {
        bail!(
            "measurements.channel_names_to_skip requires non-empty measurements.channels or measurements.channel_names_to_process"
        );
    }

    let base = if channels.is_empty() {
        process
            .as_ref()
            .map(|values| values.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default()
    } else {
        channels
    };
    let selected = base
        .into_iter()
        .filter(|channel| !skip.contains(channel))
        .filter(|channel| {
            process
                .as_ref()
                .map(|values| values.contains(channel))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    Ok(Some(selected))
}

fn measurement_metric_options(section: &IniSection) -> Result<Option<MeasurementMetricOptions>> {
    let size_metrics = optional_key(section, "size_metrics_to_save").map(|entry| {
        split_multiline_value(&entry.value)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let regionprops = optional_key(section, "regionprops_to_save").map(|entry| {
        split_multiline_value(&entry.value)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let calc_size_for_each_zslice =
        parse_optional_bool(optional_key(section, "calc_for_each_zslice_size"))?.unwrap_or(false);

    let mut saw_channel_metrics = false;
    let mut channel_metrics = BTreeMap::<String, Vec<String>>::new();
    let mut channel_metrics_to_skip = BTreeMap::<String, Vec<String>>::new();
    let calc_for_each_zslice_channels = match optional_key(section, "calc_for_each_zslice_channels")
    {
        Some(entry) => parse_calc_for_each_zslice_channels(entry)?,
        None => BTreeMap::new(),
    };
    for entry in &section.entries {
        if let Some(channel_name) = entry.normalized_key.strip_prefix("metrics_to_save_") {
            saw_channel_metrics = true;
            channel_metrics.insert(
                normalize_metric_key(channel_name),
                split_multiline_value(&entry.value)
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            );
        } else if let Some(channel_name) = entry.normalized_key.strip_prefix("metrics_to_skip_") {
            channel_metrics_to_skip.insert(
                normalize_metric_key(channel_name),
                split_multiline_value(&entry.value)
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            );
        }
    }

    if size_metrics.is_none()
        && regionprops.is_none()
        && !saw_channel_metrics
        && channel_metrics_to_skip.is_empty()
        && calc_for_each_zslice_channels.is_empty()
        && !calc_size_for_each_zslice
    {
        return Ok(None);
    }

    Ok(Some(MeasurementMetricOptions {
        channel_metrics: saw_channel_metrics.then_some(channel_metrics),
        channel_metrics_to_skip,
        calc_for_each_zslice_channels,
        calc_size_for_each_zslice,
        size_metrics,
        regionprops,
    }))
}

fn parse_calc_for_each_zslice_channels(entry: &IniEntry) -> Result<BTreeMap<String, bool>> {
    let mut channels = BTreeMap::new();
    for value in split_multiline_value(&entry.value) {
        let (channel, enabled) = value.split_once(',').ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid measurements.calc_for_each_zslice_channels value {:?}; expected channel,true|false",
                value
            )
        })?;
        let enabled = parse_bool_value(enabled).ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid boolean in measurements.calc_for_each_zslice_channels value {:?}",
                value
            )
        })?;
        let channel = channel.trim();
        if channel.is_empty() {
            bail!(
                "Invalid measurements.calc_for_each_zslice_channels value {:?}; channel name cannot be empty",
                value
            );
        }
        channels.insert(normalize_metric_key(channel), enabled);
    }
    Ok(channels)
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
    use crate::image_io::write_mask_npz;
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
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(segmentation.targets.len(), 1);
        assert_eq!(segmentation.targets[0].stop_frame, Some(2));
        assert_eq!(segmentation.phase_channel, "phase");
        assert_eq!(segmentation.fluo_channel, "phase");
        assert_eq!(
            segmentation.tracking,
            Some(TrackingConfig {
                ioa_threshold: 0.6,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::AreaPrev,
            })
        );
        assert_eq!(segmentation.segm_endname.as_deref(), Some("rust"));
        assert_eq!(workflow.measurement, None);
        Ok(())
    }

    #[test]
    fn parses_legacy_paths_to_segment_section() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[workflow]\n\
type = segmentation and/or tracking\n\n\
[paths_to_segment]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[initialization]\n\
user_ch_name = phase\n\
segm_endname = rust\n\
do_tracking = false\n\n\
[segmentation_model_params]\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(segmentation.targets.len(), 1);
        assert_eq!(segmentation.targets[0].position.position_dir, position);
        assert_eq!(segmentation.targets[0].stop_frame, Some(1));
        Ok(())
    }

    #[test]
    fn parses_python_none_for_optional_segmentation_strings() -> Result<()> {
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
segm_endname = None\n\
do_tracking = false\n\n\
[segmentation_model_params]\n\n\
[rust_cli]\n\
model_path = model.onnx\n\
fluo_channel = None\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(segmentation.fluo_channel, "phase");
        assert_eq!(segmentation.segm_endname, None);
        Ok(())
    }

    #[test]
    fn rust_fluo_channel_overrides_python_second_channel_name() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        write_test_stack(&position.join("Images").join("demo_mCherry.tif"), &[3, 4])?;
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
second_channel_name = gfp\n\n\
[segmentation_model_params]\n\n\
[rust_cli]\n\
model_path = model.onnx\n\
fluo_channel = mCherry\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(segmentation.fluo_channel, "mCherry");
        Ok(())
    }

    #[test]
    fn parses_python_model_path_without_rust_cli_section() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("workflow.ini");
        fs::create_dir_all(temp.path().join("models"))?;
        fs::write(temp.path().join("models").join("custom.onnx"), "").expect("model file");
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
do_tracking = false\n\n\
[init_segmentation_model_params]\n\
model_path = models/custom.onnx\n\n\
[segmentation_model_params]\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.model_path,
            temp.path().join("models").join("custom.onnx")
        );
        Ok(())
    }

    #[test]
    fn rejects_python_builtin_model_name_without_model_path() -> Result<()> {
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
model_name = cyto3\n\
do_tracking = false\n\n\
[segmentation_model_params]\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).expect_err("bare model name rejected");
        assert!(err.to_string().contains("initialization.model_name"));
        assert!(err.to_string().contains("explicit model paths"));
        Ok(())
    }

    #[test]
    fn parses_python_do_save_false_for_segmentation_workflow() -> Result<()> {
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
segm_endname = preview\n\
do_tracking = false\n\
do_save = false\n\n\
[segmentation_model_params]\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert!(!segmentation.save_outputs);
        Ok(())
    }

    #[test]
    fn parses_standard_postprocess_min_area_when_enabled() -> Result<()> {
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
do_postprocess = true\n\n\
[segmentation_model_params]\n\n\
[standard_postprocess_features]\n\
min_area = 4\n\
max_elongation = 2.5\n\
min_solidity = 0.8\n\
min_obj_no_zslices = 3\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.postprocess,
            Some(PostprocessConfig {
                min_area: Some(4),
                min_solidity: Some(0.8),
                max_elongation: Some(2.5),
                min_obj_no_zslices: Some(3),
            })
        );
        Ok(())
    }

    #[test]
    fn parses_python_segmentation_workflow_with_deferred_sections() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        write_test_stack(&position.join("Images").join("demo_gfp.tif"), &[3, 4])?;
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
model_name = cyto3\n\
tracker_name = overlap\n\
do_tracking = true\n\
do_postprocess = false\n\
do_save = true\n\
image_channel_tracker = phase\n\
isSegm3D = false\n\
use_ROI = false\n\
use_freehand_ROI = false\n\
second_channel_name = gfp\n\
use3DdataFor2Dsegm = false\n\
reduce_memory_usage = false\n\n\
[metadata]\n\
SizeT = 1\n\
SizeZ = 1\n\n\
[init_segmentation_model_params]\n\
diameter = 30\n\n\
[segmentation_model_params]\n\
cellprob_threshold = 0.0\n\
min_size = 15\n\
flow_threshold = 0.4\n\n\
[init_tracker_params]\n\
some_tracker_init = value\n\n\
[tracker_params]\n\
IoA_thresh = 0.6\n\
max_distance = 10\n\n\
[standard_postprocess_features]\n\
some_feature = 1\n\n\
[custom_postprocess_features]\n\
custom_feature = (0, 1)\n\n\
[preprocess.step1]\n\
method = Remove hot pixels\n\n\
[preprocess.step2]\n\
method = Gaussian filter\n\
sigma = 0, 1, 2\n\n\
[preprocess.step3]\n\
method = Spot detector filter\n\
spots_zyx_radii_pxl = 3, 4, 5\n\n\
[preprocess.step4]\n\
method = Ridge filter\n\
sigmas = 1, 2\n\n\
[preprocess.step5]\n\
method = Enhance speckles\n\
radius = 7\n\n\
[preprocess.step6]\n\
method = Correct illumination\n\
block_size = 3\n\
approximate_object_diameter = 5\n\
apply_gaussian_filter = False\n\n\
[postprocess_features.category]\n\
names =\n  feature_a\n\n\
[rust_cli]\n\
model_path = models/demo.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        assert_eq!(workflow.kind, WorkflowKind::SegmentationAndTracking);
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(segmentation.targets.len(), 1);
        assert_eq!(
            segmentation.tracking,
            Some(TrackingConfig {
                ioa_threshold: 0.6,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::AreaPrev,
            })
        );
        assert_eq!(segmentation.fluo_channel, "gfp");
        assert_eq!(segmentation.params.tile, 256);
        assert_eq!(segmentation.params.batch_size, 1);
        assert_eq!(segmentation.params.niter, 200);
        assert_eq!(
            segmentation.preprocess_steps,
            vec![
                PreprocessStep::RemoveHotPixels,
                PreprocessStep::GaussianFilter {
                    sigma_y: 1.0,
                    sigma_x: 2.0
                },
                PreprocessStep::SpotDetectorFilter {
                    radius_y: 4.0,
                    radius_x: 5.0
                },
                PreprocessStep::RidgeFilter {
                    sigmas: vec![1.0, 2.0]
                },
                PreprocessStep::EnhanceSpeckles { radius: 7 },
                PreprocessStep::CorrectIllumination {
                    block_size: 3,
                    approximate_object_diameter: 5.0,
                    apply_gaussian_filter: false
                }
            ]
        );
        assert!(segmentation.save_outputs);
        assert!(!segmentation.use_data_prep_roi);
        assert!(!segmentation.use_data_prep_free_roi);
        Ok(())
    }

    #[test]
    fn parses_fucci_preprocess_without_basicpy_as_supported_steps() -> Result<()> {
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
do_tracking = false\n\n\
[segmentation_model_params]\n\n\
[preprocess.step1]\n\
method = FUCCI pre-processing\n\
do_basicpy_background_correction = False\n\
correct_illumination_toggle = True\n\
enhance_speckles_toggle = True\n\
block_size = 7\n\
approximate_object_diameter = 9\n\
apply_gaussian_filter = False\n\
speckle_radius = 3\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.preprocess_steps,
            vec![
                PreprocessStep::CorrectIllumination {
                    block_size: 7,
                    approximate_object_diameter: 9.0,
                    apply_gaussian_filter: false
                },
                PreprocessStep::EnhanceSpeckles { radius: 3 }
            ]
        );
        Ok(())
    }

    #[test]
    fn rejects_unsupported_python_tracker_name() -> Result<()> {
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
tracker_name = Trackastra\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).expect_err("unsupported tracker rejected");
        assert!(err.to_string().contains("tracker_name"));
        assert!(err.to_string().contains("overlap"));
        Ok(())
    }

    #[test]
    fn parses_overlap_threshold_tracker_alias() -> Result<()> {
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
tracker_name = overlap\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\n\
[tracker_params]\n\
overlap_threshold = 0.7\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.tracking,
            Some(TrackingConfig {
                ioa_threshold: 0.7,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::AreaPrev,
            })
        );
        Ok(())
    }

    #[test]
    fn parses_assign_unique_new_ids_tracker_param() -> Result<()> {
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
tracker_name = overlap\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\n\
[tracker_params]\n\
IoA_thresh = 0.7\n\
assign_unique_new_IDs = False\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.tracking,
            Some(TrackingConfig {
                ioa_threshold: 0.7,
                assign_unique_new_ids: false,
                overlap_denominator: OverlapDenominator::AreaPrev,
            })
        );
        Ok(())
    }

    #[test]
    fn parses_union_tracker_overlap_denominator() -> Result<()> {
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
tracker_name = overlap\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\n\
[tracker_params]\n\
denom_overlap_matrix = union\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let segmentation = workflow.segmentation.expect("segmentation workflow");
        assert_eq!(
            segmentation.tracking,
            Some(TrackingConfig {
                ioa_threshold: 0.4,
                assign_unique_new_ids: true,
                overlap_denominator: OverlapDenominator::Union,
            })
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_tracker_overlap_denominator() -> Result<()> {
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
tracker_name = overlap\n\
do_tracking = true\n\n\
[segmentation_model_params]\n\n\
[tracker_params]\n\
denom_overlap_matrix = area_curr\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).expect_err("invalid denominator rejected");
        assert!(err.to_string().contains("denom_overlap_matrix"));
        assert!(err.to_string().contains("area_prev"));
        assert!(err.to_string().contains("union"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_optional_segmentation_param() -> Result<()> {
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
do_tracking = false\n\n\
[segmentation_model_params]\n\
min_size = many\n\n\
[rust_cli]\n\
model_path = model.onnx\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).expect_err("invalid min_size rejected");
        assert!(err.to_string().contains("min_size"));
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
    fn parses_python_measurement_only_workflow() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 2\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
channels =\n  phase\n  gfp\n\
end_filename_segm = segm_rust\n\
channel_names_to_skip =\n  gfp\n\
channel_names_to_process =\n  phase\n\
calc_for_each_zslice_channels =\n  phase,true\n  gfp,false\n\
calc_for_each_zslice_size = True\n\
size_metrics_to_save =\n  cell_area_pxl\n\
regionprops_to_save =\n  centroid\n\
metrics_to_save_phase =\n  mean\n\
metrics_to_skip_phase =\n\
save_object_counts_table = True\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        assert_eq!(workflow.kind, WorkflowKind::Measurements);
        assert!(workflow.segmentation.is_none());
        let measurement = workflow.measurement.expect("measurement workflow");
        assert_eq!(measurement.targets.len(), 1);
        assert_eq!(measurement.targets[0].position_path, position);
        assert_eq!(measurement.targets[0].stop_frame, Some(2));
        assert_eq!(measurement.segm_endname.as_deref(), Some("segm_rust"));
        assert_eq!(measurement.overwrite_policy, OverwritePolicy::Overwrite);
        assert_eq!(measurement.channel_names, Some(vec!["phase".to_string()]));
        assert!(measurement.save_object_counts_table);
        let metric_options = measurement.metric_options.expect("metric options");
        assert_eq!(
            metric_options
                .channel_metrics
                .as_ref()
                .and_then(|metrics| metrics.get("phase")),
            Some(&vec!["mean".to_string()])
        );
        assert_eq!(
            metric_options.calc_for_each_zslice_channels.get("phase"),
            Some(&true)
        );
        assert_eq!(
            metric_options.calc_for_each_zslice_channels.get("gfp"),
            Some(&false)
        );
        assert!(metric_options.calc_size_for_each_zslice);
        assert_eq!(
            metric_options.size_metrics,
            Some(vec!["cell_area_pxl".to_string()])
        );
        assert_eq!(
            metric_options.regionprops,
            Some(vec!["centroid".to_string()])
        );
        Ok(())
    }

    #[test]
    fn parses_measurement_process_channels_without_full_channel_list() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
end_filename_segm = segm_rust\n\
channel_names_to_process =\n  phase\n\
channel_names_to_skip =\n  gfp\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        let measurement = workflow.measurement.expect("measurement workflow");
        assert_eq!(measurement.channel_names, Some(vec!["phase".to_string()]));
        Ok(())
    }

    #[test]
    fn rejects_measurement_skip_channels_without_channel_base() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
end_filename_segm = segm_rust\n\
channel_names_to_skip =\n  gfp\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).expect_err("missing channel base rejected");
        assert!(err.to_string().contains("channel_names_to_skip"));
        assert!(err.to_string().contains("channel_names_to_process"));
        Ok(())
    }

    #[test]
    fn runs_python_measurement_only_workflow() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let images = position.join("Images");
        write_test_stack(&images.join("demo_gfp.tif"), &[30, 40])?;
        write_mask_npz(
            &images.join("demo_segm_rust.npz"),
            &[
                1, 1, //
                0, 0, //
                2, 0, //
                2, 0, //
            ],
            2,
            2,
            2,
        )?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
channels =\n  phase\n  gfp\n\
end_filename_segm = segm_rust\n\
channel_names_to_skip =\n  gfp\n\
metrics_to_save_phase =\n  mean\n\
size_metrics_to_save =\n  cell_area_pxl\n\
regionprops_to_save =\n\
save_object_counts_table = True\n",
                position.display()
            ),
        )?;

        let report = run_workflow_file(&workflow_path, WorkflowRunOptions { debug: false })?;
        assert!(report.segmentation_results.is_empty());
        assert_eq!(report.measurement_results.len(), 1);
        let result = &report.measurement_results[0];
        assert_eq!(result.frames_processed, 1);
        assert_eq!(
            result
                .outputs
                .segm_npz_path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("demo_segm_rust.npz")
        );
        assert_eq!(
            result
                .outputs
                .acdc_output_csv_path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("demo_acdc_output_rust.csv")
        );
        assert!(result.outputs.objects_count_csv_path.exists());

        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(headers.iter().any(|header| header == "phase_mean"));
        assert!(!headers.iter().any(|header| header == "phase_sum"));
        assert!(!headers.iter().any(|header| header == "gfp_mean"));
        assert!(headers.iter().any(|header| header == "cell_area_pxl"));
        Ok(())
    }

    #[test]
    fn runs_python_measurement_workflow_with_manual_background_metric() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        write_test_stack_pixels(
            &images.join("demo_phase.tif"),
            &[
                10, 14, //
                2, 4, //
            ],
        )?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,1\nSizeZ,1\n",
        )?;
        write_mask_npz(
            &images.join("demo_segm.npz"),
            &[
                1, 1, //
                0, 0, //
            ],
            1,
            2,
            2,
        )?;
        write_mask_npz(
            &images.join("demo_manualBackground.npz"),
            &[
                0, 0, //
                1, 1, //
            ],
            1,
            2,
            2,
        )?;

        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
channels =\n  phase\n\
end_filename_segm = segm\n\
metrics_to_save_phase =\n  mean_manualBkgr\n\
size_metrics_to_save =\n\
regionprops_to_save =\n",
                position.display()
            ),
        )?;

        let report = run_workflow_file(&workflow_path, WorkflowRunOptions { debug: false })?;
        assert_eq!(report.measurement_results.len(), 1);
        let result = &report.measurement_results[0];
        let mut reader = csv::Reader::from_path(&result.outputs.acdc_output_csv_path)?;
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let row = reader.records().next().transpose()?.expect("row");
        let metric_idx = headers
            .iter()
            .position(|header| header == "phase_mean_manualBkgr")
            .expect("manual background metric header");
        let value = row
            .get(metric_idx)
            .expect("manual background metric value")
            .parse::<f64>()?;
        assert_eq!(value, 9.0);
        assert!(!headers.iter().any(|header| header == "phase_mean"));
        assert!(!headers
            .iter()
            .any(|header| header == "phase_manualBkgr_bkgrVal_mean"));
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
unexpected_key = value\n\n\
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
            .contains("Unsupported workflow keys: [initialization].unexpected_key"));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_measurement_custom_metric_keys() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
channels =\n  phase\n\
end_filename_segm = segm\n\
channel_indipendent_custom_metrics_to_save =\n  custom_metric\n\
mixed_combine_metrics_to_skip =\n  mixed_metric\n",
                position.display()
            ),
        )?;

        let err = parse_workflow_file(&workflow_path).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(
            "Unsupported workflow keys: [measurements].channel_indipendent_custom_metrics_to_save"
        ));
        assert!(message.contains("[measurements].mixed_combine_metrics_to_skip"));
        Ok(())
    }

    #[test]
    fn accepts_empty_python_measurement_custom_metric_keys_as_noop() -> Result<()> {
        let temp = tempdir()?;
        let position = write_test_position(temp.path(), "Position_1")?;
        let workflow_path = temp.path().join("measurement_workflow.ini");
        fs::write(
            &workflow_path,
            format!(
                "[paths_info]\n\
paths =\n  {}\n\
stop_frame_numbers = 1\n\n\
[workflow]\n\
type = measurements\n\n\
[measurements]\n\
channels =\n  phase\n\
end_filename_segm = segm\n\
channel_indipendent_custom_metrics_to_save =\n\
mixed_combine_metrics_to_skip =\n",
                position.display()
            ),
        )?;

        let workflow = parse_workflow_file(&workflow_path)?;
        assert_eq!(workflow.kind, WorkflowKind::Measurements);
        assert!(workflow.measurement.is_some());
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

    fn write_test_stack_pixels(path: &Path, pixels: &[u16]) -> Result<()> {
        let file = fs::File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        encoder.write_image::<colortype::Gray16>(2, 2, pixels)?;
        Ok(())
    }
}
