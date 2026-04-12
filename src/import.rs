use crate::bioformats_bridge::{
    run_bioformats_export, run_bioformats_probe, BioFormatsExportRequest, BioFormatsProbeRequest,
};
use crate::image_io::{
    inspect_image_stack, inspect_image_volume, load_image_stack_as_f32, load_image_volume_as_f32,
};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use tiff::encoder::{colortype, TiffEncoder};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportSourceKind {
    Npz,
    H5,
    Tiff,
    VendorMicroscopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportLayoutKind {
    SingleFileMultiPosition,
    FilePerPosition,
    FilePerChannel,
    CustomMapping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportReaderBackend {
    Auto,
    Native,
    BioFormatsJvmBridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportOutputFormat {
    Tiff,
    H5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportConflictMode {
    OverwritePositionFiles,
    AddFilesToExistingExperiment,
    CreateNewPositions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataReusePolicy {
    ConfirmEverySource,
    UseForRemainingSources,
    TrustReaderForRemainingSources,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportSourceEntry {
    pub path: PathBuf,
    pub detected_kind: ImportSourceKind,
    pub backend_used: Option<ImportReaderBackend>,
    pub series_count: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportMetadataDraft {
    pub lens_na: f32,
    pub size_t: usize,
    pub size_z: usize,
    pub size_c: usize,
    pub size_s: usize,
    pub time_increment: f32,
    pub time_increment_unit: String,
    pub physical_size_x: f32,
    pub physical_size_y: f32,
    pub physical_size_z: f32,
    pub physical_size_unit: String,
    pub channel_names: Vec<String>,
    pub emission_wavelengths: Vec<f32>,
    pub image_name: String,
    pub metadata_xml: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportSamplePlaneSet {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub size_z: usize,
    pub channel_names: Vec<String>,
    pub pixels: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportSelection {
    pub selected_positions: Vec<String>,
    pub save_channels: Vec<bool>,
    pub time_range: Option<(usize, usize)>,
    pub add_image_name: bool,
    pub output_format: ImportOutputFormat,
}

impl Default for ImportSelection {
    fn default() -> Self {
        Self {
            selected_positions: vec!["All Positions".to_string()],
            save_channels: Vec::new(),
            time_range: None,
            add_image_name: false,
            output_format: ImportOutputFormat::Tiff,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportExecutionConfig {
    pub layout_kind: ImportLayoutKind,
    pub backend: ImportReaderBackend,
    pub sources: Vec<PathBuf>,
    pub destination_experiment_dir: PathBuf,
    pub conflict_mode: ImportConflictMode,
    pub metadata_policy: MetadataReusePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportChannelPlan {
    pub source_path: PathBuf,
    pub destination_path: PathBuf,
    pub channel_name: String,
    pub source_series_index: Option<usize>,
    pub source_channel_index: usize,
    pub backend: ImportReaderBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportPositionPlan {
    pub position_name: String,
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub basename: String,
    pub metadata_csv_path: PathBuf,
    pub metadata_xml_path: PathBuf,
    pub channels: Vec<ImportChannelPlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportPlan {
    pub experiment_dir: PathBuf,
    pub positions: Vec<ImportPositionPlan>,
    pub metadata_by_position: BTreeMap<String, ImportMetadataDraft>,
    pub selection: ImportSelection,
    pub conflict_mode: ImportConflictMode,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportExecutionReport {
    pub experiment_dir: PathBuf,
    pub created_positions: Vec<PathBuf>,
    pub written_files: Vec<PathBuf>,
    pub skipped_files: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

pub fn detect_import_source_kind(path: impl AsRef<Path>) -> Option<ImportSourceKind> {
    let ext = path.as_ref().extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "npz" => Some(ImportSourceKind::Npz),
        "h5" | "hdf5" => Some(ImportSourceKind::H5),
        "tif" | "tiff" => Some(ImportSourceKind::Tiff),
        "czi" | "nd2" | "lif" | "ims" | "lsm" | "vsi" | "dv" | "ome" => {
            Some(ImportSourceKind::VendorMicroscopy)
        }
        _ => None,
    }
}

pub fn discover_import_sources(path: impl AsRef<Path>) -> Result<Vec<ImportSourceEntry>> {
    let path = path.as_ref();
    let mut sources = Vec::new();
    if path.is_file() {
        if let Some(kind) = detect_import_source_kind(path) {
            sources.push(ImportSourceEntry {
                path: path.to_path_buf(),
                detected_kind: kind,
                backend_used: None,
                series_count: None,
            });
        }
        return Ok(sources);
    }
    for entry in fs::read_dir(path).with_context(|| format!("Failed to read {}", path.display()))? {
        let entry = entry?;
        let file_path = entry.path();
        if !file_path.is_file() {
            continue;
        }
        if let Some(kind) = detect_import_source_kind(&file_path) {
            sources.push(ImportSourceEntry {
                path: file_path,
                detected_kind: kind,
                backend_used: None,
                series_count: None,
            });
        }
    }
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(sources)
}

pub fn classify_import_layout(
    sources: &[ImportSourceEntry],
    explicit_layout: Option<ImportLayoutKind>,
) -> Result<ImportLayoutKind> {
    if let Some(layout) = explicit_layout {
        return Ok(layout);
    }
    if sources.is_empty() {
        bail!("At least one import source is required");
    }
    if sources.len() == 1 {
        return Ok(ImportLayoutKind::SingleFileMultiPosition);
    }
    let stems = sources
        .iter()
        .filter_map(|entry| {
            entry
                .path
                .file_stem()
                .and_then(OsStr::to_str)
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let prefix = longest_common_prefix_all(&stems);
    let suffixes = stems
        .iter()
        .map(|stem| {
            stem.strip_prefix(&prefix)
                .unwrap_or(stem)
                .trim_matches('_')
                .to_string()
        })
        .collect::<Vec<_>>();
    let unique_suffixes = suffixes.iter().collect::<BTreeSet<_>>();
    if !prefix.is_empty() && unique_suffixes.len() == sources.len() {
        return Ok(ImportLayoutKind::FilePerChannel);
    }
    Ok(ImportLayoutKind::FilePerPosition)
}

pub fn probe_import_source(
    path: impl AsRef<Path>,
    backend: ImportReaderBackend,
) -> Result<ImportMetadataDraft> {
    let path = path.as_ref();
    let source_kind = detect_import_source_kind(path)
        .ok_or_else(|| anyhow!("Unsupported import source {}", path.display()))?;
    match resolve_backend(source_kind, backend) {
        ImportReaderBackend::Native => native_probe(path),
        ImportReaderBackend::BioFormatsJvmBridge => {
            let response = run_bioformats_probe(BioFormatsProbeRequest {
                path: path.to_path_buf(),
            })?;
            Ok(ImportMetadataDraft {
                lens_na: response.lens_na,
                size_t: response.size_t,
                size_z: response.size_z,
                size_c: response.size_c,
                size_s: response.size_s,
                time_increment: response.time_increment,
                time_increment_unit: response.time_increment_unit,
                physical_size_x: response.physical_size_x,
                physical_size_y: response.physical_size_y,
                physical_size_z: response.physical_size_z,
                physical_size_unit: response.physical_size_unit,
                channel_names: response.channel_names,
                emission_wavelengths: response.emission_wavelengths,
                image_name: response.image_name,
                metadata_xml: response.metadata_xml,
            })
        }
        ImportReaderBackend::Auto => unreachable!(),
    }
}

pub fn read_import_sample_planes(
    path: impl AsRef<Path>,
    backend: ImportReaderBackend,
) -> Result<ImportSamplePlaneSet> {
    let path = path.as_ref();
    let source_kind = detect_import_source_kind(path)
        .ok_or_else(|| anyhow!("Unsupported import source {}", path.display()))?;
    match resolve_backend(source_kind, backend) {
        ImportReaderBackend::Native => native_sample_planes(path),
        ImportReaderBackend::BioFormatsJvmBridge => {
            let response = run_bioformats_probe(BioFormatsProbeRequest {
                path: path.to_path_buf(),
            })?;
            Ok(ImportSamplePlaneSet {
                width: response.preview_width,
                height: response.preview_height,
                frames: response.size_t.max(1),
                size_z: response.size_z.max(1),
                channel_names: response.channel_names,
                pixels: response.preview_pixels,
            })
        }
        ImportReaderBackend::Auto => unreachable!(),
    }
}

pub fn build_import_plan(
    config: &ImportExecutionConfig,
    discovered_sources: &[ImportSourceEntry],
    metadata_drafts: &[ImportMetadataDraft],
    selection: &ImportSelection,
) -> Result<ImportPlan> {
    if config.sources.is_empty() {
        bail!("No sources selected for import");
    }
    if discovered_sources.len() != metadata_drafts.len() {
        bail!("Metadata draft count does not match the discovered source count");
    }
    let start_index =
        next_position_index(&config.destination_experiment_dir, config.conflict_mode)?;
    let mut metadata_by_position = BTreeMap::new();
    let mut warnings = Vec::new();
    let mut positions = Vec::new();
    let resolved_layout = classify_import_layout(discovered_sources, Some(config.layout_kind))?;
    match resolved_layout {
        ImportLayoutKind::FilePerPosition => {
            for (idx, (source, metadata)) in
                discovered_sources.iter().zip(metadata_drafts).enumerate()
            {
                let position_name = format!("Position_{}", start_index + idx);
                let basename = build_basename(&source.path, 0, selection.add_image_name);
                let position_dir = config.destination_experiment_dir.join(&position_name);
                let images_dir = position_dir.join("Images");
                let channel_name = metadata
                    .channel_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| infer_channel_name(&source.path));
                let destination_path = images_dir.join(match selection.output_format {
                    ImportOutputFormat::Tiff => format!("{basename}{channel_name}.tif"),
                    ImportOutputFormat::H5 => format!("{basename}{channel_name}.h5"),
                });
                metadata_by_position.insert(position_name.clone(), metadata.clone());
                positions.push(ImportPositionPlan {
                    position_name: position_name.clone(),
                    position_dir,
                    images_dir: images_dir.clone(),
                    metadata_csv_path: images_dir.join(format!("{basename}metadata.csv")),
                    metadata_xml_path: images_dir.join(format!("{basename}metadataXML.txt")),
                    basename,
                    channels: vec![ImportChannelPlan {
                        source_path: source.path.clone(),
                        destination_path,
                        channel_name,
                        source_series_index: None,
                        source_channel_index: 0,
                        backend: resolve_backend(source.detected_kind, config.backend),
                    }],
                });
            }
        }
        ImportLayoutKind::FilePerChannel => {
            let position_name = format!("Position_{start_index}");
            let position_dir = config.destination_experiment_dir.join(&position_name);
            let images_dir = position_dir.join("Images");
            let basename = build_group_basename(discovered_sources, selection.add_image_name);
            let representative = metadata_drafts
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("At least one metadata draft is required"))?;
            metadata_by_position.insert(position_name.clone(), representative);
            let mut channels = Vec::new();
            for (source, metadata) in discovered_sources.iter().zip(metadata_drafts) {
                let channel_name = metadata
                    .channel_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| infer_channel_name(&source.path));
                let save_channel = selection.save_channels.is_empty()
                    || selection
                        .save_channels
                        .get(channels.len())
                        .copied()
                        .unwrap_or(true);
                if !save_channel {
                    warnings.push(format!("Skipping channel {} by selection", channel_name));
                    continue;
                }
                channels.push(ImportChannelPlan {
                    source_path: source.path.clone(),
                    destination_path: images_dir.join(match selection.output_format {
                        ImportOutputFormat::Tiff => format!("{basename}{channel_name}.tif"),
                        ImportOutputFormat::H5 => format!("{basename}{channel_name}.h5"),
                    }),
                    channel_name,
                    source_series_index: None,
                    source_channel_index: 0,
                    backend: resolve_backend(source.detected_kind, config.backend),
                });
            }
            positions.push(ImportPositionPlan {
                position_name,
                position_dir,
                images_dir: images_dir.clone(),
                metadata_csv_path: images_dir.join(format!("{basename}metadata.csv")),
                metadata_xml_path: images_dir.join(format!("{basename}metadataXML.txt")),
                basename,
                channels,
            });
        }
        ImportLayoutKind::SingleFileMultiPosition => {
            let source = discovered_sources
                .first()
                .ok_or_else(|| anyhow!("A single source is required"))?;
            let metadata = metadata_drafts
                .first()
                .ok_or_else(|| anyhow!("Metadata for the source is required"))?;
            if metadata.size_s <= 1
                && resolve_backend(source.detected_kind, config.backend)
                    != ImportReaderBackend::BioFormatsJvmBridge
            {
                warnings.push("Source layout is configured as single-file multi-position, but native probing only found one series. Import will create one position.".to_string());
            }
            let selected = normalized_selected_positions(selection, metadata.size_s.max(1));
            for (offset, series_index) in selected.into_iter().enumerate() {
                let position_name = format!("Position_{}", start_index + offset);
                let basename = build_basename(&source.path, series_index, selection.add_image_name);
                let position_dir = config.destination_experiment_dir.join(&position_name);
                let images_dir = position_dir.join("Images");
                metadata_by_position.insert(position_name.clone(), metadata.clone());
                let mut channels = Vec::new();
                for (channel_index, channel_name) in metadata.channel_names.iter().enumerate() {
                    let save_channel = selection.save_channels.is_empty()
                        || selection
                            .save_channels
                            .get(channel_index)
                            .copied()
                            .unwrap_or(true);
                    if !save_channel {
                        continue;
                    }
                    channels.push(ImportChannelPlan {
                        source_path: source.path.clone(),
                        destination_path: images_dir.join(match selection.output_format {
                            ImportOutputFormat::Tiff => format!("{basename}{channel_name}.tif"),
                            ImportOutputFormat::H5 => format!("{basename}{channel_name}.h5"),
                        }),
                        channel_name: channel_name.clone(),
                        source_series_index: Some(series_index),
                        source_channel_index: channel_index,
                        backend: resolve_backend(source.detected_kind, config.backend),
                    });
                }
                positions.push(ImportPositionPlan {
                    position_name,
                    position_dir,
                    images_dir: images_dir.clone(),
                    metadata_csv_path: images_dir.join(format!("{basename}metadata.csv")),
                    metadata_xml_path: images_dir.join(format!("{basename}metadataXML.txt")),
                    basename,
                    channels,
                });
            }
        }
        ImportLayoutKind::CustomMapping => {
            bail!("Custom mapping is not implemented yet in the Rust importer");
        }
    }

    Ok(ImportPlan {
        experiment_dir: config.destination_experiment_dir.clone(),
        positions,
        metadata_by_position,
        selection: selection.clone(),
        conflict_mode: config.conflict_mode,
        warnings,
    })
}

pub fn execute_import_plan(plan: &ImportPlan) -> Result<ImportExecutionReport> {
    fs::create_dir_all(&plan.experiment_dir)
        .with_context(|| format!("Failed to create {}", plan.experiment_dir.display()))?;
    let mut created_positions = Vec::new();
    let mut written_files = Vec::new();
    let skipped_files = Vec::new();

    for position in &plan.positions {
        if plan.conflict_mode == ImportConflictMode::OverwritePositionFiles
            && position.images_dir.exists()
        {
            fs::remove_dir_all(&position.images_dir)
                .with_context(|| format!("Failed to remove {}", position.images_dir.display()))?;
        }
        if !position.position_dir.exists() {
            created_positions.push(position.position_dir.clone());
        }
        fs::create_dir_all(&position.images_dir)
            .with_context(|| format!("Failed to create {}", position.images_dir.display()))?;
        let metadata = plan
            .metadata_by_position
            .get(&position.position_name)
            .ok_or_else(|| anyhow!("Missing metadata for {}", position.position_name))?;
        for channel in &position.channels {
            match channel.backend {
                ImportReaderBackend::Native => write_native_channel(
                    channel,
                    metadata,
                    plan.selection.time_range,
                    plan.selection.output_format,
                )?,
                ImportReaderBackend::BioFormatsJvmBridge => {
                    let output_path = run_bioformats_export(BioFormatsExportRequest {
                        path: channel.source_path.clone(),
                        output_path: channel.destination_path.clone(),
                        source_series_index: channel.source_series_index,
                        source_channel_index: channel.source_channel_index,
                        time_range: plan.selection.time_range,
                    })?;
                    written_files.push(output_path);
                    continue;
                }
                ImportReaderBackend::Auto => unreachable!(),
            }
            written_files.push(channel.destination_path.clone());
        }
        write_metadata_xml_atomic(&position.metadata_xml_path, &metadata.metadata_xml)?;
        write_metadata_csv_atomic(
            &position.metadata_csv_path,
            &position.basename,
            metadata,
            &position
                .channels
                .iter()
                .map(|channel| channel.channel_name.clone())
                .collect::<Vec<_>>(),
        )?;
        written_files.push(position.metadata_xml_path.clone());
        written_files.push(position.metadata_csv_path.clone());
    }

    Ok(ImportExecutionReport {
        experiment_dir: plan.experiment_dir.clone(),
        created_positions,
        written_files,
        skipped_files,
        warnings: plan.warnings.clone(),
    })
}

fn resolve_backend(kind: ImportSourceKind, backend: ImportReaderBackend) -> ImportReaderBackend {
    match backend {
        ImportReaderBackend::Auto => match kind {
            ImportSourceKind::VendorMicroscopy => ImportReaderBackend::BioFormatsJvmBridge,
            _ => ImportReaderBackend::Native,
        },
        other => other,
    }
}

fn native_probe(path: &Path) -> Result<ImportMetadataDraft> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let image_name = path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("image")
        .to_string();
    let channel_name = infer_channel_name(path);
    let (size_t, size_z) = if extension == "npz" || extension == "h5" {
        let shape = inspect_image_volume(path, None, None).or_else(|_| {
            inspect_image_stack(path).map(|stack| crate::image_io::VolumeShape {
                size_t: stack.frames,
                size_z: 1,
                height: stack.height,
                width: stack.width,
            })
        })?;
        (shape.size_t.max(1), shape.size_z.max(1))
    } else {
        let shape = inspect_image_volume(path, None, None)?;
        (shape.size_t.max(1), shape.size_z.max(1))
    };
    Ok(ImportMetadataDraft {
        lens_na: 1.4,
        size_t,
        size_z,
        size_c: 1,
        size_s: 1,
        time_increment: 1.0,
        time_increment_unit: "s".to_string(),
        physical_size_x: 1.0,
        physical_size_y: 1.0,
        physical_size_z: 1.0,
        physical_size_unit: "um".to_string(),
        channel_names: vec![channel_name],
        emission_wavelengths: vec![0.0],
        image_name,
        metadata_xml: String::new(),
    })
}

fn native_sample_planes(path: &Path) -> Result<ImportSamplePlaneSet> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension == "npz" || extension == "h5" {
        if let Ok((pixels, shape)) = load_image_volume_as_f32(path, None, None) {
            return Ok(ImportSamplePlaneSet {
                width: shape.width,
                height: shape.height,
                frames: shape.size_t,
                size_z: shape.size_z,
                channel_names: vec![infer_channel_name(path)],
                pixels,
            });
        }
    }
    let (pixels, shape) = load_image_stack_as_f32(path)?;
    Ok(ImportSamplePlaneSet {
        width: shape.width,
        height: shape.height,
        frames: shape.frames,
        size_z: 1,
        channel_names: vec![infer_channel_name(path)],
        pixels,
    })
}

fn normalized_selected_positions(selection: &ImportSelection, count: usize) -> Vec<usize> {
    if selection.selected_positions.is_empty()
        || selection
            .selected_positions
            .iter()
            .any(|value| value == "All Positions")
    {
        return (0..count).collect();
    }
    selection
        .selected_positions
        .iter()
        .filter_map(|value| {
            value
                .strip_prefix("Position_")
                .and_then(|number| number.parse::<usize>().ok())
                .map(|value| value.saturating_sub(1))
        })
        .collect()
}

fn next_position_index(experiment_dir: &Path, mode: ImportConflictMode) -> Result<usize> {
    if mode != ImportConflictMode::CreateNewPositions {
        return Ok(1);
    }
    if !experiment_dir.exists() {
        return Ok(1);
    }
    let mut max_index = 0usize;
    for entry in fs::read_dir(experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if let Some(number) = name.strip_prefix("Position_") {
            if let Ok(value) = number.parse::<usize>() {
                max_index = max_index.max(value);
            }
        }
    }
    Ok(max_index + 1)
}

fn infer_channel_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("channel");
    stem.rsplit_once('_')
        .map(|(_, suffix)| suffix.to_string())
        .unwrap_or_else(|| stem.to_string())
}

fn build_group_basename(sources: &[ImportSourceEntry], add_image_name: bool) -> String {
    if !add_image_name {
        return "imported_".to_string();
    }
    let stems = sources
        .iter()
        .filter_map(|entry| entry.path.file_stem().and_then(OsStr::to_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let prefix = longest_common_prefix_all(&stems);
    if prefix.is_empty() {
        "imported_".to_string()
    } else if prefix.ends_with('_') {
        prefix
    } else {
        format!("{prefix}_")
    }
}

fn build_basename(path: &Path, series_index: usize, add_image_name: bool) -> String {
    if !add_image_name {
        return format!("s{:02}_", series_index + 1);
    }
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("imported");
    format!("{stem}_s{:02}_", series_index + 1)
}

fn longest_common_prefix_all(items: &[String]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut prefix = first.clone();
    for item in items.iter().skip(1) {
        prefix = prefix
            .chars()
            .zip(item.chars())
            .take_while(|(left, right)| left == right)
            .map(|(value, _)| value)
            .collect();
        if prefix.is_empty() {
            break;
        }
    }
    prefix.trim_matches('_').to_string()
}

fn write_native_channel(
    channel: &ImportChannelPlan,
    metadata: &ImportMetadataDraft,
    time_range: Option<(usize, usize)>,
    output_format: ImportOutputFormat,
) -> Result<()> {
    match output_format {
        ImportOutputFormat::Tiff => {
            let (pixels, frames, size_z, height, width) = match load_image_volume_as_f32(
                &channel.source_path,
                Some(metadata.size_t),
                Some(metadata.size_z),
            ) {
                Ok((pixels, shape)) => (
                    pixels,
                    shape.size_t,
                    shape.size_z,
                    shape.height,
                    shape.width,
                ),
                Err(_) => {
                    let (pixels, shape) = load_image_stack_as_f32(&channel.source_path)?;
                    (pixels, shape.frames, 1, shape.height, shape.width)
                }
            };
            let (start_frame, end_frame) = normalize_time_range(time_range, frames)?;
            write_tiff_stack_atomic(
                &channel.destination_path,
                &pixels,
                start_frame,
                end_frame,
                size_z,
                height,
                width,
            )
        }
        ImportOutputFormat::H5 => {
            if channel
                .source_path
                .extension()
                .and_then(OsStr::to_str)
                .map(|ext| ext.eq_ignore_ascii_case("h5"))
                .unwrap_or(false)
                && time_range.is_none()
            {
                atomic_copy(&channel.source_path, &channel.destination_path)
            } else {
                bail!(
                    "H5 export is only supported when the source is already an H5 stack and no time-range crop is requested"
                );
            }
        }
    }
}

fn normalize_time_range(
    time_range: Option<(usize, usize)>,
    frame_count: usize,
) -> Result<(usize, usize)> {
    let default = (0usize, frame_count);
    let (start, end) = time_range.unwrap_or(default);
    if start >= end || end > frame_count {
        bail!(
            "Invalid time range {:?} for {} frame(s)",
            time_range,
            frame_count
        );
    }
    Ok((start, end))
}

fn write_tiff_stack_atomic(
    path: &Path,
    pixels: &[f32],
    start_frame: usize,
    end_frame: usize,
    size_z: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    let file = fs::File::create(&tmp_path)
        .with_context(|| format!("Failed to create {}", tmp_path.display()))?;
    let mut encoder = TiffEncoder::new(file)?;
    let plane_len = height * width;
    for frame_index in start_frame..end_frame {
        for z_index in 0..size_z.max(1) {
            let plane_index = frame_index * size_z.max(1) + z_index;
            let offset = plane_index * plane_len;
            let plane = &pixels[offset..offset + plane_len];
            encoder.write_image::<colortype::Gray32Float>(width as u32, height as u32, plane)?;
        }
    }
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "Failed to move {} to {}",
            tmp_path.display(),
            path.display()
        )
    })
}

fn write_metadata_xml_atomic(path: &Path, metadata_xml: &str) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, metadata_xml)
        .with_context(|| format!("Failed to write {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "Failed to move {} to {}",
            tmp_path.display(),
            path.display()
        )
    })
}

fn write_metadata_csv_atomic(
    path: &Path,
    basename: &str,
    metadata: &ImportMetadataDraft,
    channel_names: &[String],
) -> Result<()> {
    let mut rows = BTreeMap::new();
    rows.insert("basename".to_string(), basename.to_string());
    rows.insert("SizeT".to_string(), metadata.size_t.to_string());
    rows.insert("SizeZ".to_string(), metadata.size_z.to_string());
    rows.insert("SizeC".to_string(), channel_names.len().to_string());
    rows.insert("SizeS".to_string(), metadata.size_s.to_string());
    rows.insert(
        "TimeIncrement".to_string(),
        metadata.time_increment.to_string(),
    );
    rows.insert(
        "TimeIncrementUnit".to_string(),
        metadata.time_increment_unit.clone(),
    );
    rows.insert(
        "PhysicalSizeX".to_string(),
        metadata.physical_size_x.to_string(),
    );
    rows.insert(
        "PhysicalSizeY".to_string(),
        metadata.physical_size_y.to_string(),
    );
    rows.insert(
        "PhysicalSizeZ".to_string(),
        metadata.physical_size_z.to_string(),
    );
    rows.insert(
        "PhysicalSizeUnit".to_string(),
        metadata.physical_size_unit.clone(),
    );
    rows.insert("ImageName".to_string(), metadata.image_name.clone());
    rows.insert("LensNA".to_string(), metadata.lens_na.to_string());
    for (index, name) in channel_names.iter().enumerate() {
        rows.insert(format!("channel_{index}_name"), name.clone());
    }
    let tmp_path = path.with_extension("tmp");
    let mut writer = csv::Writer::from_path(&tmp_path)
        .with_context(|| format!("Failed to create {}", tmp_path.display()))?;
    writer.write_record(["Description", "values"])?;
    for (key, value) in rows {
        writer.write_record([key, value])?;
    }
    writer.flush()?;
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "Failed to move {} to {}",
            tmp_path.display(),
            path.display()
        )
    })
}

fn atomic_copy(source: &Path, destination: &Path) -> Result<()> {
    let tmp_path = destination.with_extension("tmp");
    fs::copy(source, &tmp_path).with_context(|| {
        format!(
            "Failed to copy {} to {}",
            source.display(),
            tmp_path.display()
        )
    })?;
    fs::rename(&tmp_path, destination).with_context(|| {
        format!(
            "Failed to move {} to {}",
            tmp_path.display(),
            destination.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn discovers_supported_sources_in_dir_or_file() -> Result<()> {
        let dir = tempdir()?;
        fs::write(dir.path().join("a.npz"), b"test")?;
        fs::write(dir.path().join("b.h5"), b"test")?;
        fs::write(dir.path().join("c.czi"), b"test")?;
        let discovered = discover_import_sources(dir.path())?;
        assert_eq!(discovered.len(), 3);

        let single = discover_import_sources(dir.path().join("c.czi"))?;
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].detected_kind, ImportSourceKind::VendorMicroscopy);
        Ok(())
    }

    #[test]
    fn classifies_file_per_channel_layout() -> Result<()> {
        let sources = vec![
            ImportSourceEntry {
                path: PathBuf::from("exp_phase.tif"),
                detected_kind: ImportSourceKind::Tiff,
                backend_used: None,
                series_count: None,
            },
            ImportSourceEntry {
                path: PathBuf::from("exp_gfp.tif"),
                detected_kind: ImportSourceKind::Tiff,
                backend_used: None,
                series_count: None,
            },
        ];
        let layout = classify_import_layout(&sources, None)?;
        assert_eq!(layout, ImportLayoutKind::FilePerChannel);
        Ok(())
    }

    #[test]
    fn builds_file_per_position_plan() -> Result<()> {
        let config = ImportExecutionConfig {
            layout_kind: ImportLayoutKind::FilePerPosition,
            backend: ImportReaderBackend::Native,
            sources: vec![PathBuf::from("a.tif"), PathBuf::from("b.tif")],
            destination_experiment_dir: PathBuf::from("/tmp/demo"),
            conflict_mode: ImportConflictMode::CreateNewPositions,
            metadata_policy: MetadataReusePolicy::ConfirmEverySource,
        };
        let sources = config
            .sources
            .iter()
            .map(|path| ImportSourceEntry {
                path: path.clone(),
                detected_kind: ImportSourceKind::Tiff,
                backend_used: None,
                series_count: None,
            })
            .collect::<Vec<_>>();
        let metadata = sources
            .iter()
            .map(|source| ImportMetadataDraft {
                lens_na: 1.4,
                size_t: 1,
                size_z: 1,
                size_c: 1,
                size_s: 1,
                time_increment: 1.0,
                time_increment_unit: "s".to_string(),
                physical_size_x: 1.0,
                physical_size_y: 1.0,
                physical_size_z: 1.0,
                physical_size_unit: "um".to_string(),
                channel_names: vec![infer_channel_name(&source.path)],
                emission_wavelengths: vec![0.0],
                image_name: "demo".to_string(),
                metadata_xml: String::new(),
            })
            .collect::<Vec<_>>();
        let plan = build_import_plan(&config, &sources, &metadata, &ImportSelection::default())?;
        assert_eq!(plan.positions.len(), 2);
        Ok(())
    }

    #[test]
    fn executes_native_tiff_import() -> Result<()> {
        let temp = tempdir()?;
        let source = temp.path().join("phase.tif");
        let file = fs::File::create(&source)?;
        let mut encoder = TiffEncoder::new(file)?;
        let pixels = vec![1u16, 2, 3, 4];
        encoder.write_image::<colortype::Gray16>(2, 2, &pixels)?;

        let config = ImportExecutionConfig {
            layout_kind: ImportLayoutKind::FilePerPosition,
            backend: ImportReaderBackend::Native,
            sources: vec![source.clone()],
            destination_experiment_dir: temp.path().join("experiment"),
            conflict_mode: ImportConflictMode::CreateNewPositions,
            metadata_policy: MetadataReusePolicy::ConfirmEverySource,
        };
        let discovered = discover_import_sources(&source)?;
        let metadata = vec![probe_import_source(&source, ImportReaderBackend::Native)?];
        let plan = build_import_plan(&config, &discovered, &metadata, &ImportSelection::default())?;
        let report = execute_import_plan(&plan)?;
        assert!(!report.written_files.is_empty());
        assert!(report.written_files.iter().any(|path| path
            .file_name()
            .and_then(OsStr::to_str)
            .map(|value| value.ends_with("metadata.csv"))
            .unwrap_or(false)));
        Ok(())
    }
}
