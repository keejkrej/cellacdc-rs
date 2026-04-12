#[cfg(feature = "bioformats-import")]
use anyhow::{anyhow, bail, Context, Result};
#[cfg(feature = "bioformats-import")]
use bioformats_rs::{ChannelSeparator, FormatReader, ImageMetadata, ImageReader, PixelType};
#[cfg(feature = "bioformats-import")]
use csv::Writer;
#[cfg(feature = "bioformats-import")]
use std::collections::BTreeSet;
#[cfg(feature = "bioformats-import")]
use std::fs::{self, File};
#[cfg(feature = "bioformats-import")]
use std::path::{Path, PathBuf};
#[cfg(feature = "bioformats-import")]
use tiff::encoder::{colortype, TiffEncoder};

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct RawSeriesSummary {
    pub source_path: PathBuf,
    pub series_index: usize,
    pub series_name: String,
    pub size_t: usize,
    pub size_z: usize,
    pub size_c: usize,
    pub size_x: usize,
    pub size_y: usize,
    pub channel_names: Vec<String>,
    pub emission_wavelengths_nm: Vec<Option<f64>>,
    pub time_increment_seconds: Option<f64>,
    pub physical_size_x_um: Option<f64>,
    pub physical_size_y_um: Option<f64>,
    pub physical_size_z_um: Option<f64>,
    pub objective_na: Option<f64>,
    pub used_files: Vec<PathBuf>,
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct RawImportProbe {
    pub source_path: PathBuf,
    pub series: Vec<RawSeriesSummary>,
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawImportOutputFormat {
    Tiff,
}

#[cfg(feature = "bioformats-import")]
impl Default for RawImportOutputFormat {
    fn default() -> Self {
        Self::Tiff
    }
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct RawImportSelection {
    pub source_path: PathBuf,
    pub series_indices: Option<Vec<usize>>,
    pub channel_indices: Option<Vec<usize>>,
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct RawImportExperimentConfig {
    pub target_dir: PathBuf,
    pub selections: Vec<RawImportSelection>,
    pub start_position_index: usize,
    pub output_format: RawImportOutputFormat,
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedRawPosition {
    pub source_path: PathBuf,
    pub series_index: usize,
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub basename: String,
    pub metadata_path: PathBuf,
    pub imported_files: Vec<PathBuf>,
}

#[cfg(feature = "bioformats-import")]
#[derive(Debug, Clone, PartialEq)]
pub struct RawImportedExperiment {
    pub experiment_dir: PathBuf,
    pub positions: Vec<ImportedRawPosition>,
}

#[cfg(feature = "bioformats-import")]
pub fn probe_raw_import_source(path: impl AsRef<Path>) -> Result<RawImportProbe> {
    let path = path.as_ref();
    let mut reader = ImageReader::open(path)
        .with_context(|| format!("Failed to open raw microscopy file {}", path.display()))?;
    let series_count = reader.series_count();
    let mut series = Vec::with_capacity(series_count.max(1));
    for series_index in 0..series_count.max(1) {
        reader.set_series(series_index).with_context(|| {
            format!(
                "Failed to select series {series_index} in {}",
                path.display()
            )
        })?;
        let metadata = reader.metadata().clone();
        series.push(build_series_summary(
            path,
            series_index,
            &metadata,
            reader.used_files(),
        ));
    }
    Ok(RawImportProbe {
        source_path: path.to_path_buf(),
        series,
    })
}

#[cfg(feature = "bioformats-import")]
pub fn import_raw_experiment(config: RawImportExperimentConfig) -> Result<RawImportedExperiment> {
    if config.selections.is_empty() {
        bail!("No raw import sources were selected");
    }
    if config.start_position_index == 0 {
        bail!("Raw import position numbering starts at 1");
    }

    fs::create_dir_all(&config.target_dir).with_context(|| {
        format!(
            "Failed to create target experiment directory {}",
            config.target_dir.display()
        )
    })?;

    let mut next_position_index = config.start_position_index;
    let mut positions = Vec::new();

    for selection in &config.selections {
        let probe = probe_raw_import_source(&selection.source_path)?;
        let selected_series = normalize_selected_indices(
            selection.series_indices.as_deref(),
            probe.series.len(),
            "series",
            &selection.source_path,
        )?;

        for series_index in selected_series {
            let summary = probe.series.get(series_index).ok_or_else(|| {
                anyhow!(
                    "Series {series_index} is out of bounds for {}",
                    selection.source_path.display()
                )
            })?;
            let channel_indices = normalize_selected_indices(
                selection.channel_indices.as_deref(),
                summary.size_c,
                "channel",
                &selection.source_path,
            )?;
            let imported = import_raw_series(
                &selection.source_path,
                series_index,
                &channel_indices,
                &config.target_dir,
                next_position_index,
                config.output_format,
            )?;
            positions.push(imported);
            next_position_index += 1;
        }
    }

    Ok(RawImportedExperiment {
        experiment_dir: config.target_dir,
        positions,
    })
}

#[cfg(feature = "bioformats-import")]
fn import_raw_series(
    source_path: &Path,
    series_index: usize,
    channel_indices: &[usize],
    target_dir: &Path,
    position_index: usize,
    output_format: RawImportOutputFormat,
) -> Result<ImportedRawPosition> {
    let mut reader = open_series_reader(source_path, series_index)?;
    let metadata = reader.metadata().clone();
    let summary = build_series_summary(source_path, series_index, &metadata, reader.used_files());

    let position_dir = target_dir.join(format!("Position_{position_index}"));
    let images_dir = position_dir.join("Images");
    fs::create_dir_all(&images_dir)
        .with_context(|| format!("Failed to create {}", images_dir.display()))?;

    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("imported");
    let basename = format!(
        "{}_s{:02}_",
        sanitize_filename_component(stem),
        position_index
    );

    let mut imported_files = Vec::new();
    for &channel_index in channel_indices {
        let channel_name = summary
            .channel_names
            .get(channel_index)
            .cloned()
            .unwrap_or_else(|| format!("channel_{channel_index}"));
        let channel_filename = match output_format {
            RawImportOutputFormat::Tiff => {
                format!(
                    "{}{}.tif",
                    basename,
                    sanitize_filename_component(&channel_name)
                )
            }
        };
        let channel_path = images_dir.join(channel_filename);
        export_channel_to_tiff(
            reader.as_mut(),
            &metadata,
            channel_index,
            &channel_path,
            summary.size_t,
            summary.size_z,
        )?;
        imported_files.push(channel_path);
    }

    let metadata_path = images_dir.join(format!("{basename}metadata.csv"));
    write_metadata_csv(&metadata_path, &basename, &summary, channel_indices)?;

    Ok(ImportedRawPosition {
        source_path: source_path.to_path_buf(),
        series_index,
        position_dir,
        images_dir,
        basename,
        metadata_path,
        imported_files,
    })
}

#[cfg(feature = "bioformats-import")]
fn open_series_reader(source_path: &Path, series_index: usize) -> Result<Box<dyn FormatReader>> {
    let mut reader = ImageReader::open(source_path).with_context(|| {
        format!(
            "Failed to open raw microscopy file {}",
            source_path.display()
        )
    })?;
    reader.set_series(series_index).with_context(|| {
        format!(
            "Failed to select series {series_index} in {}",
            source_path.display()
        )
    })?;

    if reader.metadata().is_rgb && !reader.metadata().is_indexed {
        let mut separated = ChannelSeparator::new(reader);
        separated.set_series(series_index).with_context(|| {
            format!(
                "Failed to initialize channel-separated series {series_index} in {}",
                source_path.display()
            )
        })?;
        Ok(Box::new(separated))
    } else {
        Ok(Box::new(reader))
    }
}

#[cfg(feature = "bioformats-import")]
fn build_series_summary(
    source_path: &Path,
    series_index: usize,
    metadata: &ImageMetadata,
    used_files: Vec<PathBuf>,
) -> RawSeriesSummary {
    let channel_names = if metadata.channel_metadata.is_empty() {
        (0..metadata.logical_channel_count() as usize)
            .map(|index| format!("channel_{index}"))
            .collect()
    } else {
        (0..metadata.logical_channel_count() as usize)
            .map(|index| {
                metadata
                    .channel_metadata
                    .get(index)
                    .and_then(|channel| channel.name.as_ref())
                    .map(|name| sanitize_filename_component(name))
                    .filter(|name| !name.is_empty())
                    .unwrap_or_else(|| format!("channel_{index}"))
            })
            .collect()
    };

    let emission_wavelengths_nm = (0..metadata.logical_channel_count() as usize)
        .map(|index| {
            metadata
                .channel_metadata
                .get(index)
                .and_then(|channel| channel.emission_wavelength_nm)
        })
        .collect();

    RawSeriesSummary {
        source_path: source_path.to_path_buf(),
        series_index,
        series_name: format!("Series {}", series_index + 1),
        size_t: metadata.size_t as usize,
        size_z: metadata.size_z as usize,
        size_c: metadata.logical_channel_count() as usize,
        size_x: metadata.size_x as usize,
        size_y: metadata.size_y as usize,
        channel_names,
        emission_wavelengths_nm,
        time_increment_seconds: metadata.time_increment_seconds,
        physical_size_x_um: metadata.physical_size_x_um,
        physical_size_y_um: metadata.physical_size_y_um,
        physical_size_z_um: metadata.physical_size_z_um,
        objective_na: metadata.objective_na,
        used_files,
    }
}

#[cfg(feature = "bioformats-import")]
fn normalize_selected_indices(
    selected: Option<&[usize]>,
    upper_bound: usize,
    label: &str,
    source_path: &Path,
) -> Result<Vec<usize>> {
    let mut ordered = BTreeSet::new();
    match selected {
        Some(indices) => {
            for &index in indices {
                if index >= upper_bound {
                    bail!(
                        "Selected {label} index {index} is out of bounds for {}",
                        source_path.display()
                    );
                }
                ordered.insert(index);
            }
        }
        None => {
            ordered.extend(0..upper_bound);
        }
    }
    if ordered.is_empty() {
        bail!(
            "No {label} indices were selected for {}",
            source_path.display()
        );
    }
    Ok(ordered.into_iter().collect())
}

#[cfg(feature = "bioformats-import")]
fn export_channel_to_tiff(
    reader: &mut dyn FormatReader,
    metadata: &ImageMetadata,
    channel_index: usize,
    output_path: &Path,
    size_t: usize,
    size_z: usize,
) -> Result<()> {
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create {}", output_path.display()))?;
    let mut encoder = TiffEncoder::new(file)?;
    let width = metadata.size_x;
    let height = metadata.size_y;

    for t in 0..size_t {
        for z in 0..size_z {
            let plane_index = metadata.get_index(z as u32, channel_index as u32, t as u32);
            let bytes = reader.open_bytes(plane_index).with_context(|| {
                format!(
                    "Failed to read plane z={z}, c={channel_index}, t={t} from {}",
                    output_path.display()
                )
            })?;
            let plane = decode_plane(&bytes, metadata.pixel_type, metadata.is_little_endian)
                .with_context(|| {
                    format!(
                        "Failed to decode plane {plane_index} for {}",
                        output_path.display()
                    )
                })?;
            plane.write_tiff(&mut encoder, width, height)?;
        }
    }

    Ok(())
}

#[cfg(feature = "bioformats-import")]
fn write_metadata_csv(
    path: &Path,
    basename: &str,
    summary: &RawSeriesSummary,
    channel_indices: &[usize],
) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record(["Description", "values"])?;
    writer.write_record(["basename", basename])?;
    writer.write_record(["SizeT", &summary.size_t.to_string()])?;
    writer.write_record(["SizeZ", &summary.size_z.to_string()])?;
    writer.write_record([
        "TimeIncrement",
        &summary.time_increment_seconds.unwrap_or(1.0).to_string(),
    ])?;
    writer.write_record([
        "PhysicalSizeZ",
        &summary.physical_size_z_um.unwrap_or(1.0).to_string(),
    ])?;
    writer.write_record([
        "PhysicalSizeY",
        &summary.physical_size_y_um.unwrap_or(1.0).to_string(),
    ])?;
    writer.write_record([
        "PhysicalSizeX",
        &summary.physical_size_x_um.unwrap_or(1.0).to_string(),
    ])?;
    if let Some(value) = summary.objective_na {
        writer.write_record(["LensNA", &value.to_string()])?;
    }

    for (exported_index, &channel_index) in channel_indices.iter().enumerate() {
        let channel_name = summary
            .channel_names
            .get(channel_index)
            .cloned()
            .unwrap_or_else(|| format!("channel_{channel_index}"));
        writer.write_record([format!("channel_{exported_index}_name"), channel_name])?;
        if let Some(wavelength) = summary
            .emission_wavelengths_nm
            .get(channel_index)
            .and_then(|value| *value)
        {
            writer.write_record([
                format!("channel_{exported_index}_emWavelen"),
                wavelength.to_string(),
            ])?;
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(feature = "bioformats-import")]
fn sanitize_filename_component(value: &str) -> String {
    let replaced = value.replace('.', "_");
    let mut out = String::with_capacity(replaced.len());
    for ch in replaced.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "channel".to_string()
    } else {
        out
    }
}

#[cfg(feature = "bioformats-import")]
enum DecodedPlane {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F32(Vec<f32>),
}

#[cfg(feature = "bioformats-import")]
impl DecodedPlane {
    fn write_tiff(&self, encoder: &mut TiffEncoder<File>, width: u32, height: u32) -> Result<()> {
        match self {
            Self::U8(values) => encoder.write_image::<colortype::Gray8>(width, height, values)?,
            Self::U16(values) => encoder.write_image::<colortype::Gray16>(width, height, values)?,
            Self::U32(values) => encoder.write_image::<colortype::Gray32>(width, height, values)?,
            Self::F32(values) => {
                encoder.write_image::<colortype::Gray32Float>(width, height, values)?
            }
        }
        Ok(())
    }
}

#[cfg(feature = "bioformats-import")]
fn decode_plane(bytes: &[u8], pixel_type: PixelType, little_endian: bool) -> Result<DecodedPlane> {
    match pixel_type {
        PixelType::Uint8 => Ok(DecodedPlane::U8(bytes.to_vec())),
        PixelType::Bit => Ok(DecodedPlane::U8(bytes.to_vec())),
        PixelType::Int8 => Ok(DecodedPlane::U8(
            bytes
                .iter()
                .map(|value| (*value as i8).max(0) as u8)
                .collect(),
        )),
        PixelType::Uint16 => Ok(DecodedPlane::U16(decode_u16_plane(bytes, little_endian)?)),
        PixelType::Int16 => Ok(DecodedPlane::U16(
            decode_u16_plane(bytes, little_endian)?
                .into_iter()
                .map(|value| (value as i16).max(0) as u16)
                .collect(),
        )),
        PixelType::Uint32 => Ok(DecodedPlane::U32(decode_u32_plane(bytes, little_endian)?)),
        PixelType::Int32 => Ok(DecodedPlane::U32(
            decode_u32_plane(bytes, little_endian)?
                .into_iter()
                .map(|value| (value as i32).max(0) as u32)
                .collect(),
        )),
        PixelType::Float32 => Ok(DecodedPlane::F32(decode_f32_plane(bytes, little_endian)?)),
        PixelType::Float64 => Ok(DecodedPlane::F32(
            decode_f64_plane(bytes, little_endian)?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        )),
    }
}

#[cfg(feature = "bioformats-import")]
fn decode_u16_plane(bytes: &[u8], little_endian: bool) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        bail!(
            "u16 plane byte length {} is not divisible by 2",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            if little_endian {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            }
        })
        .collect())
}

#[cfg(feature = "bioformats-import")]
fn decode_u32_plane(bytes: &[u8], little_endian: bool) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        bail!(
            "u32 plane byte length {} is not divisible by 4",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            if little_endian {
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            } else {
                u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            }
        })
        .collect())
}

#[cfg(feature = "bioformats-import")]
fn decode_f32_plane(bytes: &[u8], little_endian: bool) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!(
            "f32 plane byte length {} is not divisible by 4",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            if little_endian {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            } else {
                f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            }
        })
        .collect())
}

#[cfg(feature = "bioformats-import")]
fn decode_f64_plane(bytes: &[u8], little_endian: bool) -> Result<Vec<f64>> {
    if bytes.len() % 8 != 0 {
        bail!(
            "f64 plane byte length {} is not divisible by 8",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            if little_endian {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            } else {
                f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            }
        })
        .collect())
}

#[cfg(all(test, feature = "bioformats-import"))]
mod tests {
    use super::*;
    use crate::layout::resolve_measurement_position;
    use anyhow::Result;
    use std::fs;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn probes_and_imports_raw_ome_tiff_into_cellacdc_layout() -> Result<()> {
        let temp = tempdir()?;
        let source_path = temp.path().join("sample.ome.tif");
        write_test_stack(&source_path, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]], 2, 2)?;

        let probe = probe_raw_import_source(&source_path)?;
        assert_eq!(probe.series.len(), 1);
        assert_eq!(probe.series[0].size_c, 1);
        assert_eq!(probe.series[0].size_t * probe.series[0].size_z, 2);

        let target_dir = temp.path().join("experiment");
        let imported = import_raw_experiment(RawImportExperimentConfig {
            target_dir: target_dir.clone(),
            selections: vec![RawImportSelection {
                source_path: source_path.clone(),
                series_indices: None,
                channel_indices: None,
            }],
            start_position_index: 1,
            output_format: RawImportOutputFormat::Tiff,
        })?;

        assert_eq!(imported.positions.len(), 1);
        let position = &imported.positions[0];
        assert!(position.metadata_path.exists());
        assert_eq!(position.imported_files.len(), 1);

        let spec = resolve_measurement_position(&position.position_dir)?;
        assert_eq!(spec.size_t * spec.size_z, 2);
        assert_eq!(spec.channels.len(), 1);
        assert_eq!(spec.channels[0].name, "channel_0");
        assert_eq!(spec.time_increment, 1.0);
        assert_eq!(
            spec.physical_size_x,
            probe.series[0].physical_size_x_um.unwrap_or(1.0)
        );
        assert_eq!(
            spec.physical_size_y,
            probe.series[0].physical_size_y_um.unwrap_or(1.0)
        );
        Ok(())
    }

    #[test]
    fn sanitizes_filename_components_like_python_importer() {
        assert_eq!(sanitize_filename_component("DAPI.405 nm"), "DAPI_405_nm");
        assert_eq!(sanitize_filename_component("GFP/488"), "GFP_488");
        assert_eq!(sanitize_filename_component("___"), "channel");
    }

    fn write_test_stack(
        path: &Path,
        planes: &[Vec<u16>],
        height: usize,
        width: usize,
    ) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for plane in planes {
            encoder.write_image::<colortype::Gray16>(width as u32, height as u32, plane)?;
        }
        Ok(())
    }
}
