use crate::layout::PositionSpec;
use anyhow::{bail, Context, Result};
use ndarray::Array2;
use ndarray_npy::NpzWriter;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

pub fn load_tiff_as_f32(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder =
        Decoder::new(file).with_context(|| format!("Failed to decode TIFF {}", path.display()))?;
    let dimensions = decoder
        .dimensions()
        .with_context(|| format!("Failed to read TIFF dimensions from {}", path.display()))?;
    let color_type = decoder
        .colortype()
        .with_context(|| format!("Failed to read TIFF color type from {}", path.display()))?;

    match color_type {
        ColorType::Gray(_) => {}
        _ => {
            bail!(
                "Unsupported TIFF color type {:?} in {}. Phase 1 expects grayscale 2D TIFF inputs.",
                color_type,
                path.display()
            );
        }
    }

    let pixels = match decoder
        .read_image()
        .with_context(|| format!("Failed to read TIFF pixels from {}", path.display()))?
    {
        DecodingResult::U8(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U16(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U32(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U64(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I8(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I16(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I32(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I64(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::F32(values) => values,
        DecodingResult::F64(values) => values.into_iter().map(|v| v as f32).collect(),
        other => bail!(
            "Unsupported TIFF pixel format {:?} in {}",
            other,
            path.display()
        ),
    };

    if decoder.more_images() {
        bail!(
            "Unsupported multi-page TIFF in {}. Phase 1 only supports single-frame 2D TIFF inputs.",
            path.display()
        );
    }

    let (width, height) = (dimensions.0 as usize, dimensions.1 as usize);
    if pixels.len() != width * height {
        bail!(
            "Unexpected pixel count in {}: got {}, expected {}",
            path.display(),
            pixels.len(),
            width * height
        );
    }

    Ok((pixels, height, width))
}

pub fn write_mask_npz(path: &Path, masks: &[u32], height: usize, width: usize) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;

    let array = Array2::from_shape_vec((height, width), masks.to_vec())
        .with_context(|| format!("Failed to shape mask array for {}", path.display()))?;
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = NpzWriter::new_compressed(file);
    writer
        .add_array("arr_0", &array)
        .with_context(|| format!("Failed to write NPZ array to {}", path.display()))?;
    writer
        .finish()
        .with_context(|| format!("Failed to finish NPZ {}", path.display()))?;
    Ok(())
}

pub fn ensure_metadata_file(spec: &PositionSpec, height: usize, width: usize) -> Result<()> {
    if spec.metadata_path.is_some() {
        return Ok(());
    }

    let metadata_name = format!("{}metadata.csv", spec.basename);
    let metadata_path = spec.images_dir.join(metadata_name);
    let mut file = File::create(&metadata_path)
        .with_context(|| format!("Failed to create {}", metadata_path.display()))?;
    writeln!(file, "Description,values")?;
    writeln!(file, "basename,{}", spec.basename)?;
    writeln!(file, "SizeT,1")?;
    writeln!(file, "SizeZ,1")?;
    writeln!(file, "SizeY,{height}")?;
    writeln!(file, "SizeX,{width}")?;
    writeln!(file, "channel_0_name,{}", spec.phase_channel)?;
    writeln!(file, "channel_1_name,{}", spec.fluo_channel)?;
    Ok(())
}
