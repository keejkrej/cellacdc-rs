use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3};
use ndarray_npy::NpzWriter;
use std::fs::{self, File};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackShape {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

pub fn inspect_tiff_stack(path: &Path) -> Result<StackShape> {
    let (_, shape) = load_tiff_stack_as_f32(path)?;
    Ok(shape)
}

pub fn load_tiff_stack_as_f32(path: &Path) -> Result<(Vec<f32>, StackShape)> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder =
        Decoder::new(file).with_context(|| format!("Failed to decode TIFF {}", path.display()))?;

    let mut pixels = Vec::new();
    let mut frames = 0usize;
    let mut expected_shape = None;

    loop {
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
                    "Unsupported TIFF color type {:?} in {}. This phase expects grayscale 2D TIFF inputs.",
                    color_type,
                    path.display()
                );
            }
        }

        let (width, height) = (dimensions.0 as usize, dimensions.1 as usize);
        let page_pixels = match decoder
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

        if page_pixels.len() != width * height {
            bail!(
                "Unexpected pixel count in {}: got {}, expected {}",
                path.display(),
                page_pixels.len(),
                width * height
            );
        }

        let page_shape = StackShape {
            frames: 1,
            height,
            width,
        };
        if let Some(expected) = expected_shape {
            if expected != page_shape {
                bail!(
                    "TIFF pages in {} do not share the same dimensions",
                    path.display()
                );
            }
        } else {
            expected_shape = Some(page_shape);
        }

        pixels.extend(page_pixels);
        frames += 1;

        if !decoder.more_images() {
            break;
        }

        decoder
            .next_image()
            .with_context(|| format!("Failed to advance TIFF pages in {}", path.display()))?;
    }

    let shape = StackShape {
        frames,
        height: expected_shape.map(|shape| shape.height).unwrap_or(0),
        width: expected_shape.map(|shape| shape.width).unwrap_or(0),
    };
    Ok((pixels, shape))
}

pub fn write_mask_npz(
    path: &Path,
    masks: &[u32],
    frames: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;

    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = NpzWriter::new_compressed(file);

    if frames == 1 {
        let array = Array2::from_shape_vec((height, width), masks.to_vec())
            .with_context(|| format!("Failed to shape mask array for {}", path.display()))?;
        writer
            .add_array("arr_0", &array)
            .with_context(|| format!("Failed to write NPZ array to {}", path.display()))?;
    } else {
        let array = Array3::from_shape_vec((frames, height, width), masks.to_vec())
            .with_context(|| format!("Failed to shape mask stack for {}", path.display()))?;
        writer
            .add_array("arr_0", &array)
            .with_context(|| format!("Failed to write NPZ array to {}", path.display()))?;
    }

    writer
        .finish()
        .with_context(|| format!("Failed to finish NPZ {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn loads_multi_page_tiff_stack() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("stack.tif");
        write_test_stack(&path, &[1, 5])?;

        let (pixels, shape) = load_tiff_stack_as_f32(&path)?;
        assert_eq!(
            shape,
            StackShape {
                frames: 2,
                height: 2,
                width: 3
            }
        );
        assert_eq!(pixels.len(), 12);
        assert_eq!(pixels[0], 1.0);
        assert_eq!(pixels[6], 5.0);
        Ok(())
    }

    fn write_test_stack(path: &Path, frame_values: &[u16]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frame_values {
            let data = vec![*value; 6];
            encoder.write_image::<colortype::Gray16>(3, 2, &data)?;
        }
        Ok(())
    }
}
