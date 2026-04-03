use anyhow::{bail, Context, Result};
use hdf5_reader::Hdf5File;
use ndarray::{Array2, Array3, ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{NpzReader, NpzWriter};
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

#[derive(Debug, Clone, PartialEq)]
pub struct NamedArrayF32 {
    pub name: String,
    pub values: Vec<f32>,
    pub shape: StackShape,
}

pub fn inspect_image_stack(path: &Path) -> Result<StackShape> {
    let (_, shape) = load_image_stack_as_f32(path)?;
    Ok(shape)
}

pub fn load_image_stack_as_f32(path: &Path) -> Result<(Vec<f32>, StackShape)> {
    match extension(path).as_deref() {
        Some("tif") | Some("tiff") => load_tiff_stack_as_f32(path),
        Some("npz") => load_npz_stack_as_f32(path),
        Some("h5") => load_h5_stack_as_f32(path),
        other => bail!(
            "Unsupported image format {:?} for {}. Supported formats are TIFF, NPZ, and H5.",
            other,
            path.display()
        ),
    }
}

pub fn load_mask_stack_as_u32(path: &Path) -> Result<(Vec<u32>, StackShape)> {
    match extension(path).as_deref() {
        Some("tif") | Some("tiff") => load_tiff_stack_as_u32(path),
        Some("npz") => load_npz_stack_as_u32(path),
        Some("h5") => load_h5_stack_as_u32(path),
        other => bail!(
            "Unsupported mask format {:?} for {}. Supported formats are TIFF, NPZ, and H5.",
            other,
            path.display()
        ),
    }
}

pub fn load_npz_archive_arrays_as_f32(path: &Path) -> Result<Vec<NamedArrayF32>> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;
    let names = npz
        .names()
        .with_context(|| format!("Failed to list NPZ arrays in {}", path.display()))?;
    drop(npz);

    let mut arrays = Vec::with_capacity(names.len());
    for name in names {
        let shape = read_npz_shape(path, &name)?;
        let values = read_npz_pixels(path, &name)?;
        arrays.push(NamedArrayF32 {
            name,
            values,
            shape,
        });
    }
    Ok(arrays)
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn load_tiff_stack_as_f32(path: &Path) -> Result<(Vec<f32>, StackShape)> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder =
        Decoder::new(file).with_context(|| format!("Failed to decode TIFF {}", path.display()))?;

    let mut pixels = Vec::new();
    let mut frames = 0usize;
    let mut expected_shape = None;

    loop {
        let (page_pixels, page_shape) = read_tiff_page_as_f32(&mut decoder, path)?;

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

fn load_tiff_stack_as_u32(path: &Path) -> Result<(Vec<u32>, StackShape)> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder =
        Decoder::new(file).with_context(|| format!("Failed to decode TIFF {}", path.display()))?;

    let mut pixels = Vec::new();
    let mut frames = 0usize;
    let mut expected_shape = None;

    loop {
        let (page_pixels, page_shape) = read_tiff_page_as_u32(&mut decoder, path)?;

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

fn read_tiff_page_as_f32(
    decoder: &mut Decoder<File>,
    path: &Path,
) -> Result<(Vec<f32>, StackShape)> {
    let (width, height) = read_tiff_page_shape(decoder, path)?;
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

    validate_tiff_page_len(path, width, height, page_pixels.len())?;
    Ok((
        page_pixels,
        StackShape {
            frames: 1,
            height,
            width,
        },
    ))
}

fn read_tiff_page_as_u32(
    decoder: &mut Decoder<File>,
    path: &Path,
) -> Result<(Vec<u32>, StackShape)> {
    let (width, height) = read_tiff_page_shape(decoder, path)?;
    let page_pixels = match decoder
        .read_image()
        .with_context(|| format!("Failed to read TIFF pixels from {}", path.display()))?
    {
        DecodingResult::U8(values) => values.into_iter().map(|v| v as u32).collect(),
        DecodingResult::U16(values) => values.into_iter().map(|v| v as u32).collect(),
        DecodingResult::U32(values) => values,
        DecodingResult::U64(values) => values.into_iter().map(|v| v as u32).collect(),
        DecodingResult::I8(values) => values.into_iter().map(|v| v.max(0) as u32).collect(),
        DecodingResult::I16(values) => values.into_iter().map(|v| v.max(0) as u32).collect(),
        DecodingResult::I32(values) => values.into_iter().map(|v| v.max(0) as u32).collect(),
        DecodingResult::I64(values) => values.into_iter().map(|v| v.max(0) as u32).collect(),
        DecodingResult::F32(values) => values.into_iter().map(|v| v.max(0.0) as u32).collect(),
        DecodingResult::F64(values) => values.into_iter().map(|v| v.max(0.0) as u32).collect(),
        other => bail!(
            "Unsupported TIFF mask pixel format {:?} in {}",
            other,
            path.display()
        ),
    };

    validate_tiff_page_len(path, width, height, page_pixels.len())?;
    Ok((
        page_pixels,
        StackShape {
            frames: 1,
            height,
            width,
        },
    ))
}

fn read_tiff_page_shape(decoder: &mut Decoder<File>, path: &Path) -> Result<(usize, usize)> {
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

    Ok((dimensions.0 as usize, dimensions.1 as usize))
}

fn validate_tiff_page_len(
    path: &Path,
    width: usize,
    height: usize,
    actual_len: usize,
) -> Result<()> {
    if actual_len != width * height {
        bail!(
            "Unexpected pixel count in {}: got {}, expected {}",
            path.display(),
            actual_len,
            width * height
        );
    }
    Ok(())
}

fn load_npz_stack_as_f32(path: &Path) -> Result<(Vec<f32>, StackShape)> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;
    let names = npz
        .names()
        .with_context(|| format!("Failed to list NPZ arrays in {}", path.display()))?;
    let first = names
        .first()
        .ok_or_else(|| anyhow::anyhow!("NPZ archive is empty: {}", path.display()))?
        .clone();
    drop(npz);

    let shape = read_npz_shape(path, &first)?;
    let pixels = read_npz_pixels(path, &first)?;
    Ok((pixels, shape))
}

fn load_npz_stack_as_u32(path: &Path) -> Result<(Vec<u32>, StackShape)> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;
    let names = npz
        .names()
        .with_context(|| format!("Failed to list NPZ arrays in {}", path.display()))?;
    let first = names
        .first()
        .ok_or_else(|| anyhow::anyhow!("NPZ archive is empty: {}", path.display()))?
        .clone();
    drop(npz);

    let shape = read_npz_shape(path, &first)?;
    let pixels = read_npz_pixels_as_u32(path, &first)?;
    Ok((pixels, shape))
}

fn read_npz_shape(path: &Path, name: &str) -> Result<StackShape> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;

    if let Ok(array) = npz.by_name::<OwnedRepr<f32>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<f64>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u8>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u16>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u32>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u64>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i8>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i16>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i32>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i64>, IxDyn>(name) {
        return infer_stack_shape_from_dims(array.shape().to_vec(), path);
    }

    bail!("Unsupported NPZ element type in {}", path.display())
}

fn read_npz_pixels(path: &Path, name: &str) -> Result<Vec<f32>> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;

    if let Ok(array) = npz.by_name::<OwnedRepr<f32>, IxDyn>(name) {
        return Ok(flatten_array(array));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<f64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u8>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u16>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u32>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i8>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i16>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i32>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as f32)
            .collect());
    }

    bail!("Unsupported NPZ element type in {}", path.display())
}

fn read_npz_pixels_as_u32(path: &Path, name: &str) -> Result<Vec<u32>> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;

    if let Ok(array) = npz.by_name::<OwnedRepr<f32>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0.0) as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<f64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0.0) as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u8>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u16>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u32>, IxDyn>(name) {
        return Ok(flatten_array(array));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i8>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0) as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i16>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0) as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i32>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0) as u32)
            .collect());
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i64>, IxDyn>(name) {
        return Ok(flatten_array(array)
            .into_iter()
            .map(|value| value.max(0) as u32)
            .collect());
    }

    bail!("Unsupported NPZ element type in {}", path.display())
}

fn load_h5_stack_as_f32(path: &Path) -> Result<(Vec<f32>, StackShape)> {
    let file =
        Hdf5File::open(path).with_context(|| format!("Failed to open H5 {}", path.display()))?;
    let dataset = file
        .dataset("/data")
        .or_else(|_| file.dataset("data"))
        .with_context(|| format!("Failed to open dataset \"data\" in {}", path.display()))?;
    let shape = infer_stack_shape_from_dims(
        dataset.shape().iter().map(|dim| *dim as usize).collect(),
        path,
    )?;

    macro_rules! try_h5_array {
        ($ty:ty) => {
            dataset.read_array::<$ty>().map(|values| {
                values
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
            })
        };
    }

    let pixels = try_h5_array!(f32)
        .or_else(|_| try_h5_array!(f64))
        .or_else(|_| try_h5_array!(u8))
        .or_else(|_| try_h5_array!(u16))
        .or_else(|_| try_h5_array!(u32))
        .or_else(|_| try_h5_array!(u64))
        .or_else(|_| try_h5_array!(i8))
        .or_else(|_| try_h5_array!(i16))
        .or_else(|_| try_h5_array!(i32))
        .or_else(|_| try_h5_array!(i64))
        .with_context(|| format!("Unsupported H5 dataset type in {}", path.display()))?;

    Ok((pixels, shape))
}

fn load_h5_stack_as_u32(path: &Path) -> Result<(Vec<u32>, StackShape)> {
    let file =
        Hdf5File::open(path).with_context(|| format!("Failed to open H5 {}", path.display()))?;
    let dataset = file
        .dataset("/data")
        .or_else(|_| file.dataset("data"))
        .with_context(|| format!("Failed to open dataset \"data\" in {}", path.display()))?;
    let shape = infer_stack_shape_from_dims(
        dataset.shape().iter().map(|dim| *dim as usize).collect(),
        path,
    )?;

    macro_rules! try_h5_array {
        ($ty:ty) => {
            dataset.read_array::<$ty>().map(|values| {
                values
                    .into_iter()
                    .map(|value| value.max(0 as $ty) as u32)
                    .collect::<Vec<_>>()
            })
        };
    }

    let pixels = dataset
        .read_array::<u32>()
        .map(|values| values.into_iter().collect::<Vec<_>>())
        .or_else(|_| {
            dataset
                .read_array::<u16>()
                .map(|values| values.into_iter().map(|v| v as u32).collect())
        })
        .or_else(|_| {
            dataset
                .read_array::<u8>()
                .map(|values| values.into_iter().map(|v| v as u32).collect())
        })
        .or_else(|_| try_h5_array!(i32))
        .or_else(|_| try_h5_array!(i16))
        .or_else(|_| try_h5_array!(i8))
        .or_else(|_| {
            dataset
                .read_array::<f32>()
                .map(|values| values.into_iter().map(|v| v.max(0.0) as u32).collect())
        })
        .or_else(|_| {
            dataset
                .read_array::<f64>()
                .map(|values| values.into_iter().map(|v| v.max(0.0) as u32).collect())
        })
        .with_context(|| format!("Unsupported H5 dataset type in {}", path.display()))?;

    Ok((pixels, shape))
}

fn infer_stack_shape_from_dims(dims: Vec<usize>, path: &Path) -> Result<StackShape> {
    match dims.as_slice() {
        [height, width] => Ok(StackShape {
            frames: 1,
            height: *height,
            width: *width,
        }),
        [frames, height, width] => Ok(StackShape {
            frames: *frames,
            height: *height,
            width: *width,
        }),
        _ => bail!(
            "Unsupported image shape {:?} in {}. This phase only supports 2D images or 2D timelapse stacks.",
            dims,
            path.display()
        ),
    }
}

fn flatten_array<T>(array: ArrayD<T>) -> Vec<T> {
    array.into_iter().collect()
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

        let (pixels, shape) = load_image_stack_as_f32(&path)?;
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

    #[test]
    fn loads_npz_stack() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("stack.npz");
        let file = File::create(&path)?;
        let mut writer = NpzWriter::new(file);
        let array =
            Array3::from_shape_vec((2, 2, 3), vec![1u16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;

        let (pixels, shape) = load_image_stack_as_f32(&path)?;
        assert_eq!(shape.frames, 2);
        assert_eq!(shape.height, 2);
        assert_eq!(shape.width, 3);
        assert_eq!(pixels[0], 1.0);
        assert_eq!(pixels[11], 12.0);
        Ok(())
    }

    #[test]
    fn loads_masks_as_u32() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("segm.npz");
        let file = File::create(&path)?;
        let mut writer = NpzWriter::new(file);
        let array = Array3::from_shape_vec((2, 2, 2), vec![0u16, 1, 2, 3, 4, 5, 6, 7])?;
        writer.add_array("arr_0", &array)?;
        writer.finish()?;

        let (pixels, shape) = load_mask_stack_as_u32(&path)?;
        assert_eq!(shape.frames, 2);
        assert_eq!(pixels[3], 3);
        assert_eq!(pixels[7], 7);
        Ok(())
    }

    #[test]
    fn loads_all_npz_arrays() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("bkgr.npz");
        let file = File::create(&path)?;
        let mut writer = NpzWriter::new(file);
        let left = Array2::from_shape_vec((2, 2), vec![1u16, 2, 3, 4])?;
        let right = Array2::from_shape_vec((2, 2), vec![5u16, 6, 7, 8])?;
        writer.add_array("roi0_data", &left)?;
        writer.add_array("roi1_data", &right)?;
        writer.finish()?;

        let arrays = load_npz_archive_arrays_as_f32(&path)?;
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].shape.height, 2);
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
