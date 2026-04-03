use anyhow::{anyhow, bail, Context, Result};
use hdf5_reader::Hdf5File;
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{NpzReader, NpzWriter};
use std::fs::File;
use std::path::{Path, PathBuf};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{colortype, TiffEncoder};
use tiff::ColorType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskDimensionality {
    D2,
    D3,
    D4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentationLayout {
    YX,
    TYX,
    ZYX,
    TZYX,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaskPathResolution {
    pub size_t: Option<usize>,
    pub size_z: Option<usize>,
    pub layout: Option<SegmentationLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaskData {
    pub values: ArrayD<u32>,
    pub layout: SegmentationLayout,
    pub source_path: PathBuf,
}

impl MaskData {
    pub fn dimensionality(&self) -> MaskDimensionality {
        match self.values.ndim() {
            2 => MaskDimensionality::D2,
            3 => MaskDimensionality::D3,
            4 => MaskDimensionality::D4,
            ndim => panic!("unsupported mask ndim {ndim}"),
        }
    }
}

pub fn load_mask_data(path: &Path, resolution: Option<&MaskPathResolution>) -> Result<MaskData> {
    let resolution = resolution.cloned().unwrap_or_default();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    let (values, dims) = match extension.as_deref() {
        Some("npz") => load_npz(path)?,
        Some("h5") => load_h5(path)?,
        Some("tif") | Some("tiff") => load_tiff(path, &resolution)?,
        other => bail!(
            "Unsupported mask format {:?} for {}. Supported formats are TIFF, NPZ, and H5.",
            other,
            path.display()
        ),
    };

    let layout = infer_layout(&dims, &resolution, path)?;
    let values = ArrayD::from_shape_vec(IxDyn(&dims), values)
        .with_context(|| format!("Failed to shape mask data for {}", path.display()))?;
    Ok(MaskData {
        values,
        layout,
        source_path: path.to_path_buf(),
    })
}

pub fn save_mask_data(path: &Path, data: &MaskData) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    match extension.as_deref() {
        Some("npz") => save_npz(path, data),
        Some("tif") | Some("tiff") => save_tiff(path, data),
        Some("h5") => bail!(
            "Writing H5 segmentation masks is not supported yet for {}",
            path.display()
        ),
        other => bail!(
            "Unsupported mask output format {:?} for {}. Supported write formats are NPZ and TIFF.",
            other,
            path.display()
        ),
    }
}

fn load_npz(path: &Path) -> Result<(Vec<u32>, Vec<usize>)> {
    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to read NPZ {}", path.display()))?;
    let names = npz.names()?;
    let name = names
        .first()
        .ok_or_else(|| anyhow!("NPZ archive is empty: {}", path.display()))?
        .clone();
    drop(npz);

    let mut npz = NpzReader::new(File::open(path)?)
        .with_context(|| format!("Failed to reopen NPZ {}", path.display()))?;
    if let Ok(array) = npz.by_name::<OwnedRepr<u32>, IxDyn>(&name) {
        return Ok((array.iter().copied().collect(), array.shape().to_vec()));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u16>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| *value as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<u8>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| *value as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i32>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| (*value).max(0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<f32>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| value.max(0.0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }

    bail!("Unsupported NPZ mask dtype in {}", path.display())
}

fn load_h5(path: &Path) -> Result<(Vec<u32>, Vec<usize>)> {
    let file =
        Hdf5File::open(path).with_context(|| format!("Failed to open H5 {}", path.display()))?;
    let dataset = file
        .dataset("/data")
        .or_else(|_| file.dataset("data"))
        .with_context(|| format!("Failed to open dataset \"data\" in {}", path.display()))?;
    let dims = dataset
        .shape()
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let values = dataset
        .read_array::<u32>()
        .map(|values| values.into_iter().collect::<Vec<_>>())
        .or_else(|_| {
            dataset
                .read_array::<u16>()
                .map(|values| values.into_iter().map(|value| value as u32).collect())
        })
        .or_else(|_| {
            dataset
                .read_array::<u8>()
                .map(|values| values.into_iter().map(|value| value as u32).collect())
        })
        .or_else(|_| {
            dataset.read_array::<i32>().map(|values| {
                values
                    .into_iter()
                    .map(|value| value.max(0) as u32)
                    .collect()
            })
        })
        .or_else(|_| {
            dataset.read_array::<f32>().map(|values| {
                values
                    .into_iter()
                    .map(|value| value.max(0.0) as u32)
                    .collect()
            })
        })
        .with_context(|| format!("Unsupported H5 dataset type in {}", path.display()))?;
    Ok((values, dims))
}

fn load_tiff(path: &Path, resolution: &MaskPathResolution) -> Result<(Vec<u32>, Vec<usize>)> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut decoder =
        Decoder::new(file).with_context(|| format!("Failed to decode TIFF {}", path.display()))?;
    let mut pages = Vec::new();
    let mut height = None;
    let mut width = None;
    loop {
        let dimensions = decoder
            .dimensions()
            .with_context(|| format!("Failed to read TIFF dimensions from {}", path.display()))?;
        let color_type = decoder
            .colortype()
            .with_context(|| format!("Failed to read TIFF color type from {}", path.display()))?;
        if !matches!(color_type, ColorType::Gray(_)) {
            bail!(
                "Unsupported TIFF color type {:?} in {}. This phase expects grayscale masks.",
                color_type,
                path.display()
            );
        }
        let page_height = dimensions.1 as usize;
        let page_width = dimensions.0 as usize;
        if let Some(expected) = height {
            if expected != page_height || width != Some(page_width) {
                bail!(
                    "TIFF pages in {} do not share the same dimensions",
                    path.display()
                );
            }
        } else {
            height = Some(page_height);
            width = Some(page_width);
        }
        let page = match decoder.read_image()? {
            DecodingResult::U8(values) => values.into_iter().map(|value| value as u32).collect(),
            DecodingResult::U16(values) => values.into_iter().map(|value| value as u32).collect(),
            DecodingResult::U32(values) => values,
            DecodingResult::I16(values) => values
                .into_iter()
                .map(|value| value.max(0) as u32)
                .collect(),
            DecodingResult::I32(values) => values
                .into_iter()
                .map(|value| value.max(0) as u32)
                .collect(),
            DecodingResult::F32(values) => values
                .into_iter()
                .map(|value| value.max(0.0) as u32)
                .collect(),
            other => bail!(
                "Unsupported TIFF mask pixel format {:?} in {}",
                other,
                path.display()
            ),
        };
        pages.push(page);
        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    let page_count = pages.len();
    let height = height.unwrap_or(0);
    let width = width.unwrap_or(0);
    let layout = resolution
        .layout
        .or_else(|| infer_tiff_layout(page_count, &resolution.size_t, &resolution.size_z))
        .unwrap_or(SegmentationLayout::TYX);
    let dims = match layout {
        SegmentationLayout::YX => vec![height, width],
        SegmentationLayout::TYX => vec![page_count, height, width],
        SegmentationLayout::ZYX => vec![page_count, height, width],
        SegmentationLayout::TZYX => {
            let size_t = resolution.size_t.ok_or_else(|| {
                anyhow!(
                    "TIFF {} requires metadata SizeT to resolve TZYX layout",
                    path.display()
                )
            })?;
            let size_z = resolution.size_z.ok_or_else(|| {
                anyhow!(
                    "TIFF {} requires metadata SizeZ to resolve TZYX layout",
                    path.display()
                )
            })?;
            if page_count != size_t * size_z {
                bail!(
                    "TIFF page count {} does not match SizeT * SizeZ ({} * {}) in {}",
                    page_count,
                    size_t,
                    size_z,
                    path.display()
                );
            }
            vec![size_t, size_z, height, width]
        }
    };
    let values = pages.into_iter().flatten().collect::<Vec<_>>();
    Ok((values, dims))
}

fn infer_tiff_layout(
    page_count: usize,
    size_t: &Option<usize>,
    size_z: &Option<usize>,
) -> Option<SegmentationLayout> {
    match (*size_t, *size_z) {
        (Some(1) | None, Some(1) | None) if page_count == 1 => Some(SegmentationLayout::YX),
        (Some(t), Some(1) | None) if t == page_count => Some(SegmentationLayout::TYX),
        (Some(1) | None, Some(z)) if z == page_count => Some(SegmentationLayout::ZYX),
        (Some(t), Some(z)) if t > 1 && z > 1 && t * z == page_count => {
            Some(SegmentationLayout::TZYX)
        }
        _ => None,
    }
}

fn infer_layout(
    dims: &[usize],
    resolution: &MaskPathResolution,
    path: &Path,
) -> Result<SegmentationLayout> {
    if let Some(layout) = resolution.layout {
        validate_layout_dims(layout, dims, path)?;
        return Ok(layout);
    }
    match dims {
        [_, _] => Ok(SegmentationLayout::YX),
        [frames_or_z, _, _] => match (resolution.size_t, resolution.size_z) {
            (Some(size_t), Some(size_z)) if size_t > 1 && size_z <= 1 && size_t == *frames_or_z => {
                Ok(SegmentationLayout::TYX)
            }
            (Some(size_t), Some(size_z)) if size_z > 1 && size_t <= 1 && size_z == *frames_or_z => {
                Ok(SegmentationLayout::ZYX)
            }
            (Some(size_t), None) if size_t > 1 && size_t == *frames_or_z => Ok(SegmentationLayout::TYX),
            (None, Some(size_z)) if size_z > 1 && size_z == *frames_or_z => Ok(SegmentationLayout::ZYX),
            _ => bail!(
                "Ambiguous 3D segmentation layout for {} with dims {:?}. Provide metadata SizeT/SizeZ or an explicit layout.",
                path.display(),
                dims
            ),
        },
        [size_t, size_z, _, _] => {
            if let Some(expected_t) = resolution.size_t {
                if expected_t != *size_t {
                    bail!(
                        "Mask SizeT mismatch in {}: dims say {}, metadata says {}",
                        path.display(),
                        size_t,
                        expected_t
                    );
                }
            }
            if let Some(expected_z) = resolution.size_z {
                if expected_z != *size_z {
                    bail!(
                        "Mask SizeZ mismatch in {}: dims say {}, metadata says {}",
                        path.display(),
                        size_z,
                        expected_z
                    );
                }
            }
            Ok(SegmentationLayout::TZYX)
        }
        _ => bail!(
            "Unsupported segmentation shape {:?} in {}. This phase supports 2D/3D/4D masks only.",
            dims,
            path.display()
        ),
    }
}

fn validate_layout_dims(layout: SegmentationLayout, dims: &[usize], path: &Path) -> Result<()> {
    let valid = match layout {
        SegmentationLayout::YX => dims.len() == 2,
        SegmentationLayout::TYX | SegmentationLayout::ZYX => dims.len() == 3,
        SegmentationLayout::TZYX => dims.len() == 4,
    };
    if !valid {
        bail!(
            "Explicit layout {:?} is incompatible with dims {:?} in {}",
            layout,
            dims,
            path.display()
        );
    }
    Ok(())
}

fn save_npz(path: &Path, data: &MaskData) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = NpzWriter::new_compressed(file);
    match data.values.ndim() {
        2 => {
            let array = data
                .values
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .with_context(|| format!("Failed to convert mask dims for {}", path.display()))?;
            writer.add_array("arr_0", &array)?;
        }
        3 => {
            let array = data
                .values
                .clone()
                .into_dimensionality::<ndarray::Ix3>()
                .with_context(|| format!("Failed to convert mask dims for {}", path.display()))?;
            writer.add_array("arr_0", &array)?;
        }
        4 => {
            let array = data
                .values
                .clone()
                .into_dimensionality::<ndarray::Ix4>()
                .with_context(|| format!("Failed to convert mask dims for {}", path.display()))?;
            writer.add_array("arr_0", &array)?;
        }
        ndim => bail!("Unsupported mask ndim {} for {}", ndim, path.display()),
    }
    writer.finish()?;
    Ok(())
}

fn save_tiff(path: &Path, data: &MaskData) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut encoder = TiffEncoder::new(file)?;
    let planes = flatten_to_planes(data)?;
    for plane in planes {
        let height = plane.0;
        let width = plane.1;
        let pixels = plane.2;
        encoder.write_image::<colortype::Gray32>(width as u32, height as u32, &pixels)?;
    }
    Ok(())
}

fn flatten_to_planes(data: &MaskData) -> Result<Vec<(usize, usize, Vec<u32>)>> {
    match data.values.ndim() {
        2 => {
            let array = data.values.view().into_dimensionality::<ndarray::Ix2>()?;
            Ok(vec![(
                array.shape()[0],
                array.shape()[1],
                array.iter().copied().collect(),
            )])
        }
        3 => {
            let array = data.values.view().into_dimensionality::<ndarray::Ix3>()?;
            let planes = array
                .outer_iter()
                .map(|plane| {
                    (
                        plane.shape()[0],
                        plane.shape()[1],
                        plane.iter().copied().collect::<Vec<_>>(),
                    )
                })
                .collect();
            Ok(planes)
        }
        4 => {
            let array = data.values.view().into_dimensionality::<ndarray::Ix4>()?;
            let mut planes = Vec::new();
            for stack in array.outer_iter() {
                for plane in stack.outer_iter() {
                    planes.push((
                        plane.shape()[0],
                        plane.shape()[1],
                        plane.iter().copied().collect::<Vec<_>>(),
                    ));
                }
            }
            Ok(planes)
        }
        ndim => bail!("Unsupported mask ndim {} for TIFF output", ndim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrips_npz_masks() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("mask.npz");
        let data = MaskData {
            values: ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect())?,
            layout: SegmentationLayout::TYX,
            source_path: path.clone(),
        };
        save_mask_data(&path, &data)?;
        let loaded = load_mask_data(
            &path,
            Some(&MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        )?;
        assert_eq!(loaded.layout, SegmentationLayout::TYX);
        assert_eq!(loaded.values, data.values);
        Ok(())
    }

    #[test]
    fn saves_multi_page_tiff_for_4d_masks() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("mask.tif");
        let data = MaskData {
            values: ArrayD::from_shape_vec(IxDyn(&[2, 3, 4, 5]), (0..120).collect())?,
            layout: SegmentationLayout::TZYX,
            source_path: path.clone(),
        };
        save_mask_data(&path, &data)?;
        let loaded = load_mask_data(
            &path,
            Some(&MaskPathResolution {
                size_t: Some(2),
                size_z: Some(3),
                layout: Some(SegmentationLayout::TZYX),
            }),
        )?;
        assert_eq!(loaded.values.shape(), &[2, 3, 4, 5]);
        Ok(())
    }
}
