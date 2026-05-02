use anyhow::{anyhow, bail, Context, Result};
use hdf5_reader::Hdf5File;
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{read_npy, write_npy, NpzReader, NpzWriter};
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
        Some("npy") => load_npy(path)?,
        Some("npz") => load_npz(path)?,
        Some("h5") => load_h5(path)?,
        Some("tif") | Some("tiff") => load_tiff(path, &resolution)?,
        other => bail!(
            "Unsupported mask format {:?} for {}. Supported formats are NPY, NPZ, TIFF, and H5.",
            other,
            path.display()
        ),
    };

    let dims = squeeze_non_spatial_singletons(&dims, resolution.layout);
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
        Some("npy") => write_npy(path, &data.values)
            .with_context(|| format!("Failed to write NPY {}", path.display())),
        Some("npz") => save_npz(path, data),
        Some("tif") | Some("tiff") => save_tiff(path, data),
        Some("h5") => bail!(
            "Writing H5 segmentation masks is not supported yet for {}",
            path.display()
        ),
        other => bail!(
            "Unsupported mask output format {:?} for {}. Supported write formats are NPY, NPZ, and TIFF.",
            other,
            path.display()
        ),
    }
}

fn load_npy(path: &Path) -> Result<(Vec<u32>, Vec<usize>)> {
    macro_rules! try_npy {
        ($ty:ty, $map:expr) => {
            if let Ok(array) = read_npy::<_, ArrayD<$ty>>(path) {
                let dims = array.shape().to_vec();
                let values = array.iter().map($map).collect::<Vec<_>>();
                return Ok((values, dims));
            }
        };
    }

    try_npy!(u32, |value| *value);
    try_npy!(bool, |value| u32::from(*value));
    try_npy!(u16, |value| *value as u32);
    try_npy!(u8, |value| *value as u32);
    try_npy!(i64, |value| (*value).max(0) as u32);
    try_npy!(i32, |value| (*value).max(0) as u32);
    try_npy!(i16, |value| (*value).max(0) as u32);
    try_npy!(i8, |value| (*value).max(0) as u32);
    try_npy!(f64, |value| (*value).max(0.0) as u32);
    try_npy!(f32, |value| (*value).max(0.0) as u32);

    bail!("Unsupported NPY mask dtype in {}", path.display())
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
    if let Ok(array) = npz.by_name::<OwnedRepr<bool>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| u32::from(*value)).collect(),
            array.shape().to_vec(),
        ));
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
    if let Ok(array) = npz.by_name::<OwnedRepr<u64>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| *value as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i64>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| (*value).max(0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i32>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| (*value).max(0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i16>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| (*value).max(0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<i8>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| (*value).max(0) as u32).collect(),
            array.shape().to_vec(),
        ));
    }
    if let Ok(array) = npz.by_name::<OwnedRepr<f64>, IxDyn>(&name) {
        return Ok((
            array.iter().map(|value| value.max(0.0) as u32).collect(),
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

fn squeeze_non_spatial_singletons(
    dims: &[usize],
    explicit_layout: Option<SegmentationLayout>,
) -> Vec<usize> {
    if dims.len() <= 2 {
        return dims.to_vec();
    }
    let target_len = explicit_layout.map(layout_rank).unwrap_or(2);
    let spatial_start = dims.len() - 2;
    let mut squeezed = Vec::new();
    let mut remaining_to_remove = dims.len().saturating_sub(target_len);
    for dim in &dims[..spatial_start] {
        if *dim == 1 && remaining_to_remove > 0 {
            remaining_to_remove -= 1;
            continue;
        }
        squeezed.push(*dim);
    }
    squeezed.extend_from_slice(&dims[spatial_start..]);
    squeezed
}

fn layout_rank(layout: SegmentationLayout) -> usize {
    match layout {
        SegmentationLayout::YX => 2,
        SegmentationLayout::TYX | SegmentationLayout::ZYX => 3,
        SegmentationLayout::TZYX => 4,
    }
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
    use ndarray_npy::write_npy;
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
    fn squeezes_singleton_axes_in_array_backed_masks_like_python() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("mask.npz");
        let file = File::create(&path)?;
        let mut writer = NpzWriter::new(file);
        let values =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 2, 2]), vec![0u32, 1, 2, 0, 0, 3, 4, 0])?;
        writer.add_array("arr_0", &values)?;
        writer.finish()?;

        let loaded = load_mask_data(
            &path,
            Some(&MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        )?;

        assert_eq!(loaded.layout, SegmentationLayout::TYX);
        assert_eq!(loaded.values.shape(), &[2, 2, 2]);
        assert_eq!(
            loaded.values.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 0, 3, 4, 0]
        );
        Ok(())
    }

    #[test]
    fn loads_npy_masks_like_python_segmentation_files() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("mask.npy");
        let values = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![0i32, 1, 2, -3])?;
        write_npy(&path, &values)?;

        let loaded = load_mask_data(
            &path,
            Some(&MaskPathResolution {
                size_t: Some(2),
                size_z: Some(1),
                layout: Some(SegmentationLayout::TYX),
            }),
        )?;

        assert_eq!(loaded.layout, SegmentationLayout::TYX);
        assert_eq!(loaded.values.shape(), &[2, 1, 2]);
        assert_eq!(
            loaded.values.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 0]
        );
        Ok(())
    }

    #[test]
    fn loads_boolean_array_backed_masks_as_binary_labels() -> Result<()> {
        let temp = tempdir()?;
        let npy_path = temp.path().join("mask.npy");
        let bool_values = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![false, true, true, false])?;
        write_npy(&npy_path, &bool_values)?;

        let loaded_npy = load_mask_data(&npy_path, None)?;
        assert_eq!(loaded_npy.layout, SegmentationLayout::YX);
        assert_eq!(
            loaded_npy.values.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 1, 0]
        );

        let npz_path = temp.path().join("mask.npz");
        let file = File::create(&npz_path)?;
        let mut writer = NpzWriter::new(file);
        writer.add_array("arr_0", &bool_values)?;
        writer.finish()?;

        let loaded_npz = load_mask_data(&npz_path, None)?;
        assert_eq!(loaded_npz.layout, SegmentationLayout::YX);
        assert_eq!(loaded_npz.values, loaded_npy.values);
        Ok(())
    }

    #[test]
    fn loads_npz_masks_with_python_numeric_cast_variants() -> Result<()> {
        let temp = tempdir()?;

        let i16_path = temp.path().join("mask_i16.npz");
        let file = File::create(&i16_path)?;
        let mut writer = NpzWriter::new(file);
        let i16_values = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1i16, 2, 3, -4])?;
        writer.add_array("arr_0", &i16_values)?;
        writer.finish()?;
        let loaded_i16 = load_mask_data(&i16_path, None)?;
        assert_eq!(
            loaded_i16.values.iter().copied().collect::<Vec<_>>(),
            vec![0, 2, 3, 0]
        );

        let f64_path = temp.path().join("mask_f64.npz");
        let file = File::create(&f64_path)?;
        let mut writer = NpzWriter::new(file);
        let f64_values = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0f64, 1.8, -2.0, 4.2])?;
        writer.add_array("arr_0", &f64_values)?;
        writer.finish()?;
        let loaded_f64 = load_mask_data(&f64_path, None)?;
        assert_eq!(
            loaded_f64.values.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 0, 4]
        );
        Ok(())
    }

    #[test]
    fn roundtrips_npy_masks() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("mask.npy");
        let data = MaskData {
            values: ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![0u32, 1, 2, 3])?,
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
