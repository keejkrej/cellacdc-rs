use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::image_io::{load_image_stack_as_f32, load_image_volume_as_f32};
use crate::layout::{
    discover_measurement_experiment, resolve_measurement_position, MeasurementExperimentSpec,
    MeasurementPositionSpec,
};
use crate::mask_io::{load_mask_data, MaskData, MaskPathResolution, SegmentationLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameProjection {
    Max,
    ZSlice(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewPlane {
    XY,
    XZ,
    YZ,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameData<T> {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationAsset {
    pub name: String,
    pub endname: Option<String>,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositionSession {
    pub spec: MeasurementPositionSpec,
    pub segmentations: Vec<SegmentationAsset>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSession {
    pub root_path: PathBuf,
    pub positions: Vec<PositionSession>,
    pub is_single_position: bool,
}

pub fn open_experiment_session(path: impl AsRef<Path>) -> Result<ExperimentSession> {
    let path = path.as_ref();
    if let Ok(position) = open_position_session(path) {
        return Ok(ExperimentSession {
            root_path: path.to_path_buf(),
            positions: vec![position],
            is_single_position: true,
        });
    }

    let experiment = discover_measurement_experiment(path)?;
    experiment_session_from_spec(path, experiment)
}

pub fn open_position_session(path: impl AsRef<Path>) -> Result<PositionSession> {
    let spec = resolve_measurement_position(path)?;
    let segmentations = discover_segmentation_assets(&spec)?;
    Ok(PositionSession {
        spec,
        segmentations,
    })
}

impl ExperimentSession {
    pub fn reload(&self) -> Result<Self> {
        open_experiment_session(&self.root_path)
    }
}

impl PositionSession {
    pub fn position_key(&self) -> String {
        self.spec
            .position_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("Position")
            .to_string()
    }

    pub fn acdc_output_path(&self, endname: Option<&str>) -> PathBuf {
        let suffix = endname
            .filter(|value| !value.trim().is_empty())
            .map(|value| format!("_{value}"))
            .unwrap_or_default();
        self.spec
            .images_dir
            .join(format!("{}acdc_output{suffix}.csv", self.spec.basename))
    }

    pub fn custom_annotation_params_path(&self) -> PathBuf {
        self.spec
            .images_dir
            .join(format!("{}custom_annot_params.json", self.spec.basename))
    }

    pub fn data_prep_state_paths(&self) -> crate::prep::PrepOutputPaths {
        crate::prep::PrepOutputPaths {
            primary_path: self
                .spec
                .images_dir
                .join(format!("{}dataPrep_bkgrROIs.json", self.spec.basename)),
            secondary_paths: vec![
                self.spec
                    .images_dir
                    .join(format!("{}dataPrepROIs_coords.csv", self.spec.basename)),
                self.spec
                    .images_dir
                    .join(format!("{}dataPrepFreeRoi.npz", self.spec.basename)),
                self.spec
                    .images_dir
                    .join(format!("{}segmInfo.csv", self.spec.basename)),
                self.alignment_shifts_path(),
            ],
        }
    }

    pub fn aligned_channel_path(&self, channel_name: &str) -> Option<PathBuf> {
        let aligned_h5 = self
            .spec
            .images_dir
            .join(format!("{}{}_aligned.h5", self.spec.basename, channel_name));
        if aligned_h5.exists() {
            return Some(aligned_h5);
        }
        let aligned_npz = self
            .spec
            .images_dir
            .join(format!("{}{}_aligned.npz", self.spec.basename, channel_name));
        aligned_npz.exists().then_some(aligned_npz)
    }

    pub fn alignment_shifts_path(&self) -> PathBuf {
        self.spec
            .images_dir
            .join(format!("{}align_shift.npy", self.spec.basename))
    }

    pub fn channel_names(&self) -> Vec<String> {
        self.spec
            .channels
            .iter()
            .map(|channel| channel.name.clone())
            .collect()
    }

    pub fn default_channel_name(&self) -> Option<String> {
        self.spec
            .channels
            .first()
            .map(|channel| channel.name.clone())
    }

    pub fn default_phase_channel_name(&self) -> Option<String> {
        self.spec
            .channels
            .iter()
            .find(|channel| looks_like_phase_channel(&channel.name))
            .or_else(|| self.spec.channels.first())
            .map(|channel| channel.name.clone())
    }

    pub fn default_fluo_channel_name(&self) -> Option<String> {
        self.spec
            .channels
            .iter()
            .find(|channel| {
                let lower = channel.name.to_ascii_lowercase();
                !looks_like_phase_channel(&lower)
            })
            .or_else(|| self.spec.channels.get(1))
            .or_else(|| self.spec.channels.first())
            .map(|channel| channel.name.clone())
    }

    pub fn load_channel_frame(
        &self,
        channel_name: &str,
        frame_index: usize,
        projection: FrameProjection,
    ) -> Result<FrameData<f32>> {
        self.load_channel_frame_for_view(channel_name, frame_index, ViewPlane::XY, projection)
    }

    pub fn load_channel_frame_for_view(
        &self,
        channel_name: &str,
        frame_index: usize,
        view_plane: ViewPlane,
        projection: FrameProjection,
    ) -> Result<FrameData<f32>> {
        let channel = self
            .spec
            .channels
            .iter()
            .find(|channel| channel.name == channel_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown channel {channel_name:?}"))?;

        if self.spec.size_z > 1 {
            let (values, shape) = load_image_volume_as_f32(
                &channel.image_path,
                Some(self.spec.size_t),
                Some(self.spec.size_z),
            )
            .with_context(|| {
                format!(
                    "Failed to load image data for channel {} from {}",
                    channel.name,
                    channel.image_path.display()
                )
            })?;
            extract_volume_frame_f32(
                &values,
                shape.height,
                shape.width,
                shape.size_t,
                shape.size_z,
                frame_index,
                view_plane,
                projection,
            )
        } else {
            let (values, shape) =
                load_image_stack_as_f32(&channel.image_path).with_context(|| {
                    format!(
                        "Failed to load image data for channel {} from {}",
                        channel.name,
                        channel.image_path.display()
                    )
                })?;
            extract_stack_frame_f32(
                &values,
                shape.height,
                shape.width,
                shape.frames,
                frame_index,
            )
        }
    }

    pub fn load_segmentation_frame(
        &self,
        endname: Option<&str>,
        frame_index: usize,
        projection: FrameProjection,
    ) -> Result<Option<FrameData<u32>>> {
        self.load_segmentation_frame_for_view(endname, frame_index, ViewPlane::XY, projection)
    }

    pub fn load_segmentation_frame_for_view(
        &self,
        endname: Option<&str>,
        frame_index: usize,
        view_plane: ViewPlane,
        projection: FrameProjection,
    ) -> Result<Option<FrameData<u32>>> {
        let Some(mask) = self.load_segmentation_mask(endname)? else {
            return Ok(None);
        };
        let shape = mask.values.shape().to_vec();
        let values = mask.values.as_slice_memory_order().ok_or_else(|| {
            anyhow::anyhow!(
                "Mask data is not contiguous: {}",
                mask.source_path.display()
            )
        })?;

        let frame = match mask.layout {
            SegmentationLayout::YX => {
                if frame_index > 0 {
                    bail!(
                        "Requested frame {} from a single-frame segmentation {}",
                        frame_index,
                        mask.source_path.display()
                    );
                }
                FrameData {
                    width: shape[1],
                    height: shape[0],
                    pixels: values.to_vec(),
                }
            }
            SegmentationLayout::TYX => {
                extract_stack_frame_u32(values, shape[1], shape[2], shape[0], frame_index)?
            }
            SegmentationLayout::ZYX => {
                if frame_index > 0 {
                    bail!(
                        "Requested frame {} from a single-frame z-stack segmentation {}",
                        frame_index,
                        mask.source_path.display()
                    );
                }
                extract_volume_frame_u32(
                    values,
                    shape[1],
                    shape[2],
                    1,
                    shape[0],
                    0,
                    view_plane,
                    projection,
                )?
            }
            SegmentationLayout::TZYX => extract_volume_frame_u32(
                values,
                shape[2],
                shape[3],
                shape[0],
                shape[1],
                frame_index,
                view_plane,
                projection,
            )?,
        };

        Ok(Some(frame))
    }

    pub fn load_segmentation_mask(&self, endname: Option<&str>) -> Result<Option<MaskData>> {
        let Some(asset) = self.segmentation_asset(endname) else {
            return Ok(None);
        };
        let is_segm_3d = self
            .spec
            .segm_is_3d
            .get(&asset.name)
            .copied()
            .unwrap_or(false);
        let resolution = MaskPathResolution {
            size_t: Some(self.spec.size_t),
            size_z: Some(if is_segm_3d { self.spec.size_z } else { 1 }),
            layout: None,
        };
        let mask = load_mask_data(&asset.path, Some(&resolution)).with_context(|| {
            format!(
                "Failed to load segmentation masks from {}",
                asset.path.display()
            )
        })?;
        Ok(Some(mask))
    }

    pub fn segmentation_asset(&self, endname: Option<&str>) -> Option<&SegmentationAsset> {
        self.segmentations
            .iter()
            .find(|asset| asset.endname.as_deref() == normalize_endname(endname))
    }
}

fn experiment_session_from_spec(
    root_path: &Path,
    experiment: MeasurementExperimentSpec,
) -> Result<ExperimentSession> {
    let mut positions = Vec::with_capacity(experiment.positions.len());
    for spec in experiment.positions {
        let segmentations = discover_segmentation_assets(&spec)?;
        positions.push(PositionSession {
            spec,
            segmentations,
        });
    }
    Ok(ExperimentSession {
        root_path: root_path.to_path_buf(),
        positions,
        is_single_position: false,
    })
}

fn discover_segmentation_assets(spec: &MeasurementPositionSpec) -> Result<Vec<SegmentationAsset>> {
    let mut segmentations = Vec::new();
    for entry in fs::read_dir(&spec.images_dir)
        .with_context(|| format!("Failed to read {}", spec.images_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(suffix) = file_name.strip_prefix(&spec.basename) else {
            continue;
        };
        if suffix.starts_with("segm_hyperparams") || !suffix.starts_with("segm") {
            continue;
        }
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !matches!(
            ext.to_ascii_lowercase().as_str(),
            "npz" | "tif" | "tiff" | "h5"
        ) {
            continue;
        }
        let Some(stem) = Path::new(suffix).file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let endname = stem
            .strip_prefix("segm")
            .and_then(|rest| rest.strip_prefix('_'))
            .map(|rest| rest.to_string());
        segmentations.push(SegmentationAsset {
            name: stem.to_string(),
            endname,
            path,
        });
    }
    segmentations.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(segmentations)
}

fn looks_like_phase_channel(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ["phase", "bright", "bf", "dic", "pc"]
        .iter()
        .any(|token| lower.contains(token))
}

fn normalize_endname(endname: Option<&str>) -> Option<&str> {
    endname.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

fn extract_stack_frame_f32(
    values: &[f32],
    height: usize,
    width: usize,
    frames: usize,
    frame_index: usize,
) -> Result<FrameData<f32>> {
    if frame_index >= frames {
        bail!(
            "Requested frame {} but stack has {} frame(s)",
            frame_index,
            frames
        );
    }
    let plane_len = height * width;
    let start = frame_index * plane_len;
    let end = start + plane_len;
    Ok(FrameData {
        width,
        height,
        pixels: values[start..end].to_vec(),
    })
}

fn extract_stack_frame_u32(
    values: &[u32],
    height: usize,
    width: usize,
    frames: usize,
    frame_index: usize,
) -> Result<FrameData<u32>> {
    if frame_index >= frames {
        bail!(
            "Requested frame {} but stack has {} frame(s)",
            frame_index,
            frames
        );
    }
    let plane_len = height * width;
    let start = frame_index * plane_len;
    let end = start + plane_len;
    Ok(FrameData {
        width,
        height,
        pixels: values[start..end].to_vec(),
    })
}

fn extract_volume_frame_f32(
    values: &[f32],
    height: usize,
    width: usize,
    size_t: usize,
    size_z: usize,
    frame_index: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<f32>> {
    if frame_index >= size_t {
        bail!(
            "Requested frame {} but volume has {} timepoint(s)",
            frame_index,
            size_t
        );
    }
    let plane_len = height * width;
    let frame_offset = frame_index * size_z * plane_len;
    extract_oriented_frame_f32(
        &values[frame_offset..frame_offset + size_z * plane_len],
        size_z,
        height,
        width,
        view_plane,
        projection,
    )
}

fn extract_volume_frame_u32(
    values: &[u32],
    height: usize,
    width: usize,
    size_t: usize,
    size_z: usize,
    frame_index: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<u32>> {
    if frame_index >= size_t {
        bail!(
            "Requested frame {} but volume has {} timepoint(s)",
            frame_index,
            size_t
        );
    }
    let plane_len = height * width;
    let frame_offset = frame_index * size_z * plane_len;
    extract_oriented_frame_u32(
        &values[frame_offset..frame_offset + size_z * plane_len],
        size_z,
        height,
        width,
        view_plane,
        projection,
    )
}

fn extract_oriented_frame_f32(
    values: &[f32],
    size_z: usize,
    size_y: usize,
    size_x: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<f32>> {
    match view_plane {
        ViewPlane::XY => {
            let plane_len = size_y * size_x;
            let pixels = match projection {
                FrameProjection::Max => {
                    let mut projected = vec![f32::NEG_INFINITY; plane_len];
                    for z in 0..size_z {
                        let start = z * plane_len;
                        let end = start + plane_len;
                        for (dst, src) in projected.iter_mut().zip(&values[start..end]) {
                            if *src > *dst {
                                *dst = *src;
                            }
                        }
                    }
                    projected
                }
                FrameProjection::ZSlice(z_index) => {
                    if z_index >= size_z {
                        bail!("Requested z-slice {} but volume has {} slice(s)", z_index, size_z);
                    }
                    let start = z_index * plane_len;
                    values[start..start + plane_len].to_vec()
                }
            };
            Ok(FrameData {
                width: size_x,
                height: size_y,
                pixels,
            })
        }
        ViewPlane::XZ => {
            let slice_y = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= size_y {
                        bail!("Requested y-slice {} but volume has {} row(s)", index, size_y);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![f32::NEG_INFINITY; size_z * size_x];
            for z in 0..size_z {
                for x in 0..size_x {
                    let value = if let Some(y_index) = slice_y {
                        values[z * size_y * size_x + y_index * size_x + x]
                    } else {
                        let mut best = f32::NEG_INFINITY;
                        for y in 0..size_y {
                            best = best.max(values[z * size_y * size_x + y * size_x + x]);
                        }
                        best
                    };
                    pixels[z * size_x + x] = value;
                }
            }
            Ok(FrameData {
                width: size_x,
                height: size_z,
                pixels,
            })
        }
        ViewPlane::YZ => {
            let slice_x = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= size_x {
                        bail!("Requested x-slice {} but volume has {} column(s)", index, size_x);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![f32::NEG_INFINITY; size_z * size_y];
            for z in 0..size_z {
                for y in 0..size_y {
                    let value = if let Some(x_index) = slice_x {
                        values[z * size_y * size_x + y * size_x + x_index]
                    } else {
                        let mut best = f32::NEG_INFINITY;
                        for x in 0..size_x {
                            best = best.max(values[z * size_y * size_x + y * size_x + x]);
                        }
                        best
                    };
                    pixels[z * size_y + y] = value;
                }
            }
            Ok(FrameData {
                width: size_y,
                height: size_z,
                pixels,
            })
        }
    }
}

fn extract_oriented_frame_u32(
    values: &[u32],
    size_z: usize,
    size_y: usize,
    size_x: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<u32>> {
    match view_plane {
        ViewPlane::XY => {
            let plane_len = size_y * size_x;
            let pixels = match projection {
                FrameProjection::Max => {
                    let mut projected = vec![0u32; plane_len];
                    for z in 0..size_z {
                        let start = z * plane_len;
                        let end = start + plane_len;
                        for (dst, src) in projected.iter_mut().zip(&values[start..end]) {
                            if *src > *dst {
                                *dst = *src;
                            }
                        }
                    }
                    projected
                }
                FrameProjection::ZSlice(z_index) => {
                    if z_index >= size_z {
                        bail!("Requested z-slice {} but volume has {} slice(s)", z_index, size_z);
                    }
                    let start = z_index * plane_len;
                    values[start..start + plane_len].to_vec()
                }
            };
            Ok(FrameData {
                width: size_x,
                height: size_y,
                pixels,
            })
        }
        ViewPlane::XZ => {
            let slice_y = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= size_y {
                        bail!("Requested y-slice {} but volume has {} row(s)", index, size_y);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![0u32; size_z * size_x];
            for z in 0..size_z {
                for x in 0..size_x {
                    let value = if let Some(y_index) = slice_y {
                        values[z * size_y * size_x + y_index * size_x + x]
                    } else {
                        let mut best = 0u32;
                        for y in 0..size_y {
                            best = best.max(values[z * size_y * size_x + y * size_x + x]);
                        }
                        best
                    };
                    pixels[z * size_x + x] = value;
                }
            }
            Ok(FrameData {
                width: size_x,
                height: size_z,
                pixels,
            })
        }
        ViewPlane::YZ => {
            let slice_x = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= size_x {
                        bail!("Requested x-slice {} but volume has {} column(s)", index, size_x);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![0u32; size_z * size_y];
            for z in 0..size_z {
                for y in 0..size_y {
                    let value = if let Some(x_index) = slice_x {
                        values[z * size_y * size_x + y * size_x + x_index]
                    } else {
                        let mut best = 0u32;
                        for x in 0..size_x {
                            best = best.max(values[z * size_y * size_x + y * size_x + x]);
                        }
                        best
                    };
                    pixels[z * size_y + y] = value;
                }
            }
            Ok(FrameData {
                width: size_y,
                height: size_z,
                pixels,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use ndarray_npy::NpzWriter;
    use std::fs::{self, File};
    use tempfile::tempdir;
    use tiff::encoder::{colortype, TiffEncoder};

    #[test]
    fn opens_position_session_and_discovers_segmentations() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;

        write_stack(&images.join("demo_phase.tif"), &[1, 2])?;
        write_stack(&images.join("demo_fluo.tif"), &[3, 4])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        write_mask(
            &images.join("demo_segm.npz"),
            Array3::from_shape_vec((2, 1, 1), vec![0u32, 1u32])?,
        )?;
        write_mask(
            &images.join("demo_segm_rust.npz"),
            Array3::from_shape_vec((2, 1, 1), vec![1u32, 2u32])?,
        )?;

        let session = open_position_session(&position)?;
        assert_eq!(session.channel_names(), vec!["fluo", "phase"]);
        assert_eq!(session.segmentations.len(), 2);
        assert_eq!(session.segmentations[0].name, "segm");
        assert_eq!(session.segmentations[1].endname.as_deref(), Some("rust"));
        Ok(())
    }

    #[test]
    fn loads_frame_pixels_for_selected_channel_and_segmentation() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;

        write_stack(&images.join("demo_phase.tif"), &[7, 9])?;
        write_stack(&images.join("demo_fluo.tif"), &[11, 13])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        write_mask(
            &images.join("demo_segm.npz"),
            Array3::from_shape_vec((2, 1, 1), vec![4u32, 6u32])?,
        )?;

        let session = open_position_session(&position)?;
        let frame = session.load_channel_frame("phase", 1, FrameProjection::Max)?;
        assert_eq!(frame.pixels, vec![9.0]);

        let segm = session
            .load_segmentation_frame(None, 1, FrameProjection::Max)?
            .expect("expected segmentation frame");
        assert_eq!(segm.pixels, vec![6]);
        Ok(())
    }

    #[test]
    fn discovers_new_segmentation_versions_after_save_as() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_3");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;

        write_stack(&images.join("demo_phase.tif"), &[7, 9])?;
        write_stack(&images.join("demo_fluo.tif"), &[11, 13])?;
        fs::write(
            images.join("demo_metadata.csv"),
            "Description,values\nbasename,demo_\nSizeT,2\nSizeZ,1\n",
        )?;
        write_mask(
            &images.join("demo_segm.npz"),
            Array3::from_shape_vec((2, 1, 1), vec![4u32, 6u32])?,
        )?;
        write_mask(
            &images.join("demo_segm_corrected.npz"),
            Array3::from_shape_vec((2, 1, 1), vec![8u32, 9u32])?,
        )?;

        let session = open_position_session(&position)?;
        assert_eq!(session.segmentations.len(), 2);
        assert_eq!(
            session.segmentations[1].endname.as_deref(),
            Some("corrected")
        );
        Ok(())
    }

    fn write_stack(path: &Path, frames: &[u8]) -> Result<()> {
        let file = File::create(path)?;
        let mut encoder = TiffEncoder::new(file)?;
        for value in frames {
            encoder
                .new_image::<colortype::Gray8>(1, 1)?
                .write_data(&[*value])?;
        }
        Ok(())
    }

    fn write_mask(path: &Path, array: Array3<u32>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = NpzWriter::new_compressed(file);
        writer.add_array("arr_0", &array)?;
        writer.finish()?;
        Ok(())
    }
}
