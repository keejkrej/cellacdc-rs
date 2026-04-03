use anyhow::{bail, Result};
use std::collections::BTreeMap;

use crate::image_io::VolumeShape;
use crate::segm_info::ZProjectionMode;

pub fn project_frame_f32(
    values: &[f32],
    shape: VolumeShape,
    frame_i: usize,
    z_slice: usize,
    mode: ZProjectionMode,
) -> Result<Vec<f32>> {
    if frame_i >= shape.size_t {
        bail!(
            "Frame index {} exceeds available frames {}",
            frame_i,
            shape.size_t
        );
    }
    if z_slice >= shape.size_z {
        bail!(
            "z-slice index {} exceeds available depth {}",
            z_slice,
            shape.size_z
        );
    }

    let plane_len = shape.height * shape.width;
    let frame_len = plane_len * shape.size_z;
    let frame_start = frame_i * frame_len;
    let frame = &values[frame_start..frame_start + frame_len];

    match mode {
        ZProjectionMode::SingleZSlice => {
            let start = z_slice * plane_len;
            Ok(frame[start..start + plane_len].to_vec())
        }
        ZProjectionMode::MaxZProjection => {
            let mut projected = vec![f32::NEG_INFINITY; plane_len];
            for z in 0..shape.size_z {
                let start = z * plane_len;
                for (idx, value) in frame[start..start + plane_len].iter().enumerate() {
                    projected[idx] = projected[idx].max(*value);
                }
            }
            Ok(projected)
        }
        ZProjectionMode::MeanZProjection => {
            let mut projected = vec![0.0; plane_len];
            for z in 0..shape.size_z {
                let start = z * plane_len;
                for (idx, value) in frame[start..start + plane_len].iter().enumerate() {
                    projected[idx] += *value;
                }
            }
            let denom = shape.size_z as f32;
            for value in &mut projected {
                *value /= denom;
            }
            Ok(projected)
        }
        ZProjectionMode::MedianZProjection => {
            let mut projected = vec![0.0; plane_len];
            let mut column = vec![0.0; shape.size_z];
            for idx in 0..plane_len {
                for z in 0..shape.size_z {
                    column[z] = frame[z * plane_len + idx];
                }
                column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                projected[idx] = column[shape.size_z / 2];
            }
            Ok(projected)
        }
    }
}

pub fn project_mask_volume_max(
    values: &[u32],
    size_z: usize,
    height: usize,
    width: usize,
) -> Vec<u32> {
    let plane_len = height * width;
    let mut projected = vec![0; plane_len];
    for idx in 0..plane_len {
        let mut label = 0;
        for z in 0..size_z {
            let value = values[z * plane_len + idx];
            if value != 0 {
                label = value;
            }
        }
        projected[idx] = label;
    }
    projected
}

pub fn count_mask_volume_labels(values: &[u32]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    for label in values.iter().copied().filter(|label| *label != 0) {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

pub fn connect_3d_lab_z_boundaries(
    lab: &[u32],
    size_z: usize,
    height: usize,
    width: usize,
) -> Vec<u32> {
    let plane_len = height * width;
    let mut pixels = BTreeMap::<u32, Vec<(usize, usize, usize)>>::new();
    for z in 0..size_z {
        let plane = &lab[z * plane_len..(z + 1) * plane_len];
        for y in 0..height {
            for x in 0..width {
                let label = plane[y * width + x];
                if label != 0 {
                    pixels.entry(label).or_default().push((z, y, x));
                }
            }
        }
    }

    let mut connected = vec![0; lab.len()];
    for (label, voxels) in pixels {
        let mut by_z = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (z, y, x) in voxels {
            by_z.entry(z).or_default().push((y, x));
        }
        if by_z.len() <= 1 {
            for (z, points) in by_z {
                for (y, x) in points {
                    connected[z * plane_len + y * width + x] = label;
                }
            }
            continue;
        }

        let z_keys = by_z.keys().copied().collect::<Vec<_>>();
        let center_z = z_keys[z_keys.len() / 2];
        let center_layer = by_z.get(&center_z).unwrap();
        let z_min = *z_keys.first().unwrap();
        let z_max = *z_keys.last().unwrap();
        for z in z_min..=z_max {
            for (y, x) in center_layer {
                connected[z * plane_len + y * width + x] = label;
            }
        }
    }
    connected
}

pub fn stack_2d_lab_to_3d(lab: &[u32], size_z: usize) -> Vec<u32> {
    let mut stacked = Vec::with_capacity(lab.len() * size_z);
    for _ in 0..size_z {
        stacked.extend_from_slice(lab);
    }
    stacked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_mean_and_single_slice() -> Result<()> {
        let shape = VolumeShape {
            size_t: 1,
            size_z: 2,
            height: 1,
            width: 2,
        };
        let values = vec![1.0, 3.0, 5.0, 7.0];
        assert_eq!(
            project_frame_f32(&values, shape, 0, 1, ZProjectionMode::SingleZSlice)?,
            vec![5.0, 7.0]
        );
        assert_eq!(
            project_frame_f32(&values, shape, 0, 0, ZProjectionMode::MeanZProjection)?,
            vec![3.0, 5.0]
        );
        Ok(())
    }

    #[test]
    fn stacks_2d_mask_to_3d() {
        let stacked = stack_2d_lab_to_3d(&[0, 1, 2, 3], 3);
        assert_eq!(stacked.len(), 12);
        assert_eq!(&stacked[4..8], &[0, 1, 2, 3]);
    }

    #[test]
    fn connects_objects_between_z_boundaries() {
        let lab = vec![
            0, 1, //
            0, 0, //
            0, 0, //
            0, 1, //
        ];
        let connected = connect_3d_lab_z_boundaries(&lab, 2, 2, 2);
        assert_eq!(connected, vec![0, 0, 0, 1, 0, 0, 0, 1]);
    }
}
