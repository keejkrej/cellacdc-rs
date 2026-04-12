use crate::session::{open_position_session, FrameProjection};
use crate::tabular::read_table;
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameInspectionConfig {
    pub position_path: PathBuf,
    pub segm_endname: Option<String>,
    pub frame_index: usize,
    pub projection: FrameProjection,
    pub selected_label: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectMeasurementSummary {
    pub label: u32,
    pub area_pixels: usize,
    pub area_um2: f64,
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub bbox_min_x: usize,
    pub bbox_min_y: usize,
    pub bbox_max_x: usize,
    pub bbox_max_y: usize,
    pub channel_mean: BTreeMap<String, f64>,
    pub channel_sum: BTreeMap<String, f64>,
    pub cell_cycle_stage: Option<String>,
    pub generation_num: Option<i64>,
    pub relative_id: Option<i64>,
    pub relationship: Option<String>,
    pub is_history_known: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameInspection {
    pub frame_index: usize,
    pub time_seconds: f64,
    pub object_count: usize,
    pub available_labels: Vec<u32>,
    pub selected_object: Option<ObjectMeasurementSummary>,
}

#[derive(Debug, Clone)]
struct FrameRegion {
    label: u32,
    area: usize,
    pixels: Vec<(usize, usize)>,
    bbox_min_x: usize,
    bbox_min_y: usize,
    bbox_max_x: usize,
    bbox_max_y: usize,
    centroid_x: f64,
    centroid_y: f64,
}

pub fn inspect_position_frame(config: FrameInspectionConfig) -> Result<FrameInspection> {
    let position = open_position_session(&config.position_path)?;
    let segmentation = position
        .load_segmentation_frame(
            config.segm_endname.as_deref(),
            config.frame_index,
            config.projection,
        )?
        .ok_or_else(|| anyhow!("No segmentation is available for the selected frame"))?;
    let regions = extract_regions(&segmentation.pixels, segmentation.height, segmentation.width);
    let available_labels = regions.iter().map(|region| region.label).collect::<Vec<_>>();

    let selected_object = if let Some(selected_label) = config.selected_label {
        let region = regions
            .iter()
            .find(|region| region.label == selected_label)
            .cloned();
        if let Some(region) = region {
            let mut channel_mean = BTreeMap::new();
            let mut channel_sum = BTreeMap::new();
            for channel in position.channel_names() {
                let frame = position.load_channel_frame(
                    &channel,
                    config.frame_index,
                    config.projection,
                )?;
                let mut sum = 0.0f64;
                for &(x, y) in &region.pixels {
                    sum += frame.pixels[y * frame.width + x] as f64;
                }
                let mean = if region.area > 0 {
                    sum / region.area as f64
                } else {
                    0.0
                };
                channel_sum.insert(channel.clone(), sum);
                channel_mean.insert(channel, mean);
            }
            let cca = load_cell_cycle_fields(
                &position.spec.images_dir,
                &position.spec.basename,
                config.segm_endname.as_deref(),
                config.frame_index,
                selected_label,
            )?;
            Some(ObjectMeasurementSummary {
                label: region.label,
                area_pixels: region.area,
                area_um2: region.area as f64
                    * position.spec.physical_size_x
                    * position.spec.physical_size_y,
                centroid_x: region.centroid_x,
                centroid_y: region.centroid_y,
                bbox_min_x: region.bbox_min_x,
                bbox_min_y: region.bbox_min_y,
                bbox_max_x: region.bbox_max_x,
                bbox_max_y: region.bbox_max_y,
                channel_mean,
                channel_sum,
                cell_cycle_stage: cca.as_ref().map(|value| value.cell_cycle_stage.clone()),
                generation_num: cca.as_ref().map(|value| value.generation_num),
                relative_id: cca.as_ref().map(|value| value.relative_id),
                relationship: cca.as_ref().map(|value| value.relationship.clone()),
                is_history_known: cca.as_ref().map(|value| value.is_history_known),
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(FrameInspection {
        frame_index: config.frame_index,
        time_seconds: position.spec.time_increment * config.frame_index as f64,
        object_count: regions.len(),
        available_labels,
        selected_object,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellCycleInspection {
    cell_cycle_stage: String,
    generation_num: i64,
    relative_id: i64,
    relationship: String,
    is_history_known: bool,
}

fn load_cell_cycle_fields(
    images_dir: &std::path::Path,
    basename: &str,
    segm_endname: Option<&str>,
    frame_index: usize,
    label: u32,
) -> Result<Option<CellCycleInspection>> {
    let suffix = segm_endname
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!("_{value}"))
        .unwrap_or_default();
    let path = images_dir.join(format!("{basename}acdc_output{suffix}.csv"));
    if !path.exists() {
        return Ok(None);
    }
    let table = read_table(&path)?;
    let frame_col = table.header_index("frame_i")?;
    let id_col = table.header_index("Cell_ID")?;
    let stage_col = table.header_index("cell_cycle_stage")?;
    let generation_col = table.header_index("generation_num")?;
    let relative_col = table.header_index("relative_ID")?;
    let relationship_col = table.header_index("relationship")?;
    let history_col = table.header_index("is_history_known")?;
    for row in &table.rows {
        if row[frame_col].as_i64() != Some(frame_index as i64) || row[id_col].as_i64() != Some(label as i64) {
            continue;
        }
        return Ok(Some(CellCycleInspection {
            cell_cycle_stage: row[stage_col].as_string_lossy(),
            generation_num: row[generation_col].as_i64().unwrap_or(-1),
            relative_id: row[relative_col].as_i64().unwrap_or(-1),
            relationship: row[relationship_col].as_string_lossy(),
            is_history_known: match &row[history_col] {
                crate::tabular::TableValue::Bool(value) => *value,
                crate::tabular::TableValue::Number(value) => *value != 0.0,
                crate::tabular::TableValue::Text(value) => {
                    matches!(value.to_ascii_lowercase().as_str(), "true" | "1")
                }
                crate::tabular::TableValue::Empty => false,
            },
        }));
    }
    Ok(None)
}

fn extract_regions(mask_frame: &[u32], height: usize, width: usize) -> Vec<FrameRegion> {
    #[derive(Debug, Clone)]
    struct RegionAccumulator {
        area: usize,
        pixels: Vec<(usize, usize)>,
        min_x: usize,
        min_y: usize,
        max_x: usize,
        max_y: usize,
        sum_x: f64,
        sum_y: f64,
    }

    let mut accumulators = BTreeMap::<u32, RegionAccumulator>::new();
    for y in 0..height {
        for x in 0..width {
            let label = mask_frame[y * width + x];
            if label == 0 {
                continue;
            }
            let entry = accumulators.entry(label).or_insert_with(|| RegionAccumulator {
                area: 0,
                pixels: Vec::new(),
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                sum_x: 0.0,
                sum_y: 0.0,
            });
            entry.area += 1;
            entry.pixels.push((x, y));
            entry.min_x = entry.min_x.min(x);
            entry.min_y = entry.min_y.min(y);
            entry.max_x = entry.max_x.max(x);
            entry.max_y = entry.max_y.max(y);
            entry.sum_x += x as f64;
            entry.sum_y += y as f64;
        }
    }

    accumulators
        .into_iter()
        .map(|(label, acc)| FrameRegion {
            label,
            area: acc.area,
            pixels: acc.pixels,
            bbox_min_x: acc.min_x,
            bbox_min_y: acc.min_y,
            bbox_max_x: acc.max_x,
            bbox_max_y: acc.max_y,
            centroid_x: acc.sum_x / acc.area as f64,
            centroid_y: acc.sum_y / acc.area as f64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_regions_and_centroids() {
        let regions = extract_regions(
            &[
                0, 1, 1, //
                0, 0, 2, //
                3, 3, 2, //
            ],
            3,
            3,
        );
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].label, 1);
        assert_eq!(regions[0].area, 2);
        assert!(regions[0].centroid_x > 1.0);
    }
}
