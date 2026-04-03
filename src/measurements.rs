use csv::Writer;
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementRow {
    pub frame_i: usize,
    pub cell_id: u32,
    pub time_seconds: f64,
    pub is_cell_dead: u8,
    pub is_cell_excluded: u8,
    pub was_manually_edited: u8,
    pub x_centroid: i32,
    pub y_centroid: i32,
    pub cell_area_pxl: u32,
}

#[derive(Debug, Default, Clone)]
struct RegionAccumulator {
    area: u32,
    sum_x: u64,
    sum_y: u64,
}

pub fn rows_from_mask(
    masks: &[u32],
    height: usize,
    width: usize,
    frame_i: usize,
    time_seconds: f64,
) -> Vec<MeasurementRow> {
    let mut accumulators = BTreeMap::<u32, RegionAccumulator>::new();

    for y in 0..height {
        for x in 0..width {
            let label = masks[y * width + x];
            if label == 0 {
                continue;
            }
            let entry = accumulators.entry(label).or_default();
            entry.area += 1;
            entry.sum_x += x as u64;
            entry.sum_y += y as u64;
        }
    }

    accumulators
        .into_iter()
        .map(|(cell_id, region)| MeasurementRow {
            frame_i,
            cell_id,
            time_seconds,
            is_cell_dead: 0,
            is_cell_excluded: 0,
            was_manually_edited: 0,
            x_centroid: (region.sum_x / region.area as u64) as i32,
            y_centroid: (region.sum_y / region.area as u64) as i32,
            cell_area_pxl: region.area,
        })
        .collect()
}

pub fn write_acdc_output_csv(path: &Path, rows: &[MeasurementRow]) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("Failed to create {}", path.display()))?;
    writer.write_record([
        "frame_i",
        "time_seconds",
        "Cell_ID",
        "is_cell_dead",
        "is_cell_excluded",
        "x_centroid",
        "y_centroid",
        "cell_area_pxl",
        "was_manually_edited",
    ])?;

    for row in rows {
        writer.write_record([
            row.frame_i.to_string(),
            row.time_seconds.to_string(),
            row.cell_id.to_string(),
            row.is_cell_dead.to_string(),
            row.is_cell_excluded.to_string(),
            row.x_centroid.to_string(),
            row.y_centroid.to_string(),
            row.cell_area_pxl.to_string(),
            row.was_manually_edited.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_centroids_and_area() {
        let mask = vec![
            0, 1, 1, //
            0, 0, 2, //
            2, 2, 2, //
        ];
        let rows = rows_from_mask(&mask, 3, 3, 0, 12.0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cell_id, 1);
        assert_eq!(rows[0].cell_area_pxl, 2);
        assert_eq!(rows[0].time_seconds, 12.0);
        assert_eq!(rows[1].cell_id, 2);
        assert_eq!(rows[1].cell_area_pxl, 4);
    }
}
