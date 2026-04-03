use csv::Writer;
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementRow {
    pub frame_i: usize,
    pub cell_id: u32,
    pub time_seconds: f64,
    pub time_minutes: f64,
    pub time_hours: f64,
    pub is_cell_dead: u8,
    pub is_cell_excluded: u8,
    pub was_manually_edited: u8,
    pub x_centroid: i32,
    pub y_centroid: i32,
    pub cell_area_pxl: u32,
    pub cell_area_um2: f64,
    pub disappears_before_end: Option<u8>,
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
    pixel_area_um2: f64,
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
            time_minutes: time_seconds / 60.0,
            time_hours: time_seconds / 3600.0,
            is_cell_dead: 0,
            is_cell_excluded: 0,
            was_manually_edited: 0,
            x_centroid: (region.sum_x / region.area as u64) as i32,
            y_centroid: (region.sum_y / region.area as u64) as i32,
            cell_area_pxl: region.area,
            cell_area_um2: region.area as f64 * pixel_area_um2,
            disappears_before_end: None,
        })
        .collect()
}

pub fn write_acdc_output_csv(path: &Path, rows: &[MeasurementRow]) -> Result<()> {
    let include_disappears = rows.iter().any(|row| row.disappears_before_end.is_some());
    let mut writer =
        Writer::from_path(path).with_context(|| format!("Failed to create {}", path.display()))?;

    let mut header = vec![
        "frame_i".to_string(),
        "Cell_ID".to_string(),
        "time_seconds".to_string(),
        "time_minutes".to_string(),
        "time_hours".to_string(),
        "is_cell_dead".to_string(),
        "is_cell_excluded".to_string(),
        "was_manually_edited".to_string(),
        "x_centroid".to_string(),
        "y_centroid".to_string(),
        "cell_area_pxl".to_string(),
        "cell_area_um2".to_string(),
    ];
    if include_disappears {
        header.push("disappears_before_end".to_string());
    }
    writer.write_record(header)?;

    for row in rows {
        let mut record = vec![
            row.frame_i.to_string(),
            row.cell_id.to_string(),
            row.time_seconds.to_string(),
            row.time_minutes.to_string(),
            row.time_hours.to_string(),
            row.is_cell_dead.to_string(),
            row.is_cell_excluded.to_string(),
            row.was_manually_edited.to_string(),
            row.x_centroid.to_string(),
            row.y_centroid.to_string(),
            row.cell_area_pxl.to_string(),
            row.cell_area_um2.to_string(),
        ];
        if include_disappears {
            record.push(row.disappears_before_end.unwrap_or(0).to_string());
        }
        writer.write_record(record)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_centroids_area_and_time_columns() {
        let mask = vec![
            0, 1, 1, //
            0, 0, 2, //
            2, 2, 2, //
        ];
        let rows = rows_from_mask(&mask, 3, 3, 0, 12.0, 0.25);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cell_id, 1);
        assert_eq!(rows[0].cell_area_pxl, 2);
        assert_eq!(rows[0].cell_area_um2, 0.5);
        assert_eq!(rows[0].time_seconds, 12.0);
        assert_eq!(rows[0].time_minutes, 0.2);
        assert_eq!(rows[1].cell_id, 2);
        assert_eq!(rows[1].cell_area_pxl, 4);
    }
}
