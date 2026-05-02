use anyhow::{Context, Result};
use csv::{ReaderBuilder, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_TIME_INCREMENT: f64 = 1.0;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetadataSummary {
    pub basename: Option<String>,
    pub size_t: Option<usize>,
    pub size_z: Option<usize>,
    pub time_increment: Option<f64>,
    pub physical_size_z: Option<f64>,
    pub physical_size_x: Option<f64>,
    pub physical_size_y: Option<f64>,
    pub segm_is_3d: BTreeMap<String, bool>,
}

pub fn read_metadata_map(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open metadata file {}", path.display()))?;
    let first_header = reader
        .headers()
        .ok()
        .and_then(|headers| headers.get(0).map(str::to_string));

    for record in reader.records() {
        let record = record?;
        let key = record.get(0).unwrap_or_default().trim();
        if key.is_empty() {
            continue;
        }
        if first_header.as_deref() == Some(key) {
            break;
        }
        let value = record
            .iter()
            .skip(1)
            .collect::<Vec<_>>()
            .join(",")
            .trim()
            .to_string();
        map.insert(key.to_string(), value);
    }

    Ok(map)
}

pub fn read_metadata_summary(path: &Path) -> Result<MetadataSummary> {
    let values = read_metadata_map(path)?;
    Ok(MetadataSummary {
        basename: values
            .get("basename")
            .cloned()
            .filter(|value| !value.is_empty()),
        size_t: parse_optional_usize(values.get("SizeT"))?,
        size_z: parse_optional_usize(values.get("SizeZ"))?,
        time_increment: parse_optional_f64(values.get("TimeIncrement"))?,
        physical_size_z: parse_optional_f64(values.get("PhysicalSizeZ"))?,
        physical_size_x: parse_optional_f64(values.get("PhysicalSizeX"))?,
        physical_size_y: parse_optional_f64(values.get("PhysicalSizeY"))?,
        segm_is_3d: parse_segm_is_3d_map(&values),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn ensure_position_metadata(
    existing_path: Option<&Path>,
    images_dir: &Path,
    basename: &str,
    phase_channel: &str,
    fluo_channel: &str,
    frames: usize,
    size_z: usize,
    height: usize,
    width: usize,
    time_increment: f64,
    physical_size_z: f64,
    physical_size_y: f64,
    physical_size_x: f64,
    segm_endname: &str,
    is_segm_3d: bool,
) -> Result<PathBuf> {
    let target_path = existing_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| images_dir.join(format!("{basename}metadata.csv")));

    let mut values = if target_path.exists() {
        read_metadata_map(&target_path)?
    } else {
        BTreeMap::new()
    };

    values.insert("basename".into(), basename.to_string());
    values.insert("SizeT".into(), frames.to_string());
    values.insert("SizeZ".into(), size_z.to_string());
    values.insert("SizeY".into(), height.to_string());
    values.insert("SizeX".into(), width.to_string());
    values.insert("TimeIncrement".into(), time_increment.to_string());
    values.insert("channel_0_name".into(), phase_channel.to_string());
    values.insert("channel_1_name".into(), fluo_channel.to_string());
    values.insert(
        format!("{segm_endname}_isSegm3D"),
        if is_segm_3d { "True" } else { "False" }.into(),
    );
    values.insert("PhysicalSizeZ".into(), physical_size_z.to_string());
    values.insert("PhysicalSizeY".into(), physical_size_y.to_string());
    values.insert("PhysicalSizeX".into(), physical_size_x.to_string());

    let mut writer = Writer::from_path(&target_path)
        .with_context(|| format!("Failed to create {}", target_path.display()))?;
    writer.write_record(["Description", "values"])?;

    let ordered_keys = vec![
        "basename".to_string(),
        "SizeT".to_string(),
        "SizeZ".to_string(),
        "SizeY".to_string(),
        "SizeX".to_string(),
        "TimeIncrement".to_string(),
        "PhysicalSizeZ".to_string(),
        "PhysicalSizeY".to_string(),
        "PhysicalSizeX".to_string(),
        "channel_0_name".to_string(),
        "channel_1_name".to_string(),
        format!("{segm_endname}_isSegm3D"),
    ];

    for key in ordered_keys {
        if let Some(value) = values.remove(&key) {
            writer.write_record([key.as_str(), value.as_str()])?;
        }
    }

    for (key, value) in values {
        writer.write_record([key.as_str(), value.as_str()])?;
    }

    writer.flush()?;
    Ok(target_path)
}

fn parse_segm_is_3d_map(values: &BTreeMap<String, String>) -> BTreeMap<String, bool> {
    values
        .iter()
        .filter_map(|(key, value)| {
            key.strip_suffix("_isSegm3D")
                .map(|segm_name| (segm_name.to_string(), parse_bool_like(value)))
        })
        .collect()
}

fn parse_bool_like(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn parse_optional_usize(value: Option<&String>) -> Result<Option<usize>> {
    match value {
        Some(raw) if !raw.trim().is_empty() => raw
            .parse::<f64>()
            .map(|value| Some(value as usize))
            .with_context(|| format!("Failed to parse metadata value {raw:?} as usize")),
        _ => Ok(None),
    }
}

fn parse_optional_f64(value: Option<&String>) -> Result<Option<f64>> {
    match value {
        Some(raw) if !raw.trim().is_empty() => raw
            .parse::<f64>()
            .map(Some)
            .with_context(|| format!("Failed to parse metadata value {raw:?} as f64")),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn reads_metadata_summary_fields() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_metadata.csv");
        let mut file = fs::File::create(&path)?;
        writeln!(file, "Description,values")?;
        writeln!(file, "basename,demo_")?;
        writeln!(file, "SizeT,3")?;
        writeln!(file, "SizeZ,1")?;
        writeln!(file, "TimeIncrement,12.5")?;
        writeln!(file, "PhysicalSizeZ,0.75")?;
        writeln!(file, "PhysicalSizeX,0.25")?;
        writeln!(file, "PhysicalSizeY,0.5")?;
        writeln!(file, "segm_isSegm3D,False")?;
        writeln!(file, "segm_nuclei_isSegm3D,True")?;

        let summary = read_metadata_summary(&path)?;
        assert_eq!(summary.basename.as_deref(), Some("demo_"));
        assert_eq!(summary.size_t, Some(3));
        assert_eq!(summary.size_z, Some(1));
        assert_eq!(summary.time_increment, Some(12.5));
        assert_eq!(summary.physical_size_z, Some(0.75));
        assert_eq!(summary.physical_size_x, Some(0.25));
        assert_eq!(summary.physical_size_y, Some(0.5));
        assert_eq!(summary.segm_is_3d.get("segm"), Some(&false));
        assert_eq!(summary.segm_is_3d.get("segm_nuclei"), Some(&true));
        Ok(())
    }

    #[test]
    fn ignores_duplicate_appended_metadata_after_repeated_header() -> Result<()> {
        let temp = tempdir()?;
        let path = temp.path().join("demo_metadata.csv");
        let mut file = fs::File::create(&path)?;
        writeln!(file, "Description,values")?;
        writeln!(file, "basename,demo_")?;
        writeln!(file, "SizeT,2")?;
        writeln!(file, "Description,values")?;
        writeln!(file, "basename,stale_")?;
        writeln!(file, "SizeT,9")?;

        let summary = read_metadata_summary(&path)?;

        assert_eq!(summary.basename.as_deref(), Some("demo_"));
        assert_eq!(summary.size_t, Some(2));
        Ok(())
    }

    #[test]
    fn creates_or_updates_required_metadata_fields() -> Result<()> {
        let temp = tempdir()?;
        let images_dir = temp.path();
        let path = ensure_position_metadata(
            None, images_dir, "demo_", "phase", "fluo", 4, 3, 32, 24, 15.0, 1.5, 0.5, 0.25, "segm",
            true,
        )?;

        let values = read_metadata_map(&path)?;
        assert_eq!(values.get("SizeT").map(String::as_str), Some("4"));
        assert_eq!(values.get("SizeZ").map(String::as_str), Some("3"));
        assert_eq!(values.get("TimeIncrement").map(String::as_str), Some("15"));
        assert_eq!(
            values.get("channel_0_name").map(String::as_str),
            Some("phase")
        );
        assert_eq!(
            values.get("segm_isSegm3D").map(String::as_str),
            Some("True")
        );
        Ok(())
    }
}
