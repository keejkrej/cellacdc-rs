use anyhow::{Context, Result};
use csv::{Reader, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_TIME_INCREMENT: f64 = 1.0;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetadataSummary {
    pub basename: Option<String>,
    pub size_t: Option<usize>,
    pub size_z: Option<usize>,
    pub time_increment: Option<f64>,
}

pub fn read_metadata_map(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    let mut reader = Reader::from_path(path)
        .with_context(|| format!("Failed to open metadata file {}", path.display()))?;

    for record in reader.records() {
        let record = record?;
        let key = record.get(0).unwrap_or_default().trim();
        if key.is_empty() {
            continue;
        }
        let value = record.get(1).unwrap_or_default().trim().to_string();
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
    })
}

pub fn ensure_position_metadata(
    existing_path: Option<&Path>,
    images_dir: &Path,
    basename: &str,
    phase_channel: &str,
    fluo_channel: &str,
    frames: usize,
    height: usize,
    width: usize,
    time_increment: f64,
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
    values.insert("SizeZ".into(), "1".into());
    values.insert("SizeY".into(), height.to_string());
    values.insert("SizeX".into(), width.to_string());
    values.insert("TimeIncrement".into(), time_increment.to_string());
    values.insert("channel_0_name".into(), phase_channel.to_string());
    values.insert("channel_1_name".into(), fluo_channel.to_string());
    values
        .entry("PhysicalSizeZ".into())
        .or_insert_with(|| "1.0".into());
    values
        .entry("PhysicalSizeY".into())
        .or_insert_with(|| "1.0".into());
    values
        .entry("PhysicalSizeX".into())
        .or_insert_with(|| "1.0".into());

    let mut writer = Writer::from_path(&target_path)
        .with_context(|| format!("Failed to create {}", target_path.display()))?;
    writer.write_record(["Description", "values"])?;

    for key in [
        "basename",
        "SizeT",
        "SizeZ",
        "SizeY",
        "SizeX",
        "TimeIncrement",
        "PhysicalSizeZ",
        "PhysicalSizeY",
        "PhysicalSizeX",
        "channel_0_name",
        "channel_1_name",
    ] {
        if let Some(value) = values.remove(key) {
            writer.write_record([key, value.as_str()])?;
        }
    }

    for (key, value) in values {
        writer.write_record([key.as_str(), value.as_str()])?;
    }

    writer.flush()?;
    Ok(target_path)
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

        let summary = read_metadata_summary(&path)?;
        assert_eq!(summary.basename.as_deref(), Some("demo_"));
        assert_eq!(summary.size_t, Some(3));
        assert_eq!(summary.size_z, Some(1));
        assert_eq!(summary.time_increment, Some(12.5));
        Ok(())
    }

    #[test]
    fn creates_or_updates_required_metadata_fields() -> Result<()> {
        let temp = tempdir()?;
        let images_dir = temp.path();
        let path =
            ensure_position_metadata(None, images_dir, "demo_", "phase", "fluo", 4, 32, 24, 15.0)?;

        let values = read_metadata_map(&path)?;
        assert_eq!(values.get("SizeT").map(String::as_str), Some("4"));
        assert_eq!(values.get("TimeIncrement").map(String::as_str), Some("15"));
        assert_eq!(
            values.get("channel_0_name").map(String::as_str),
            Some("phase")
        );
        Ok(())
    }
}
