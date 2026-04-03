use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionSpec {
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub basename: String,
    pub phase_channel: String,
    pub fluo_channel: String,
    pub phase_image: PathBuf,
    pub fluo_image: PathBuf,
    pub metadata_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentSpec {
    pub experiment_dir: PathBuf,
    pub positions: Vec<PositionSpec>,
}

pub fn resolve_position(
    path: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
) -> Result<PositionSpec> {
    let phase_channel = phase_channel.into();
    let fluo_channel = fluo_channel.into();

    let input = path.as_ref();
    let (position_dir, images_dir) = normalize_position_path(input)?;
    let files = list_dir_sorted(&images_dir)?;

    let metadata_path = files
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("metadata.csv"))
                .unwrap_or(false)
        })
        .cloned();

    let basename = metadata_path
        .as_ref()
        .map(|path| read_basename_from_metadata(path))
        .transpose()?
        .flatten()
        .or_else(|| infer_basename(&files))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Failed to determine Cell-ACDC basename in {}",
                images_dir.display()
            )
        })?;

    let phase_image = find_channel_file(&files, &basename, &phase_channel)
        .with_context(|| format!("Failed to resolve phase channel \"{phase_channel}\""))?;
    let fluo_image = find_channel_file(&files, &basename, &fluo_channel)
        .with_context(|| format!("Failed to resolve fluorescence channel \"{fluo_channel}\""))?;

    Ok(PositionSpec {
        position_dir,
        images_dir,
        basename,
        phase_channel,
        fluo_channel,
        phase_image,
        fluo_image,
        metadata_path,
    })
}

pub fn discover_experiment(
    experiment_dir: impl AsRef<Path>,
    phase_channel: impl Into<String>,
    fluo_channel: impl Into<String>,
) -> Result<ExperimentSpec> {
    let experiment_dir = experiment_dir.as_ref().to_path_buf();
    if !experiment_dir.is_dir() {
        bail!(
            "Experiment path is not a directory: {}",
            experiment_dir.display()
        );
    }

    let phase_channel = phase_channel.into();
    let fluo_channel = fluo_channel.into();
    let mut positions = Vec::new();

    for entry in fs::read_dir(&experiment_dir)
        .with_context(|| format!("Failed to read {}", experiment_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let file_name = match path.file_name().and_then(|name| name.to_str()) {
            Some(value) => value,
            None => continue,
        };

        if !file_name.starts_with("Position_") {
            continue;
        }

        positions.push(resolve_position(
            &path,
            phase_channel.clone(),
            fluo_channel.clone(),
        )?);
    }

    positions.sort_by(|a, b| a.position_dir.cmp(&b.position_dir));
    if positions.is_empty() {
        bail!(
            "No Cell-ACDC positions found under {}",
            experiment_dir.display()
        );
    }

    Ok(ExperimentSpec {
        experiment_dir,
        positions,
    })
}

fn normalize_position_path(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if !path.exists() {
        bail!("Path does not exist: {}", path.display());
    }

    if path.file_name().and_then(|name| name.to_str()) == Some("Images") {
        let position_dir = path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("Images directory has no parent: {}", path.display()))?;
        return Ok((position_dir, path.to_path_buf()));
    }

    let images_dir = path.join("Images");
    if images_dir.is_dir() {
        return Ok((path.to_path_buf(), images_dir));
    }

    bail!(
        "Expected a Cell-ACDC position directory or Images directory, got {}",
        path.display()
    )
}

fn list_dir_sorted(path: &Path) -> Result<Vec<PathBuf>> {
    let mut items = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("Failed to read {}", path.display()))? {
        items.push(entry?.path());
    }
    items.sort();
    Ok(items)
}

fn read_basename_from_metadata(path: &Path) -> Result<Option<String>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Failed to open metadata file {}", path.display()))?;
    for record in reader.records() {
        let record = record?;
        if record.get(0) == Some("basename") {
            let value = record.get(1).unwrap_or_default().trim();
            if !value.is_empty() {
                return Ok(Some(value.to_string()));
            }
        }
    }
    Ok(None)
}

fn infer_basename(files: &[PathBuf]) -> Option<String> {
    let mut candidates: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            if !(name.ends_with(".tif") || name.ends_with(".tiff")) {
                return None;
            }
            let stem = Path::new(name).file_stem()?.to_str()?;
            Some(stem.to_string())
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let mut prefix = candidates.remove(0);
    for candidate in candidates {
        prefix = longest_common_prefix(&prefix, &candidate);
        if prefix.is_empty() {
            break;
        }
    }

    if prefix.is_empty() {
        None
    } else if prefix.ends_with('_') {
        Some(prefix)
    } else {
        Some(format!("{prefix}_"))
    }
}

fn longest_common_prefix(left: &str, right: &str) -> String {
    let mut out = String::new();
    for (l, r) in left.chars().zip(right.chars()) {
        if l != r {
            break;
        }
        out.push(l);
    }
    out
}

fn find_channel_file(files: &[PathBuf], basename: &str, channel_name: &str) -> Result<PathBuf> {
    let expected_tif = format!("{basename}{channel_name}.tif");
    let expected_tiff = format!("{basename}{channel_name}.tiff");
    let unsupported = [
        format!("{basename}{channel_name}.h5"),
        format!("{basename}{channel_name}_aligned.h5"),
        format!("{basename}{channel_name}.npz"),
        format!("{basename}{channel_name}_aligned.npz"),
    ];

    let mut tif_match = None;
    let mut unsupported_match = None;
    for path in files {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name == expected_tif || file_name == expected_tiff {
            tif_match = Some(path.clone());
        }
        if unsupported.iter().any(|name| name == file_name) {
            unsupported_match = Some(path.clone());
        }
    }

    if let Some(path) = tif_match {
        return Ok(path);
    }

    if let Some(path) = unsupported_match {
        bail!(
            "Found channel file {}, but phase 1 only supports 2D TIFF inputs",
            path.display()
        );
    }

    bail!("No TIFF file found for channel \"{channel_name}\" with basename \"{basename}\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn resolves_position_from_images_dir() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_1");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        fs::write(images.join("test_phase.tif"), [])?;
        fs::write(images.join("test_fluo.tif"), [])?;
        let mut metadata = fs::File::create(images.join("test_metadata.csv"))?;
        writeln!(metadata, "Description,values")?;
        writeln!(metadata, "basename,test_")?;

        let spec = resolve_position(images, "phase", "fluo")?;
        assert_eq!(spec.basename, "test_");
        assert!(spec.phase_image.ends_with("test_phase.tif"));
        Ok(())
    }

    #[test]
    fn infers_basename_when_metadata_missing() -> Result<()> {
        let temp = tempdir()?;
        let position = temp.path().join("Position_2");
        let images = position.join("Images");
        fs::create_dir_all(&images)?;
        fs::write(images.join("abc_phase.tif"), [])?;
        fs::write(images.join("abc_fluo.tif"), [])?;

        let spec = resolve_position(&position, "phase", "fluo")?;
        assert_eq!(spec.basename, "abc_");
        Ok(())
    }

    #[test]
    fn discovers_positions_in_experiment() -> Result<()> {
        let temp = tempdir()?;
        for idx in 1..=2 {
            let images = temp.path().join(format!("Position_{idx}")).join("Images");
            fs::create_dir_all(&images)?;
            fs::write(images.join("demo_phase.tif"), [])?;
            fs::write(images.join("demo_fluo.tif"), [])?;
        }

        let experiment = discover_experiment(temp.path(), "phase", "fluo")?;
        assert_eq!(experiment.positions.len(), 2);
        Ok(())
    }
}
