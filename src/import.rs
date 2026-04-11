use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportSourceKind {
    Npz,
    H5,
    Tiff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportSource {
    pub path: PathBuf,
    pub kind: ImportSourceKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportExperimentConfig {
    pub source_dir: PathBuf,
    pub target_dir: PathBuf,
    pub position_name: String,
    pub copy_files: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedExperiment {
    pub experiment_dir: PathBuf,
    pub position_dir: PathBuf,
    pub images_dir: PathBuf,
    pub imported_files: Vec<PathBuf>,
    pub skipped_files: Vec<PathBuf>,
}

pub fn detect_import_source_kind(path: impl AsRef<Path>) -> Option<ImportSourceKind> {
    let ext = path.as_ref().extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "npz" => Some(ImportSourceKind::Npz),
        "h5" | "hdf5" => Some(ImportSourceKind::H5),
        "tif" | "tiff" => Some(ImportSourceKind::Tiff),
        _ => None,
    }
}

pub fn discover_import_sources(dir: impl AsRef<Path>) -> Result<Vec<ImportSource>> {
    let dir = dir.as_ref();
    let mut sources = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(kind) = detect_import_source_kind(&path) {
            sources.push(ImportSource { path, kind });
        }
    }
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(sources)
}

pub fn import_experiment(config: ImportExperimentConfig) -> Result<ImportedExperiment> {
    let sources = discover_import_sources(&config.source_dir)?;
    if sources.is_empty() {
        bail!(
            "No supported import sources found in {}",
            config.source_dir.display()
        );
    }
    let experiment_dir = config.target_dir;
    let position_dir = experiment_dir.join(&config.position_name);
    let images_dir = position_dir.join("Images");
    fs::create_dir_all(&images_dir)?;

    let mut imported_files = Vec::new();
    for source in &sources {
        let destination = images_dir.join(
            source
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("imported.dat"),
        );
        if config.copy_files {
            fs::copy(&source.path, &destination).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    source.path.display(),
                    destination.display()
                )
            })?;
        } else {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&source.path, &destination).with_context(|| {
                format!(
                    "Failed to symlink {} to {}",
                    source.path.display(),
                    destination.display()
                )
            })?;
            #[cfg(not(unix))]
            fs::copy(&source.path, &destination).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    source.path.display(),
                    destination.display()
                )
            })?;
        }
        imported_files.push(destination);
    }

    Ok(ImportedExperiment {
        experiment_dir,
        position_dir,
        images_dir,
        imported_files,
        skipped_files: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn discovers_supported_sources() -> Result<()> {
        let dir = tempdir()?;
        fs::write(dir.path().join("a.npz"), b"test")?;
        fs::write(dir.path().join("b.h5"), b"test")?;
        fs::write(dir.path().join("c.txt"), b"test")?;
        let discovered = discover_import_sources(dir.path())?;
        assert_eq!(discovered.len(), 2);
        Ok(())
    }

    #[test]
    fn imports_supported_sources_into_position_images() -> Result<()> {
        let source = tempdir()?;
        let target = tempdir()?;
        fs::write(source.path().join("demo_phase.npz"), b"test")?;
        let imported = import_experiment(ImportExperimentConfig {
            source_dir: source.path().to_path_buf(),
            target_dir: target.path().join("experiment"),
            position_name: "Position_1".to_string(),
            copy_files: true,
        })?;
        assert_eq!(imported.imported_files.len(), 1);
        assert!(imported.images_dir.exists());
        Ok(())
    }
}
