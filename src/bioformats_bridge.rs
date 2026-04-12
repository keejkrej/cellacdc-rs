use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BioFormatsProbeRequest {
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BioFormatsProbeResponse {
    pub lens_na: f32,
    pub size_t: usize,
    pub size_z: usize,
    pub size_c: usize,
    pub size_s: usize,
    pub time_increment: f32,
    pub time_increment_unit: String,
    pub physical_size_x: f32,
    pub physical_size_y: f32,
    pub physical_size_z: f32,
    pub physical_size_unit: String,
    pub channel_names: Vec<String>,
    pub emission_wavelengths: Vec<f32>,
    pub image_name: String,
    pub metadata_xml: String,
    pub preview_width: usize,
    pub preview_height: usize,
    pub preview_pixels: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BioFormatsExportRequest {
    pub path: PathBuf,
    pub output_path: PathBuf,
    pub source_series_index: Option<usize>,
    pub source_channel_index: usize,
    pub time_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BioFormatsExportResponse {
    pub output_path: PathBuf,
}

#[derive(Debug, Serialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum BridgeRequest<'a> {
    Probe {
        path: &'a Path,
    },
    Export {
        path: &'a Path,
        output_path: &'a Path,
        source_series_index: Option<usize>,
        source_channel_index: usize,
        time_range: Option<(usize, usize)>,
    },
}

pub fn run_bioformats_probe(request: BioFormatsProbeRequest) -> Result<BioFormatsProbeResponse> {
    run_bridge_command(BridgeRequest::Probe {
        path: &request.path,
    })
}

pub fn run_bioformats_export(request: BioFormatsExportRequest) -> Result<PathBuf> {
    let response: BioFormatsExportResponse = run_bridge_command(BridgeRequest::Export {
        path: &request.path,
        output_path: &request.output_path,
        source_series_index: request.source_series_index,
        source_channel_index: request.source_channel_index,
        time_range: request.time_range,
    })?;
    Ok(response.output_path)
}

fn run_bridge_command<T>(request: BridgeRequest<'_>) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let jar_path = bridge_jar_path();
    if !jar_path.exists() {
        bail!(
            "Bio-Formats bridge is not available at {}. Native import still works for TIFF/NPZ/H5 sources, but vendor microscopy files require the bridge.",
            jar_path.display()
        );
    }

    let request_json = serde_json::to_string(&request)?;
    let output = Command::new("java")
        .arg("-jar")
        .arg(&jar_path)
        .arg(request_json)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
        .with_context(|| {
            "Failed to launch Bio-Formats bridge. Ensure a Java runtime is installed."
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Bio-Formats bridge failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8(output.stdout)
        .with_context(|| "Bio-Formats bridge returned non-UTF8 output")?;
    serde_json::from_str::<T>(&stdout)
        .with_context(|| "Bio-Formats bridge returned malformed JSON output")
}

fn bridge_jar_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("bioformats-bridge")
        .join("bridge.jar")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_bridge_returns_actionable_error() {
        let error = run_bioformats_probe(BioFormatsProbeRequest {
            path: PathBuf::from("demo.czi"),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Bio-Formats bridge is not available"));
    }
}
