use anyhow::{anyhow, Result};
use cellpose_rs::{preprocess, CellposeSession, SegmentParams as CellposeSegmentParams};
use std::path::{Path, PathBuf};

use crate::runner::SegmentationParams;

pub trait Segmenter {
    fn segment_pair(
        &mut self,
        phase: Vec<f32>,
        fluo: Vec<f32>,
        height: usize,
        width: usize,
        params: &SegmentationParams,
    ) -> Result<Vec<u32>>;
}

pub struct CellposeModel {
    session: CellposeSession,
}

impl CellposeModel {
    pub fn new(model_path: impl AsRef<Path>, cpu: bool) -> Result<Self> {
        let model_path = normalize_model_path(model_path.as_ref());
        let session = CellposeSession::new(&model_path, cpu).map_err(|err| {
            anyhow!(
                "Failed to create cellpose session from {}: {err}",
                model_path.display()
            )
        })?;
        Ok(Self { session })
    }
}

impl Segmenter for CellposeModel {
    fn segment_pair(
        &mut self,
        phase: Vec<f32>,
        fluo: Vec<f32>,
        height: usize,
        width: usize,
        params: &SegmentationParams,
    ) -> Result<Vec<u32>> {
        let chw = preprocess::build_chw_image(phase, fluo, height, width);
        let output = self
            .session
            .segment(
                &chw,
                height,
                width,
                CellposeSegmentParams {
                    tile: params.tile,
                    batch_size: params.batch_size,
                    cellprob_threshold: params.cellprob_threshold,
                    niter: params.niter,
                    min_size: params.min_size,
                },
            )
            .map_err(|err| anyhow!("Cellpose segmentation failed: {err}"))?;
        Ok(output)
    }
}

fn normalize_model_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.onnx")
    } else {
        path.to_path_buf()
    }
}
