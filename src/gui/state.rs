use cellacdc_rs::FrameProjection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ProjectionMode {
    Max,
    ZSlice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AppRoute {
    Launcher,
    DataStructure,
    DataPrep,
    #[serde(alias = "Viewer")]
    Segmentation,
    Annotation,
    Utilities,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum UtilityTool {
    CountObjects,
    FillHoles,
    Connect3d,
    Stack2dTo3d,
    CombineChannels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum UtilityScopeMode {
    Auto,
    Position,
    Experiment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ResolutionLayoutChoice {
    Auto,
    Yx,
    Tyx,
    Zyx,
    Tzyx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct UtilityState {
    pub(crate) selected_tool: UtilityTool,
    pub(crate) segmentation_path: String,
    pub(crate) output_path: String,
    pub(crate) scope_path: String,
    pub(crate) recipe_path: String,
    pub(crate) append_name: String,
    pub(crate) resolution_size_t: String,
    pub(crate) resolution_size_z: String,
    pub(crate) resolution_layout: ResolutionLayoutChoice,
    pub(crate) stack_target_size_z: usize,
    pub(crate) scope_mode: UtilityScopeMode,
}

impl Default for UtilityState {
    fn default() -> Self {
        Self {
            selected_tool: UtilityTool::CountObjects,
            segmentation_path: String::new(),
            output_path: String::new(),
            scope_path: String::new(),
            recipe_path: String::new(),
            append_name: "combined".to_string(),
            resolution_size_t: String::new(),
            resolution_size_z: String::new(),
            resolution_layout: ResolutionLayoutChoice::Auto,
            stack_target_size_z: 3,
            scope_mode: UtilityScopeMode::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ViewKey {
    pub(crate) position_dir: PathBuf,
    pub(crate) channel: String,
    pub(crate) frame_index: usize,
    pub(crate) projection: FrameProjection,
    pub(crate) segmentation_endname: Option<String>,
    pub(crate) overlay_alpha_bits: u32,
    pub(crate) show_overlay: bool,
}
