use cellacdc_rs::{FrameProjection, MaskEditSession};
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
pub(crate) enum AnnotationTool {
    Select,
    Brush,
    Eraser,
    Relabel,
    Merge,
    Delete,
}

impl Default for AnnotationTool {
    fn default() -> Self {
        Self::Select
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnnotationPendingAction {
    ChangePosition(usize),
    ChangeSegmentation(Option<String>),
}

#[derive(Debug, Clone)]
pub(crate) struct AnnotationWorkspaceState {
    pub(crate) tool: AnnotationTool,
    pub(crate) brush_radius: usize,
    pub(crate) relabel_target: String,
    pub(crate) merge_target: String,
    pub(crate) save_as_endname: String,
    pub(crate) pending_action: Option<AnnotationPendingAction>,
}

impl Default for AnnotationWorkspaceState {
    fn default() -> Self {
        Self {
            tool: AnnotationTool::Select,
            brush_radius: 5,
            relabel_target: String::new(),
            merge_target: String::new(),
            save_as_endname: "edited".to_string(),
            pending_action: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedMaskDocument {
    pub(crate) position_dir: PathBuf,
    pub(crate) segmentation_endname: Option<String>,
    pub(crate) session: MaskEditSession,
    pub(crate) revision: u64,
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
