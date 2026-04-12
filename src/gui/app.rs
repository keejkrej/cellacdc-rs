use super::jobs::JobHandle;
use super::persist::{self, PersistedState};
use super::state::{
    AnnotationPendingAction, AnnotationWorkspaceState, AppRoute, DataPrepWorkspaceState,
    DataStructureWorkspaceState, InspectionKey, LoadedMaskDocument, ProjectionMode, ViewKey,
};
use super::workspaces;
use anyhow::{anyhow, bail, Context, Result};
use cellacdc_rs::{
    apply_custom_annotation_mutation, assign_mother_bud, build_snapshot_profile,
    derive_custom_annotation_memberships, export_frame_image, export_frame_sequence,
    find_next_mother_candidate, global_custom_annotation_definitions_path, inspect_position_frame,
    load_custom_annotation_definitions, load_data_prep_state, mark_unknown_lineage,
    open_experiment_session, propagate_lineage_for_position, review_lineage_frame,
    save_cell_cycle_annotations, save_custom_annotation_definitions,
    set_lineage_parent_for_position, validate_custom_annotation_definition,
    write_custom_annotations_to_acdc_output, CellCycleEdit, CellCyclePropagationConfig,
    CustomAnnotationDefinition, CustomAnnotationKind, CustomAnnotationMutation, ExperimentSession,
    FrameData, FrameInspection, FrameInspectionConfig, FrameProjection, ImageExportFormat,
    LineageFrameEdit, MaskData, MaskEditCommand, MaskEditSession, MaskPathResolution,
    MaskRecoveryState, MaskSaveMode, OverlayMarker, OverlayRenderStyle, OverwritePolicy,
    PositionSession, RenderFrameRequest, ScaleBarStyle, SegmentationLayout, SegmentationParams,
    TimestampStyle, ViewPlane,
};
use eframe::egui::{self, TextureHandle};
use rfd::FileDialog;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const MAX_LOG_LINES: usize = 200;

pub fn launch_gui() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 920.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Cell-ACDC Rust",
        options,
        Box::new(|cc| Ok(Box::new(CellAcdcGui::new(cc)))),
    )
    .map_err(|err| anyhow!(err.to_string()))
}

pub(crate) struct CellAcdcGui {
    pub(crate) persisted: PersistedState,
    pub(crate) last_non_launcher_route: AppRoute,
    pub(crate) experiment: Option<ExperimentSession>,
    pub(crate) selected_position_idx: usize,
    pub(crate) selected_frame_idx: usize,
    pub(crate) texture: Option<TextureHandle>,
    pub(crate) texture_key: Option<ViewKey>,
    pub(crate) last_error: Option<String>,
    pub(crate) logs: Vec<String>,
    pub(crate) active_job: Option<JobHandle>,
    pub(crate) data_structure: DataStructureWorkspaceState,
    pub(crate) data_prep: DataPrepWorkspaceState,
    pub(crate) annotation: AnnotationWorkspaceState,
    pub(crate) annotation_document: Option<LoadedMaskDocument>,
    pub(crate) inspection_key: Option<InspectionKey>,
    pub(crate) frame_inspection: Option<FrameInspection>,
    pub(crate) status_text: String,
    pub(crate) pending_annotation_autosave: bool,
    pub(crate) last_annotation_change_at: Option<Instant>,
}

impl CellAcdcGui {
    pub(crate) fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = persist::load(cc.storage);
        let data_prep_active_channel = persisted.data_prep_active_channel.clone();
        let data_prep_projection_mode = persisted.data_prep_projection_mode;
        let data_prep_z_index = persisted.data_prep_z_index;
        let data_structure_destination_path = persisted.data_structure_destination_path.clone();
        let data_structure_backend = persisted.data_structure_backend;
        let data_structure_layout_kind = persisted.data_structure_layout_kind;
        let data_structure_conflict_mode = persisted.data_structure_conflict_mode;
        let data_structure_metadata_policy = persisted.data_structure_metadata_policy;
        let data_structure_output_format = persisted.data_structure_output_format;
        let last_non_launcher_route = match persisted.route {
            AppRoute::Launcher => AppRoute::Segmentation,
            route => route,
        };
        let mut app = Self {
            annotation: AnnotationWorkspaceState {
                tool: persisted.annotation_tool,
                mode: persisted.gui_mode,
                lineage_tool: persisted.lineage_tool,
                brush_radius: persisted.annotation_brush_radius,
                view_plane: persisted.view_plane,
                custom_annotation_toolbar: super::state::CustomAnnotationToolbarState {
                    show_all: persisted.show_all_custom_annotations,
                },
                tracking_params: super::state::TrackingParamsDialogState {
                    ioa_threshold: persisted.track_ioa_threshold,
                },
                ..Default::default()
            },
            persisted,
            last_non_launcher_route,
            experiment: None,
            selected_position_idx: 0,
            selected_frame_idx: 0,
            texture: None,
            texture_key: None,
            last_error: None,
            logs: Vec::new(),
            active_job: None,
            data_structure: DataStructureWorkspaceState {
                destination_path: data_structure_destination_path,
                backend: data_structure_backend,
                layout_kind: data_structure_layout_kind,
                conflict_mode: data_structure_conflict_mode,
                metadata_policy: data_structure_metadata_policy,
                output_format: data_structure_output_format,
                ..Default::default()
            },
            data_prep: DataPrepWorkspaceState {
                active_channel: data_prep_active_channel,
                projection_mode: data_prep_projection_mode,
                z_index: data_prep_z_index,
                ..Default::default()
            },
            annotation_document: None,
            inspection_key: None,
            frame_inspection: None,
            status_text: String::new(),
            pending_annotation_autosave: false,
            last_annotation_change_at: None,
        };
        if let Some(path) = app.persisted.last_opened_path.clone() {
            let path = PathBuf::from(path);
            if path.exists() {
                if let Err(err) = app.open_path(path) {
                    app.last_error = Some(err.to_string());
                }
            }
        }
        app
    }

    pub(crate) fn pick_and_open_session(&mut self) {
        if let Some(path) = FileDialog::new().pick_folder() {
            if let Err(err) = self.open_path(path) {
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn set_route(&mut self, route: AppRoute) {
        if route != AppRoute::Launcher {
            self.last_non_launcher_route = route;
        }
        self.persisted.route = route;
        if route == AppRoute::Annotation {
            self.ensure_annotation_document_loaded();
        } else if route == AppRoute::DataPrep {
            self.reload_data_prep_state();
        }
    }

    pub(crate) fn restore_current_session_route(&mut self) {
        let route = match self.last_non_launcher_route {
            AppRoute::Annotation => AppRoute::Annotation,
            AppRoute::Segmentation => AppRoute::Segmentation,
            _ => AppRoute::Segmentation,
        };
        self.set_route(route);
    }

    pub(crate) fn open_path(&mut self, path: PathBuf) -> Result<()> {
        let experiment = open_experiment_session(&path)?;
        self.experiment = Some(experiment);
        self.selected_position_idx = 0;
        self.selected_frame_idx = 0;
        self.annotation_document = None;
        self.annotation.pending_action = None;
        self.persisted.last_opened_path = Some(path.display().to_string());
        self.set_route(AppRoute::Segmentation);
        self.push_recent_path(path);
        self.sync_selection_with_position();
        self.reload_data_prep_state();
        self.reload_custom_annotation_store();
        self.invalidate_texture();
        self.last_error = None;
        self.append_log("Opened Cell-ACDC session".to_string());
        Ok(())
    }

    pub(crate) fn reload_experiment(&mut self) {
        let Some(current) = self.experiment.as_ref() else {
            return;
        };
        match current.reload() {
            Ok(experiment) => {
                self.experiment = Some(experiment);
                self.sync_selection_with_position();
                self.reload_data_prep_state();
                self.reload_custom_annotation_store();
                self.ensure_annotation_document_loaded();
                self.invalidate_texture();
                self.append_log("Reloaded experiment session".to_string());
                self.last_error = None;
            }
            Err(err) => self.last_error = Some(err.to_string()),
        }
    }

    pub(crate) fn push_recent_path(&mut self, path: PathBuf) {
        let display = path.display().to_string();
        self.persisted.recent_paths.retain(|item| item != &display);
        self.persisted.recent_paths.insert(0, display);
        self.persisted.recent_paths.truncate(8);
    }

    pub(crate) fn append_log(&mut self, message: String) {
        self.logs.push(message);
        if self.logs.len() > MAX_LOG_LINES {
            let overflow = self.logs.len() - MAX_LOG_LINES;
            self.logs.drain(0..overflow);
        }
    }

    pub(crate) fn invalidate_texture(&mut self) {
        self.texture_key = None;
        self.inspection_key = None;
    }

    pub(crate) fn selected_position(&self) -> Option<&PositionSession> {
        self.experiment
            .as_ref()
            .and_then(|experiment| experiment.positions.get(self.selected_position_idx))
    }

    pub(crate) fn selected_segmentation_path(&self) -> Option<PathBuf> {
        let position = self.selected_position()?;
        position
            .segmentation_asset(self.persisted.selected_segmentation_endname.as_deref())
            .map(|asset| asset.path.clone())
    }

    pub(crate) fn sync_selection_with_position(&mut self) {
        let Some(position) = self.selected_position().cloned() else {
            self.annotation_document = None;
            return;
        };
        let channel_names = position.channel_names();
        let default_channel = position.default_channel_name();
        let default_phase = position.default_phase_channel_name();
        let default_fluo = position.default_fluo_channel_name();
        let first_segmentation = position
            .segmentations
            .first()
            .and_then(|asset| asset.endname.clone());
        let size_t = position.spec.size_t;
        let size_z = position.spec.size_z;

        if self.persisted.selected_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.selected_channel)
        {
            self.persisted.selected_channel =
                default_channel.unwrap_or_else(|| String::from("<missing>"));
        }

        if self.persisted.phase_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.phase_channel)
        {
            self.persisted.phase_channel =
                default_phase.unwrap_or_else(|| self.persisted.selected_channel.clone());
        }

        if self.persisted.fluo_channel.is_empty()
            || !channel_names
                .iter()
                .any(|name| name == &self.persisted.fluo_channel)
        {
            self.persisted.fluo_channel =
                default_fluo.unwrap_or_else(|| self.persisted.selected_channel.clone());
        }

        let has_selected_segm = position
            .segmentations
            .iter()
            .any(|asset| asset.endname == self.persisted.selected_segmentation_endname);
        if !has_selected_segm {
            self.persisted.selected_segmentation_endname = first_segmentation;
        }

        if self.selected_frame_idx >= size_t {
            self.selected_frame_idx = size_t.saturating_sub(1);
        }
        if size_z <= 1 {
            self.persisted.z_index = 0;
            self.annotation.view_plane = ViewPlane::XY;
            self.persisted.view_plane = ViewPlane::XY;
        } else {
            self.persisted.z_index = self.persisted.z_index.min(size_z - 1);
        }
        if size_t <= 1 {
            self.annotation.mode = super::state::GuiMode::Snapshot;
        } else if self.annotation.mode == super::state::GuiMode::Snapshot {
            self.annotation.mode = self.persisted.gui_mode;
            if self.annotation.mode == super::state::GuiMode::Snapshot {
                self.annotation.mode = super::state::GuiMode::Viewer;
            }
        }
        self.annotation.save_as_endname = self
            .persisted
            .selected_segmentation_endname
            .clone()
            .unwrap_or_else(|| "edited".to_string());
    }

    pub(crate) fn reload_data_prep_state(&mut self) {
        let Some(position) = self.selected_position().cloned() else {
            self.data_prep = DataPrepWorkspaceState {
                active_channel: self.persisted.data_prep_active_channel.clone(),
                projection_mode: self.persisted.data_prep_projection_mode,
                z_index: self.persisted.data_prep_z_index,
                ..Default::default()
            };
            return;
        };
        match load_data_prep_state(
            &position.spec.position_dir,
            Some(&self.persisted.data_prep_active_channel),
        ) {
            Ok(state) => {
                self.data_prep.active_channel = if state.active_channel.is_empty() {
                    position
                        .default_channel_name()
                        .unwrap_or_else(|| self.persisted.selected_channel.clone())
                } else {
                    state.active_channel
                };
                self.data_prep.segm_info = state.segm_info;
                self.data_prep.crop_rois = state.crop_rois;
                self.data_prep.background_rois = state.background_rois;
                self.data_prep.free_roi = state.free_roi;
                self.data_prep.pending_crop_preview = None;
                self.data_prep.last_loaded_position = Some(position.spec.position_dir.clone());
                if let Some(free_roi) = self.data_prep.free_roi.as_ref() {
                    let (y0, x0, y1, x1) = free_roi.bbox_yxxy;
                    self.data_prep.free_roi_points = vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
                } else {
                    self.data_prep.free_roi_points.clear();
                }
                if !position
                    .channel_names()
                    .iter()
                    .any(|name| name == &self.data_prep.active_channel)
                {
                    self.data_prep.active_channel =
                        position.default_channel_name().unwrap_or_default();
                }
                self.data_prep.z_index = self
                    .data_prep
                    .z_index
                    .min(position.spec.size_z.saturating_sub(1));
                self.last_error = None;
            }
            Err(err) => {
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn current_projection(&self) -> FrameProjection {
        match self.persisted.projection_mode {
            ProjectionMode::Max => FrameProjection::Max,
            ProjectionMode::ZSlice => FrameProjection::ZSlice(self.persisted.z_index),
        }
    }

    pub(crate) fn current_snapshot_profile(&self) -> Option<cellacdc_rs::SnapshotProfile> {
        let position = self.selected_position()?;
        Some(build_snapshot_profile(
            position.spec.size_t,
            position.spec.size_z,
            self.annotation.view_plane,
        ))
    }

    pub(crate) fn current_position_key(&self) -> Option<String> {
        self.selected_position().map(PositionSession::position_key)
    }

    pub(crate) fn current_view_depth_limit(&self) -> Option<usize> {
        let position = self.selected_position()?;
        match self.annotation.view_plane {
            ViewPlane::XY => Some(position.spec.size_z),
            ViewPlane::XZ | ViewPlane::YZ => position
                .load_channel_frame_for_view(
                    &self.persisted.selected_channel,
                    self.selected_frame_idx
                        .min(position.spec.size_t.saturating_sub(1)),
                    ViewPlane::XY,
                    FrameProjection::ZSlice(0),
                )
                .ok()
                .map(|frame| {
                    if self.annotation.view_plane == ViewPlane::XZ {
                        frame.height
                    } else {
                        frame.width
                    }
                }),
        }
    }

    pub(crate) fn experiment_position_keys(&self) -> Vec<String> {
        self.experiment
            .as_ref()
            .map(|experiment| {
                experiment
                    .positions
                    .iter()
                    .map(PositionSession::position_key)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn reload_custom_annotation_store(&mut self) {
        let Some(experiment) = self.experiment.as_ref() else {
            self.annotation.custom_annotations = Default::default();
            self.annotation.active_custom_annotation.active_name = None;
            self.annotation.custom_annotations_dirty = false;
            return;
        };
        let global_path = global_custom_annotation_definitions_path();
        let mut definitions = load_custom_annotation_definitions(&global_path).unwrap_or_default();
        let position_paths = experiment
            .positions
            .iter()
            .map(|position| position.spec.position_dir.clone())
            .collect::<Vec<_>>();
        match derive_custom_annotation_memberships(
            &position_paths,
            self.persisted.selected_segmentation_endname.as_deref(),
        ) {
            Ok(mut derived) => {
                for (name, definition) in &derived.definitions {
                    let name = name.clone();
                    let definition = definition.clone();
                    definitions.entry(name).or_insert(definition);
                }
                derived.definitions = definitions;
                self.annotation.custom_annotations = derived;
                if let Some(active) = self.annotation.active_custom_annotation.active_name.clone() {
                    if !self
                        .annotation
                        .custom_annotations
                        .definitions
                        .contains_key(&active)
                    {
                        self.annotation.active_custom_annotation.active_name = None;
                    }
                }
                self.annotation.custom_annotations_dirty = false;
            }
            Err(err) => {
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn current_custom_annotation_column_exists(&self, name: &str) -> Result<bool> {
        let Some(position) = self.selected_position() else {
            return Ok(false);
        };
        let path =
            position.acdc_output_path(self.persisted.selected_segmentation_endname.as_deref());
        if !path.exists() {
            return Ok(false);
        }
        let table = cellacdc_rs::read_table(&path)?;
        Ok(table.maybe_header_index(name).is_some())
    }

    pub(crate) fn persist_custom_annotation_definitions(
        &mut self,
        position_keys: &[String],
    ) -> Result<()> {
        let definitions = self.annotation.custom_annotations.definitions.clone();
        let global_path = global_custom_annotation_definitions_path();
        save_custom_annotation_definitions(&global_path, &definitions)?;
        if let Some(experiment) = self.experiment.as_ref() {
            for position in &experiment.positions {
                let key = position.position_key();
                if !position_keys.is_empty()
                    && !position_keys.iter().any(|selected| selected == &key)
                {
                    continue;
                }
                save_custom_annotation_definitions(
                    position.custom_annotation_params_path(),
                    &definitions,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn current_custom_annotation_markers(&self) -> Result<Vec<OverlayMarker>> {
        self.custom_annotation_markers_for_frame(self.selected_frame_idx)
    }

    pub(crate) fn custom_annotation_markers_for_frame(
        &self,
        frame_index: usize,
    ) -> Result<Vec<OverlayMarker>> {
        if self.annotation.view_plane != ViewPlane::XY {
            return Ok(Vec::new());
        }
        let Some(position_key) = self.current_position_key() else {
            return Ok(Vec::new());
        };
        let Some(position) = self.selected_position() else {
            return Ok(Vec::new());
        };
        let Some(segmentation) = position.load_segmentation_frame_for_view(
            self.persisted.selected_segmentation_endname.as_deref(),
            frame_index,
            ViewPlane::XY,
            self.current_projection(),
        )?
        else {
            return Ok(Vec::new());
        };
        let centroids = label_centroids(&segmentation);
        let visible_names = if self.annotation.custom_annotation_toolbar.show_all {
            self.annotation
                .custom_annotations
                .definitions
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            self.annotation
                .custom_annotations
                .definitions
                .iter()
                .filter_map(|(name, definition)| {
                    (self
                        .annotation
                        .active_custom_annotation
                        .active_name
                        .as_deref()
                        == Some(name.as_str())
                        || !definition.hide_when_inactive)
                        .then_some(name.clone())
                })
                .collect::<Vec<_>>()
        };
        let mut markers = Vec::new();
        let Some(per_position) = self
            .annotation
            .custom_annotations
            .annotated_ids_by_position
            .get(&position_key)
        else {
            return Ok(markers);
        };
        for name in visible_names {
            let Some(definition) = self.annotation.custom_annotations.definitions.get(&name) else {
                continue;
            };
            let Some(per_frame) = per_position.get(&name) else {
                continue;
            };
            let Some(ids) = per_frame.get(&frame_index) else {
                continue;
            };
            for id in ids {
                let Some((x, y)) = centroids.get(id).copied() else {
                    continue;
                };
                markers.push(OverlayMarker {
                    x,
                    y,
                    symbol: definition.symbol.clone(),
                    color: definition.symbol_color_rgba,
                    size: 15,
                });
            }
        }
        Ok(markers)
    }

    pub(crate) fn current_view_key(&self) -> Option<ViewKey> {
        let position = self.selected_position()?;
        Some(ViewKey {
            position_dir: position.spec.position_dir.clone(),
            channel: self.persisted.selected_channel.clone(),
            frame_index: self.selected_frame_idx,
            view_plane: self.annotation.view_plane,
            projection: self.current_projection(),
            segmentation_endname: self.persisted.selected_segmentation_endname.clone(),
            overlay_alpha_bits: self.persisted.overlay_alpha.to_bits(),
            show_overlay: self.persisted.show_segmentation_overlay,
            show_overlay_labels: self.persisted.display.show_overlay_labels,
            overlay_single_channel_mode: self.persisted.display.overlay_single_channel_mode,
            true_transparency: self.persisted.display.true_transparency,
            add_scale_bar: self.persisted.display.add_scale_bar,
            add_timestamp: self.persisted.display.add_timestamp,
            highlighted_label: self.active_highlighted_label(),
            selected_label: self.current_annotation_label(),
        })
    }

    pub(crate) fn selected_measurement_suffix(&self) -> Option<String> {
        self.persisted.selected_segmentation_endname.clone()
    }

    pub(crate) fn run_output_suffix(&self) -> Option<String> {
        let trimmed = self.persisted.run_output_suffix.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    pub(crate) fn overwrite_policy(&self) -> OverwritePolicy {
        if self.persisted.overwrite_outputs {
            OverwritePolicy::Overwrite
        } else {
            OverwritePolicy::Refuse
        }
    }

    pub(crate) fn segmentation_params(&self) -> SegmentationParams {
        SegmentationParams {
            tile: self.persisted.tile,
            batch_size: self.persisted.batch_size,
            cellprob_threshold: self.persisted.cellprob_threshold,
            niter: self.persisted.niter,
            min_size: self.persisted.min_size,
        }
    }

    pub(crate) fn autofill_utility_from_selected_segmentation(&mut self) {
        let Some(segmentation_path) = self.selected_segmentation_path() else {
            return;
        };
        self.persisted.utility.segmentation_path = segmentation_path.display().to_string();
        if self.persisted.utility.output_path.trim().is_empty() {
            self.persisted.utility.output_path = workspaces::suggested_output_path(
                self.persisted.utility.selected_tool,
                &segmentation_path,
            )
            .display()
            .to_string();
        }
        if self.persisted.utility.scope_path.trim().is_empty() {
            if let Some(position) = self.selected_position() {
                self.persisted.utility.scope_path =
                    position.spec.position_dir.display().to_string();
            }
        }
    }

    pub(crate) fn clear_annotation_document(&mut self) {
        self.annotation_document = None;
        self.frame_inspection = None;
        self.inspection_key = None;
        self.pending_annotation_autosave = false;
        self.annotation.editor_undo.clear();
        self.annotation.editor_redo.clear();
    }

    pub(crate) fn current_annotation_document(&self) -> Option<&LoadedMaskDocument> {
        self.annotation_document.as_ref().filter(|document| {
            self.selected_position()
                .map(|position| {
                    document.position_dir == position.spec.position_dir
                        && document.segmentation_endname
                            == self.persisted.selected_segmentation_endname
                })
                .unwrap_or(false)
        })
    }

    pub(crate) fn current_annotation_document_mut(&mut self) -> Option<&mut LoadedMaskDocument> {
        let position_dir = self.selected_position()?.spec.position_dir.clone();
        let selected_endname = self.persisted.selected_segmentation_endname.clone();
        self.annotation_document.as_mut().filter(|document| {
            document.position_dir == position_dir
                && document.segmentation_endname == selected_endname
        })
    }

    pub(crate) fn current_annotation_label(&self) -> Option<u32> {
        self.current_annotation_document()
            .and_then(|document| document.session.selection.selected_label)
    }

    pub(crate) fn annotation_document_dirty(&self) -> bool {
        self.current_annotation_document()
            .map(|document| document.session.dirty)
            .unwrap_or(false)
            || self.annotation.custom_annotations_dirty
    }

    pub(crate) fn annotation_recovery_state(&self) -> MaskRecoveryState {
        self.current_annotation_document()
            .map(|document| document.session.recovery_state())
            .unwrap_or(MaskRecoveryState::None)
    }

    pub(crate) fn annotation_edits_allowed(&self) -> bool {
        if self.annotation_recovery_state() == MaskRecoveryState::RecoveryAvailable {
            return false;
        }
        if self
            .current_snapshot_profile()
            .map(|profile| !profile.editing_allowed_on_current_plane)
            .unwrap_or(false)
        {
            return false;
        }
        let Some(document) = self.current_annotation_document() else {
            return false;
        };
        !matches!(
            (document.session.data.layout, self.current_projection()),
            (
                SegmentationLayout::ZYX | SegmentationLayout::TZYX,
                FrameProjection::Max
            )
        )
    }

    pub(crate) fn request_position_selection(&mut self, idx: usize) {
        if idx == self.selected_position_idx {
            return;
        }
        if self.persisted.route == AppRoute::Annotation && self.annotation_document_dirty() {
            self.annotation.pending_action = Some(AnnotationPendingAction::ChangePosition(idx));
            return;
        }
        self.apply_position_selection(idx);
    }

    pub(crate) fn apply_position_selection(&mut self, idx: usize) {
        self.selected_position_idx = idx;
        self.selected_frame_idx = 0;
        self.sync_selection_with_position();
        self.reload_data_prep_state();
        self.clear_annotation_document();
        self.reload_custom_annotation_store();
        self.ensure_annotation_document_loaded();
        self.invalidate_texture();
    }

    pub(crate) fn request_segmentation_selection(&mut self, endname: Option<String>) {
        if endname == self.persisted.selected_segmentation_endname {
            return;
        }
        if self.persisted.route == AppRoute::Annotation && self.annotation_document_dirty() {
            self.annotation.pending_action =
                Some(AnnotationPendingAction::ChangeSegmentation(endname));
            return;
        }
        self.apply_segmentation_selection(endname);
    }

    pub(crate) fn apply_segmentation_selection(&mut self, endname: Option<String>) {
        self.persisted.selected_segmentation_endname = endname.clone();
        self.annotation.save_as_endname = endname.unwrap_or_else(|| "edited".to_string());
        self.clear_annotation_document();
        self.reload_custom_annotation_store();
        self.ensure_annotation_document_loaded();
        self.invalidate_texture();
    }

    pub(crate) fn apply_pending_annotation_action(&mut self) {
        let Some(action) = self.annotation.pending_action.take() else {
            return;
        };
        match action {
            AnnotationPendingAction::ChangePosition(idx) => self.apply_position_selection(idx),
            AnnotationPendingAction::ChangeSegmentation(endname) => {
                self.apply_segmentation_selection(endname)
            }
        }
    }

    pub(crate) fn cancel_pending_annotation_action(&mut self) {
        self.annotation.pending_action = None;
    }

    pub(crate) fn discard_annotation_changes_and_continue(&mut self) {
        self.clear_annotation_document();
        self.reload_custom_annotation_store();
        self.apply_pending_annotation_action();
    }

    pub(crate) fn save_annotation_changes_and_continue(&mut self) {
        if let Err(err) = self.save_current_annotation_overwrite() {
            self.last_error = Some(err.to_string());
            return;
        }
        self.apply_pending_annotation_action();
    }

    pub(crate) fn ensure_annotation_document_loaded(&mut self) {
        if self.persisted.route != AppRoute::Annotation {
            return;
        }
        let Some(position) = self.selected_position().cloned() else {
            self.annotation_document = None;
            return;
        };
        let Some(asset) = position
            .segmentation_asset(self.persisted.selected_segmentation_endname.as_deref())
            .cloned()
        else {
            self.annotation_document = None;
            return;
        };
        if self.current_annotation_document().is_some() {
            return;
        }
        let resolution = self.annotation_resolution_for_position(&position, &asset.name);
        match MaskEditSession::from_source_path(&asset.path, Some(&resolution)) {
            Ok(mut session) => {
                session.selection.frame_index = self.selected_frame_idx;
                session.selection.z_index = match self.current_projection() {
                    FrameProjection::Max => None,
                    FrameProjection::ZSlice(z_index) => Some(z_index),
                };
                self.annotation_document = Some(LoadedMaskDocument {
                    position_dir: position.spec.position_dir.clone(),
                    segmentation_endname: asset.endname.clone(),
                    session,
                    revision: 0,
                });
                self.last_error = None;
                self.invalidate_texture();
            }
            Err(err) => {
                self.annotation_document = None;
                self.last_error = Some(err.to_string());
            }
        }
    }

    pub(crate) fn annotation_resolution_for_position(
        &self,
        position: &PositionSession,
        asset_name: &str,
    ) -> MaskPathResolution {
        let is_segm_3d = position
            .spec
            .segm_is_3d
            .get(asset_name)
            .copied()
            .unwrap_or(false);
        MaskPathResolution {
            size_t: Some(position.spec.size_t),
            size_z: Some(if is_segm_3d { position.spec.size_z } else { 1 }),
            layout: None,
        }
    }

    pub(crate) fn restore_annotation_recovery(&mut self) -> Result<()> {
        let Some(position) = self.selected_position().cloned() else {
            return Ok(());
        };
        let Some(asset) = position
            .segmentation_asset(self.persisted.selected_segmentation_endname.as_deref())
            .cloned()
        else {
            return Ok(());
        };
        let resolution = self.annotation_resolution_for_position(&position, &asset.name);
        let session = MaskEditSession::load_with_recovery(&asset.path, Some(&resolution), true)?;
        self.annotation_document = Some(LoadedMaskDocument {
            position_dir: position.spec.position_dir.clone(),
            segmentation_endname: asset.endname,
            session,
            revision: self
                .annotation_document
                .as_ref()
                .map(|document| document.revision + 1)
                .unwrap_or(1),
        });
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn discard_annotation_recovery(&mut self) -> Result<()> {
        let Some(document) = self.current_annotation_document_mut() else {
            return Ok(());
        };
        document.session.discard_recovery()?;
        document.revision += 1;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn annotation_save_as_path(&self, endname: &str) -> Result<PathBuf> {
        let position = self
            .selected_position()
            .ok_or_else(|| anyhow!("No position selected"))?;
        let trimmed = endname.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Save-as endname is required");
        }
        let file_name = if position.spec.basename.ends_with('_') {
            format!("{}segm_{}.npz", position.spec.basename, trimmed)
        } else {
            format!("{}_segm_{}.npz", position.spec.basename, trimmed)
        };
        Ok(position.spec.images_dir.join(file_name))
    }

    pub(crate) fn save_current_annotation_overwrite(&mut self) -> Result<()> {
        let selected_endname = self.persisted.selected_segmentation_endname.clone();
        let document = self
            .current_annotation_document_mut()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let saved = document.session.save_with_mode(MaskSaveMode::Overwrite)?;
        document.revision += 1;
        self.pending_annotation_autosave = false;
        self.flush_pending_manual_tracking_edits(selected_endname.as_deref())?;
        self.write_custom_annotations_for_selected_positions(
            &[self
                .current_position_key()
                .ok_or_else(|| anyhow!("No position selected"))?],
            selected_endname.as_deref(),
        )?;
        self.annotation.custom_annotations_dirty = false;
        self.append_log(format!("Saved GUI edits -> {}", saved.display()));
        self.reload_experiment();
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn save_current_annotation_as_version(&mut self) -> Result<()> {
        let endname = self.annotation.save_as_endname.trim().to_string();
        let target_path = self.annotation_save_as_path(&endname)?;
        if target_path.exists() {
            anyhow::bail!(
                "Segmentation version already exists: {}",
                target_path.display()
            );
        }
        let document = self
            .current_annotation_document_mut()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let saved = document
            .session
            .save_with_mode(MaskSaveMode::SaveAs(target_path.clone()))?;
        document.revision += 1;
        self.pending_annotation_autosave = false;
        self.copy_acdc_output_to_new_version(&endname)?;
        self.flush_pending_manual_tracking_edits(Some(&endname))?;
        self.write_custom_annotations_for_selected_positions(
            &[self
                .current_position_key()
                .ok_or_else(|| anyhow!("No position selected"))?],
            Some(&endname),
        )?;
        self.persisted.selected_segmentation_endname = Some(endname.clone());
        self.annotation.custom_annotations_dirty = false;
        self.append_log(format!("Saved GUI version -> {}", saved.display()));
        self.reload_experiment();
        self.clear_annotation_document();
        self.ensure_annotation_document_loaded();
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn request_save_current_annotation_overwrite(
        &mut self,
        quick_save: bool,
    ) -> Result<()> {
        let profile = self.current_snapshot_profile();
        if profile
            .as_ref()
            .map(|value| value.is_snapshot)
            .unwrap_or(false)
            && !quick_save
            && self
                .experiment
                .as_ref()
                .map(|experiment| experiment.positions.len())
                .unwrap_or(0)
                > 1
        {
            self.annotation.snapshot_save_dialog.quick_save = false;
            self.annotation.snapshot_save_dialog.selected_positions =
                vec![self.current_position_key().unwrap_or_default()];
            self.annotation.dialogs.snapshot_save_scope_open = true;
            return Ok(());
        }
        let positions = vec![self
            .current_position_key()
            .ok_or_else(|| anyhow!("No position selected"))?];
        self.save_current_annotation_overwrite_for_positions(&positions)
    }

    pub(crate) fn save_current_annotation_overwrite_for_positions(
        &mut self,
        position_keys: &[String],
    ) -> Result<()> {
        self.save_current_annotation_overwrite()?;
        if self
            .current_snapshot_profile()
            .map(|value| value.is_snapshot)
            .unwrap_or(false)
        {
            let segm_endname = self.persisted.selected_segmentation_endname.clone();
            self.persist_custom_annotation_definitions(position_keys)?;
            self.write_custom_annotations_for_selected_positions(
                position_keys,
                segm_endname.as_deref(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn copy_acdc_output_to_new_version(&self, endname: &str) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Ok(());
        };
        let source_path =
            position.acdc_output_path(self.persisted.selected_segmentation_endname.as_deref());
        if !source_path.exists() {
            return Ok(());
        }
        let target_path = position.acdc_output_path(Some(endname));
        if target_path.exists() {
            return Ok(());
        }
        let table = cellacdc_rs::read_table(&source_path)?;
        cellacdc_rs::write_table(&target_path, &table)?;
        Ok(())
    }

    pub(crate) fn write_custom_annotations_for_selected_positions(
        &mut self,
        position_keys: &[String],
        segm_endname: Option<&str>,
    ) -> Result<()> {
        self.persist_custom_annotation_definitions(position_keys)?;
        if let Some(experiment) = self.experiment.as_ref() {
            for position in &experiment.positions {
                let key = position.position_key();
                if !position_keys.is_empty() && !position_keys.iter().any(|item| item == &key) {
                    continue;
                }
                if position.acdc_output_path(segm_endname).exists() {
                    write_custom_annotations_to_acdc_output(
                        &position.spec.position_dir,
                        segm_endname,
                        &self.annotation.custom_annotations,
                        false,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn open_custom_annotation_editor(
        &mut self,
        existing_name: Option<&str>,
    ) -> Result<()> {
        self.annotation.custom_annotation_dialog = Default::default();
        if let Some(name) = existing_name {
            let definition = self
                .annotation
                .custom_annotations
                .definitions
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("Unknown custom annotation {name}"))?;
            self.annotation.custom_annotation_dialog.editing_name = Some(name.to_string());
            self.annotation.custom_annotation_dialog.name = definition.name;
            self.annotation.custom_annotation_dialog.kind_index = match definition.kind {
                CustomAnnotationKind::SingleTimePoint => 0,
                CustomAnnotationKind::MultipleTimePoints => 1,
                CustomAnnotationKind::MultipleValuesClass => 2,
            };
            self.annotation.custom_annotation_dialog.symbol = definition.symbol;
            self.annotation.custom_annotation_dialog.shortcut =
                definition.shortcut.unwrap_or_default();
            self.annotation.custom_annotation_dialog.description = definition.description;
            self.annotation.custom_annotation_dialog.keep_active = definition.keep_active;
            self.annotation.custom_annotation_dialog.hide_when_inactive =
                definition.hide_when_inactive;
            self.annotation.custom_annotation_dialog.color = definition.symbol_color_rgba;
        }
        self.annotation.dialogs.custom_annotation_editor_open = true;
        Ok(())
    }

    pub(crate) fn apply_custom_annotation_dialog(&mut self) -> Result<()> {
        let dialog = self.annotation.custom_annotation_dialog.clone();
        let kind = match dialog.kind_index {
            0 => CustomAnnotationKind::SingleTimePoint,
            1 => {
                bail!("Multiple time-points custom annotations are not implemented yet")
            }
            2 => bail!("Multiple values class custom annotations are not implemented yet"),
            _ => bail!("Unsupported custom annotation type"),
        };
        let shortcut =
            (!dialog.shortcut.trim().is_empty()).then(|| dialog.shortcut.trim().to_string());
        if let Some(shortcut) = &shortcut {
            self.validate_custom_annotation_shortcut(shortcut, dialog.editing_name.as_deref())?;
        }
        let definition = CustomAnnotationDefinition {
            name: dialog.name.trim().to_string(),
            kind,
            symbol: dialog.symbol,
            shortcut,
            description: dialog.description.trim().to_string(),
            keep_active: dialog.keep_active,
            hide_when_inactive: dialog.hide_when_inactive,
            symbol_color_rgba: dialog.color,
        };
        validate_custom_annotation_definition(&definition)?;
        if !dialog.reuse_existing_column
            && self.current_custom_annotation_column_exists(&definition.name)?
            && dialog.editing_name.as_deref() != Some(definition.name.as_str())
        {
            bail!(
                "A custom annotation column named {:?} already exists. Enable reuse to continue.",
                definition.name
            );
        }
        let mutation = if let Some(old_name) = dialog.editing_name {
            CustomAnnotationMutation::UpdateDefinition {
                old_name,
                definition: definition.clone(),
            }
        } else {
            validate_custom_annotation_definition(&definition)?;
            let mut updated = self.annotation.custom_annotations.clone();
            updated
                .definitions
                .insert(definition.name.clone(), definition.clone());
            self.annotation.custom_annotations = updated;
            self.annotation.active_custom_annotation.active_name = Some(definition.name.clone());
            self.annotation.custom_annotations_dirty = true;
            self.persist_custom_annotation_definitions(&[self
                .current_position_key()
                .ok_or_else(|| anyhow!("No position selected"))?])?;
            self.annotation.dialogs.custom_annotation_editor_open = false;
            self.invalidate_texture();
            return Ok(());
        };
        self.annotation.custom_annotations =
            apply_custom_annotation_mutation(&self.annotation.custom_annotations, mutation)?;
        self.annotation.active_custom_annotation.active_name = Some(definition.name.clone());
        self.annotation.custom_annotations_dirty = true;
        self.persist_custom_annotation_definitions(&[self
            .current_position_key()
            .ok_or_else(|| anyhow!("No position selected"))?])?;
        self.annotation.dialogs.custom_annotation_editor_open = false;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn remove_custom_annotation(
        &mut self,
        name: &str,
        remove_column: bool,
    ) -> Result<()> {
        let mutation = if remove_column {
            CustomAnnotationMutation::RemoveDefinitionAndColumn {
                annotation_name: name.to_string(),
            }
        } else {
            CustomAnnotationMutation::RemoveDefinitionKeepColumn {
                annotation_name: name.to_string(),
            }
        };
        self.annotation.custom_annotations =
            apply_custom_annotation_mutation(&self.annotation.custom_annotations, mutation)?;
        if self
            .annotation
            .active_custom_annotation
            .active_name
            .as_deref()
            == Some(name)
        {
            self.annotation.active_custom_annotation.active_name = None;
        }
        self.annotation.custom_annotations_dirty = true;
        let current_position = self
            .current_position_key()
            .ok_or_else(|| anyhow!("No position selected"))?;
        self.persist_custom_annotation_definitions(&[current_position.clone()])?;
        if remove_column {
            if let Some(position) = self.selected_position() {
                let table_path = position
                    .acdc_output_path(self.persisted.selected_segmentation_endname.as_deref());
                if table_path.exists() {
                    let mut table = cellacdc_rs::read_table(&table_path)?;
                    if let Some(col_idx) = table.maybe_header_index(name) {
                        table.headers.remove(col_idx);
                        for row in &mut table.rows {
                            row.remove(col_idx);
                        }
                        cellacdc_rs::write_table(&table_path, &table)?;
                    }
                }
            }
        }
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn update_custom_annotation_flags(
        &mut self,
        name: &str,
        keep_active: Option<bool>,
        hide_when_inactive: Option<bool>,
    ) -> Result<()> {
        let definition = self
            .annotation
            .custom_annotations
            .definitions
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Unknown custom annotation {name}"))?;
        let mut updated = definition.clone();
        if let Some(keep_active) = keep_active {
            updated.keep_active = keep_active;
        }
        if let Some(hide_when_inactive) = hide_when_inactive {
            updated.hide_when_inactive = hide_when_inactive;
        }
        self.annotation.custom_annotations = apply_custom_annotation_mutation(
            &self.annotation.custom_annotations,
            CustomAnnotationMutation::UpdateDefinition {
                old_name: name.to_string(),
                definition: updated,
            },
        )?;
        self.annotation.custom_annotations_dirty = true;
        self.persist_custom_annotation_definitions(&[self
            .current_position_key()
            .ok_or_else(|| anyhow!("No position selected"))?])?;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn toggle_custom_annotation_for_object(
        &mut self,
        frame_index: usize,
        object_id: u32,
    ) -> Result<()> {
        let annotation_name = self
            .annotation
            .active_custom_annotation
            .active_name
            .clone()
            .ok_or_else(|| anyhow!("Activate a custom annotation first"))?;
        let position_key = self
            .current_position_key()
            .ok_or_else(|| anyhow!("No position selected"))?;
        let before = self.annotation.custom_annotations.clone();
        let selected_before = self.current_annotation_label();
        let after = apply_custom_annotation_mutation(
            &before,
            CustomAnnotationMutation::ToggleObject {
                position_key,
                annotation_name,
                frame_index,
                object_id,
            },
        )?;
        self.annotation.custom_annotations = after.clone();
        self.annotation.custom_annotations_dirty = true;
        self.annotation
            .editor_undo
            .push(super::state::EditorHistoryKind::CustomAnnotation(
                super::state::CustomAnnotationCommandSnapshot {
                    before,
                    after,
                    selected_label_before: selected_before,
                    selected_label_after: Some(object_id),
                },
            ));
        self.annotation.editor_redo.clear();
        self.annotation_select_label(Some(object_id))?;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn validate_custom_annotation_shortcut(
        &self,
        shortcut: &str,
        editing_name: Option<&str>,
    ) -> Result<()> {
        let binding = super::state::ShortcutBinding {
            key: shortcut.to_ascii_uppercase(),
            command: false,
            shift: false,
            alt: false,
        };
        let candidate = super::shortcuts::binding_to_shortcut(&binding)
            .ok_or_else(|| anyhow!("Unsupported custom annotation shortcut {shortcut:?}"))?;
        let reserved_actions = [
            super::state::GuiActionId::Save,
            super::state::GuiActionId::Undo,
            super::state::GuiActionId::Redo,
            super::state::GuiActionId::FindId,
            super::state::GuiActionId::ToolSelect,
            super::state::GuiActionId::ToolBrush,
            super::state::GuiActionId::ToolEraser,
            super::state::GuiActionId::ManualTracking,
            super::state::GuiActionId::RepeatTracking,
            super::state::GuiActionId::AssignMotherToBud,
            super::state::GuiActionId::UnknownLineage,
            super::state::GuiActionId::NoLineageTool,
            super::state::GuiActionId::PropagateLineage,
        ];
        for action in reserved_actions {
            if let Some(existing) =
                super::shortcuts::shortcut_for_action(&self.persisted.shortcut_overrides, action)
            {
                if existing == candidate {
                    bail!(
                        "Shortcut {shortcut:?} is already used by {}",
                        super::actions::action_label(action)
                    );
                }
            }
        }
        for definition in self.annotation.custom_annotations.definitions.values() {
            if editing_name == Some(definition.name.as_str()) {
                continue;
            }
            if definition.shortcut.as_deref() == Some(shortcut) {
                bail!(
                    "Shortcut {shortcut:?} is already used by custom annotation {:?}",
                    definition.name
                );
            }
        }
        Ok(())
    }

    fn flush_pending_manual_tracking_edits(&mut self, segm_endname: Option<&str>) -> Result<()> {
        let Some(position_dir) = self
            .selected_position()
            .map(|position| position.spec.position_dir.clone())
        else {
            return Ok(());
        };
        let edits = std::mem::take(&mut self.annotation.pending_manual_tracking_edits);
        for edit in edits {
            cellacdc_rs::apply_manual_tracking_edit(&position_dir, segm_endname, &edit)?;
        }
        Ok(())
    }

    pub(crate) fn run_annotation_command(&mut self, command: MaskEditCommand) -> Result<()> {
        let is_selection_only = matches!(command, MaskEditCommand::SelectLabel { .. });
        let document = self
            .current_annotation_document_mut()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let result = document.session.apply_command(command)?;
        if result.changed_pixels > 0 {
            document.revision += 1;
            self.pending_annotation_autosave = true;
            self.last_annotation_change_at = Some(Instant::now());
            self.annotation
                .editor_undo
                .push(super::state::EditorHistoryKind::MaskEdit);
            self.annotation.editor_redo.clear();
        } else {
            document.revision += 1;
        }
        if is_selection_only {
            return Ok(());
        }
        self.last_error = None;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn annotation_undo(&mut self) {
        let mut changed = false;
        let Some(history) = self.annotation.editor_undo.pop() else {
            if let Some(document) = self.current_annotation_document_mut() {
                if document.session.undo() {
                    document.revision += 1;
                    changed = true;
                    self.pending_annotation_autosave = true;
                    self.last_annotation_change_at = Some(Instant::now());
                }
            }
            if changed {
                self.invalidate_texture();
            }
            return;
        };
        match history {
            super::state::EditorHistoryKind::MaskEdit => {
                if let Some(document) = self.current_annotation_document_mut() {
                    if document.session.undo() {
                        document.revision += 1;
                        changed = true;
                        self.pending_annotation_autosave = true;
                        self.last_annotation_change_at = Some(Instant::now());
                        self.annotation
                            .editor_redo
                            .push(super::state::EditorHistoryKind::MaskEdit);
                    }
                }
            }
            super::state::EditorHistoryKind::CustomAnnotation(snapshot) => {
                self.annotation.custom_annotations = snapshot.before.clone();
                self.annotation.custom_annotations_dirty = true;
                let _ = self.annotation_select_label(snapshot.selected_label_before);
                self.annotation
                    .editor_redo
                    .push(super::state::EditorHistoryKind::CustomAnnotation(snapshot));
                changed = true;
            }
        }
        if changed {
            self.invalidate_texture();
        }
    }

    pub(crate) fn annotation_redo(&mut self) {
        let mut changed = false;
        let Some(history) = self.annotation.editor_redo.pop() else {
            if let Some(document) = self.current_annotation_document_mut() {
                if document.session.redo() {
                    document.revision += 1;
                    changed = true;
                    self.pending_annotation_autosave = true;
                    self.last_annotation_change_at = Some(Instant::now());
                }
            }
            if changed {
                self.invalidate_texture();
            }
            return;
        };
        match history {
            super::state::EditorHistoryKind::MaskEdit => {
                if let Some(document) = self.current_annotation_document_mut() {
                    if document.session.redo() {
                        document.revision += 1;
                        changed = true;
                        self.pending_annotation_autosave = true;
                        self.last_annotation_change_at = Some(Instant::now());
                        self.annotation
                            .editor_undo
                            .push(super::state::EditorHistoryKind::MaskEdit);
                    }
                }
            }
            super::state::EditorHistoryKind::CustomAnnotation(snapshot) => {
                self.annotation.custom_annotations = snapshot.after.clone();
                self.annotation.custom_annotations_dirty = true;
                let _ = self.annotation_select_label(snapshot.selected_label_after);
                self.annotation
                    .editor_undo
                    .push(super::state::EditorHistoryKind::CustomAnnotation(snapshot));
                changed = true;
            }
        }
        if changed {
            self.invalidate_texture();
        }
    }

    pub(crate) fn annotation_select_label(&mut self, label: Option<u32>) -> Result<()> {
        self.run_annotation_command(MaskEditCommand::SelectLabel { label })
    }

    pub(crate) fn current_segmentation_frame_data(&self) -> Result<Option<FrameData<u32>>> {
        let Some(position) = self.selected_position() else {
            return Ok(None);
        };
        if self.persisted.route == AppRoute::Annotation {
            if let Some(document) = self.current_annotation_document() {
                return Ok(Some(mask_frame_for_projection(
                    &document.session.data,
                    self.selected_frame_idx,
                    self.annotation.view_plane,
                    self.current_projection(),
                )?));
            }
        }
        position.load_segmentation_frame_for_view(
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx,
            self.annotation.view_plane,
            self.current_projection(),
        )
    }

    pub(crate) fn current_render_request(&self) -> Result<Option<RenderFrameRequest>> {
        let Some(position) = self.selected_position() else {
            return Ok(None);
        };
        let frame = position.load_channel_frame_for_view(
            &self.persisted.selected_channel,
            self.selected_frame_idx,
            self.annotation.view_plane,
            self.current_projection(),
        )?;
        let segmentation = if self.persisted.show_segmentation_overlay {
            self.current_segmentation_frame_data()?
        } else {
            None
        };
        Ok(Some(RenderFrameRequest {
            frame,
            segmentation,
            overlay: OverlayRenderStyle {
                enabled: self.persisted.show_segmentation_overlay,
                alpha: self.persisted.overlay_alpha,
                selected_label: self.current_annotation_label(),
                highlighted_label: self.active_highlighted_label(),
                show_labels: self.persisted.display.show_overlay_labels,
                single_channel_mode: self.persisted.display.overlay_single_channel_mode,
                true_transparency: self.persisted.display.true_transparency,
                label_color: self.persisted.display.overlay_label_color,
                label_scale: self.persisted.display.overlay_label_scale,
            },
            markers: self.current_custom_annotation_markers()?,
            scale_bar: ScaleBarStyle {
                enabled: self.persisted.display.add_scale_bar,
                ..Default::default()
            },
            timestamp: TimestampStyle {
                enabled: self.persisted.display.add_timestamp,
                ..Default::default()
            },
            frame_index: self.selected_frame_idx,
            time_seconds: Some(position.spec.time_increment * self.selected_frame_idx as f64),
            physical_size_x: Some(position.spec.physical_size_x),
        }))
    }

    pub(crate) fn active_highlighted_label(&self) -> Option<u32> {
        if self.persisted.display.highlight_searched_object {
            self.annotation
                .highlight
                .searched_label
                .or(self.annotation.highlight.hovered_label)
        } else {
            self.annotation.highlight.hovered_label
        }
    }

    pub(crate) fn flush_annotation_autosave_if_due(&mut self) {
        if !self.pending_annotation_autosave {
            return;
        }
        let due_after = self.persisted.display.autosave.as_seconds();
        let elapsed = self
            .last_annotation_change_at
            .map(|at| at.elapsed().as_secs())
            .unwrap_or(due_after);
        if elapsed < due_after {
            return;
        }
        if let Some(document) = self.current_annotation_document_mut() {
            match document.session.save_autosave() {
                Ok(path) => {
                    self.append_log(format!("Autosaved GUI recovery -> {}", path.display()));
                    self.pending_annotation_autosave = false;
                }
                Err(err) => {
                    self.last_error = Some(err.to_string());
                }
            }
        }
    }

    pub(crate) fn current_inspection(&mut self) -> Result<Option<&FrameInspection>> {
        let Some(position) = self.selected_position() else {
            self.frame_inspection = None;
            self.inspection_key = None;
            return Ok(None);
        };
        if self.annotation.view_plane != ViewPlane::XY {
            self.frame_inspection = None;
            self.inspection_key = None;
            return Ok(None);
        }
        let key = InspectionKey {
            position_dir: position.spec.position_dir.clone(),
            segmentation_endname: self.persisted.selected_segmentation_endname.clone(),
            frame_index: self.selected_frame_idx,
            projection: self.current_projection(),
            selected_label: self.current_annotation_label(),
            revision: self
                .current_annotation_document()
                .map(|document| document.revision)
                .unwrap_or(0),
        };
        if self.inspection_key.as_ref() != Some(&key) {
            if key.segmentation_endname.is_some() || self.selected_segmentation_path().is_some() {
                let inspection = inspect_position_frame(FrameInspectionConfig {
                    position_path: position.spec.position_dir.clone(),
                    segm_endname: self.persisted.selected_segmentation_endname.clone(),
                    frame_index: self.selected_frame_idx,
                    projection: self.current_projection(),
                    selected_label: self.current_annotation_label(),
                })?;
                self.frame_inspection = Some(inspection);
            } else {
                self.frame_inspection = None;
            }
            self.inspection_key = Some(key);
        }
        Ok(self.frame_inspection.as_ref())
    }

    pub(crate) fn load_cell_cycle_dialog_state(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let table = cellacdc_rs::load_cell_cycle_annotations(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
        )?;
        self.annotation.cell_cycle_table.records = table.records;
        self.annotation.cell_cycle_dialog.error = None;
        Ok(())
    }

    pub(crate) fn save_cell_cycle_dialog_state(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let existing = cellacdc_rs::load_cell_cycle_annotations(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
        )?;
        let edits = self
            .annotation
            .cell_cycle_table
            .records
            .iter()
            .filter(|record| record.frame_i == self.selected_frame_idx as i64)
            .map(|record| CellCycleEdit {
                frame_i: record.frame_i,
                cell_id: record.cell_id,
                cell_cycle_stage: Some(record.cell_cycle_stage.clone()),
                generation_num: Some(record.generation_num),
                relative_id: Some(record.relative_id),
                relationship: Some(record.relationship.clone()),
                emerg_frame_i: Some(record.emerg_frame_i),
                division_frame_i: Some(record.division_frame_i),
                is_history_known: Some(record.is_history_known),
            })
            .collect::<Vec<_>>();
        let updated = if self.annotation.cell_cycle_dialog.apply_to_future {
            let end_frame_i = self
                .annotation
                .cell_cycle_dialog
                .propagate_end_frame
                .trim()
                .parse::<i64>()
                .ok();
            cellacdc_rs::propagate_cell_cycle_edits(
                &existing,
                &edits,
                &CellCyclePropagationConfig {
                    start_frame_i: self.selected_frame_idx as i64,
                    end_frame_i,
                },
            )?
        } else {
            cellacdc_rs::apply_cell_cycle_edits(&existing, &edits)?
        };
        save_cell_cycle_annotations(&updated)?;
        self.annotation.cell_cycle_table.records = updated.records;
        self.append_log("Saved cell-cycle annotations".to_string());
        Ok(())
    }

    pub(crate) fn assign_selected_bud_to_mother(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let bud_id = self
            .current_annotation_label()
            .ok_or_else(|| anyhow!("Select a bud ID first"))? as i64;
        let mother_id = self
            .annotation
            .mother_target
            .trim()
            .parse::<i64>()
            .with_context(|| "Mother ID must be an integer")?;
        let updated = assign_mother_bud(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx as i64,
            bud_id,
            mother_id,
        )?;
        self.annotation.cell_cycle_table.records = updated.records;
        self.append_log(format!("Assigned bud {bud_id} to mother {mother_id}"));
        Ok(())
    }

    pub(crate) fn refresh_lineage_review(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let review = review_lineage_frame(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx as i64,
        )?;
        self.annotation.lineage_review.review = Some(review);
        Ok(())
    }

    pub(crate) fn select_next_lineage_candidate(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let cell_id = self
            .current_annotation_label()
            .ok_or_else(|| anyhow!("Select an object first"))? as i64;
        if let Some(candidate) = find_next_mother_candidate(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx as i64,
            cell_id,
        )? {
            self.annotation.mother_target = candidate.to_string();
            self.annotation.highlight.searched_label = Some(candidate as u32);
            self.append_log(format!(
                "Suggested mother candidate for {cell_id}: {candidate}"
            ));
        }
        Ok(())
    }

    pub(crate) fn mark_selected_lineage_unknown(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let cell_id = self
            .current_annotation_label()
            .ok_or_else(|| anyhow!("Select an object first"))? as i64;
        mark_unknown_lineage(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx as i64,
            cell_id,
        )?;
        self.append_log(format!("Marked lineage unknown for Cell_ID {cell_id}"));
        self.refresh_lineage_review()?;
        Ok(())
    }

    pub(crate) fn propagate_selected_lineage(&mut self) -> Result<()> {
        let Some(position) = self.selected_position() else {
            return Err(anyhow!("No session is open"));
        };
        let cell_id = self
            .current_annotation_label()
            .ok_or_else(|| anyhow!("Select an object first"))? as i64;
        if !self.annotation.mother_target.trim().is_empty() {
            let parent_id = self
                .annotation
                .mother_target
                .trim()
                .parse::<i64>()
                .with_context(|| "Mother ID must be an integer")?;
            set_lineage_parent_for_position(
                &position.spec.position_dir,
                self.persisted.selected_segmentation_endname.as_deref(),
                LineageFrameEdit {
                    frame_i: self.selected_frame_idx as i64,
                    cell_id,
                    parent_id,
                },
            )?;
        }
        propagate_lineage_for_position(
            &position.spec.position_dir,
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx as i64,
            &[cell_id],
        )?;
        self.append_log(format!(
            "Propagated lineage from frame {} for Cell_ID {}",
            self.selected_frame_idx, cell_id
        ));
        self.refresh_lineage_review()?;
        Ok(())
    }

    pub(crate) fn reveal_current_position(&self) -> Result<()> {
        let path = self
            .selected_position()
            .map(|position| position.spec.position_dir.clone())
            .or_else(|| {
                self.experiment
                    .as_ref()
                    .map(|experiment| experiment.root_path.clone())
            })
            .ok_or_else(|| anyhow!("No session is open"))?;
        let status = if cfg!(target_os = "macos") {
            Command::new("open").arg("-R").arg(&path).status()?
        } else if cfg!(target_os = "windows") {
            Command::new("explorer").arg(&path).status()?
        } else {
            Command::new("xdg-open").arg(&path).status()?
        };
        if !status.success() {
            anyhow::bail!("Failed to reveal {}", path.display());
        }
        Ok(())
    }

    pub(crate) fn prepare_export_defaults(&mut self) {
        let Some(position) = self.selected_position().cloned() else {
            return;
        };
        if self.annotation.export_image.path.trim().is_empty() {
            self.annotation.export_image.path = position
                .spec
                .images_dir
                .join(format!(
                    "{}_gui_frame_{:04}.png",
                    position.spec.basename, self.selected_frame_idx
                ))
                .display()
                .to_string();
        }
        if self.annotation.export_video.output_path.trim().is_empty() {
            self.annotation.export_video.output_path = position
                .spec
                .images_dir
                .join(format!("{}_gui_export", position.spec.basename))
                .display()
                .to_string();
        }
        self.annotation.export_video.start_frame = 0;
        self.annotation.export_video.end_frame = position.spec.size_t.saturating_sub(1);
    }

    pub(crate) fn export_current_image(&mut self) -> Result<PathBuf> {
        let path = PathBuf::from(self.annotation.export_image.path.trim());
        let mut request = self
            .current_render_request()?
            .ok_or_else(|| anyhow!("Nothing is available to export"))?;
        request.overlay.enabled = self.annotation.export_image.include_overlay;
        request.overlay.show_labels = self.annotation.export_image.include_labels;
        request.scale_bar.enabled = self.annotation.export_image.include_scale_bar;
        request.timestamp.enabled = self.annotation.export_image.include_timestamp;
        let exported = export_frame_image(&request, &path)?;
        self.append_log(format!("Exported image -> {}", exported.display()));
        Ok(exported)
    }

    pub(crate) fn export_current_video_or_sequence(&mut self) -> Result<()> {
        let Some(position) = self.selected_position().cloned() else {
            anyhow::bail!("No session is open");
        };
        let start = self
            .annotation
            .export_video
            .start_frame
            .min(position.spec.size_t.saturating_sub(1));
        let end = self
            .annotation
            .export_video
            .end_frame
            .min(position.spec.size_t.saturating_sub(1));
        if end < start {
            anyhow::bail!("Video export range is invalid");
        }
        let mut requests = Vec::new();
        for frame_index in start..=end {
            let frame = position.load_channel_frame_for_view(
                &self.persisted.selected_channel,
                frame_index,
                self.annotation.view_plane,
                self.current_projection(),
            )?;
            let segmentation = if self.annotation.export_video.include_overlay {
                position.load_segmentation_frame_for_view(
                    self.persisted.selected_segmentation_endname.as_deref(),
                    frame_index,
                    self.annotation.view_plane,
                    self.current_projection(),
                )?
            } else {
                None
            };
            requests.push(RenderFrameRequest {
                frame,
                segmentation,
                overlay: OverlayRenderStyle {
                    enabled: self.annotation.export_video.include_overlay,
                    alpha: self.persisted.overlay_alpha,
                    selected_label: self.current_annotation_label(),
                    highlighted_label: self.active_highlighted_label(),
                    show_labels: self.annotation.export_video.include_labels,
                    single_channel_mode: self.persisted.display.overlay_single_channel_mode,
                    true_transparency: self.persisted.display.true_transparency,
                    label_color: self.persisted.display.overlay_label_color,
                    label_scale: self.persisted.display.overlay_label_scale,
                },
                markers: self.custom_annotation_markers_for_frame(frame_index)?,
                scale_bar: ScaleBarStyle {
                    enabled: self.annotation.export_video.include_scale_bar,
                    ..Default::default()
                },
                timestamp: TimestampStyle {
                    enabled: self.annotation.export_video.include_timestamp,
                    ..Default::default()
                },
                frame_index,
                time_seconds: Some(position.spec.time_increment * frame_index as f64),
                physical_size_x: Some(position.spec.physical_size_x),
            });
        }

        let output_path = PathBuf::from(self.annotation.export_video.output_path.trim());
        let extension = output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());
        if extension.as_deref() == Some("mp4") {
            let sequence_dir = output_path.with_extension("");
            let frames =
                export_frame_sequence(&requests, &sequence_dir, "frame", ImageExportFormat::Png)?;
            if !self.try_encode_mp4(&sequence_dir, &output_path)? {
                self.append_log(format!(
                    "ffmpeg was not available. Exported PNG sequence instead -> {} ({} frame(s))",
                    sequence_dir.display(),
                    frames.len()
                ));
            } else {
                self.append_log(format!("Exported video -> {}", output_path.display()));
            }
        } else {
            let frames =
                export_frame_sequence(&requests, &output_path, "frame", ImageExportFormat::Png)?;
            self.append_log(format!(
                "Exported image sequence -> {} ({} frame(s))",
                output_path.display(),
                frames.len()
            ));
        }
        Ok(())
    }

    fn try_encode_mp4(
        &self,
        sequence_dir: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<bool> {
        let status = Command::new("ffmpeg")
            .arg("-y")
            .arg("-framerate")
            .arg("4")
            .arg("-i")
            .arg(sequence_dir.join("frame_%04d.png"))
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg(output_path)
            .status();
        match status {
            Ok(status) => Ok(status.success()),
            Err(_) => Ok(false),
        }
    }
}

pub(crate) fn mask_frame_for_projection(
    data: &MaskData,
    frame_index: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<u32>> {
    let shape = data.values.shape().to_vec();
    let values = data.values.as_slice_memory_order().ok_or_else(|| {
        anyhow!(
            "Mask data is not contiguous: {}",
            data.source_path.display()
        )
    })?;
    match data.layout {
        SegmentationLayout::YX => {
            if frame_index > 0 {
                anyhow::bail!(
                    "Requested frame {} from a single-frame segmentation {}",
                    frame_index,
                    data.source_path.display()
                );
            }
            Ok(FrameData {
                width: shape[1],
                height: shape[0],
                pixels: values.to_vec(),
            })
        }
        SegmentationLayout::TYX => {
            let height = shape[1];
            let width = shape[2];
            if frame_index >= shape[0] {
                anyhow::bail!(
                    "Frame {} is out of bounds for {} frame(s)",
                    frame_index,
                    shape[0]
                );
            }
            let plane_len = height * width;
            let offset = frame_index * plane_len;
            Ok(FrameData {
                width,
                height,
                pixels: values[offset..offset + plane_len].to_vec(),
            })
        }
        SegmentationLayout::ZYX => {
            if frame_index > 0 {
                anyhow::bail!(
                    "Requested frame {} from a single-frame z-stack segmentation {}",
                    frame_index,
                    data.source_path.display()
                );
            }
            project_mask_volume(values, shape[0], shape[1], shape[2], view_plane, projection)
        }
        SegmentationLayout::TZYX => {
            let size_t = shape[0];
            let size_z = shape[1];
            let height = shape[2];
            let width = shape[3];
            if frame_index >= size_t {
                anyhow::bail!(
                    "Frame {} is out of bounds for {} frame(s)",
                    frame_index,
                    size_t
                );
            }
            let plane_len = height * width;
            let frame_offset = frame_index * size_z * plane_len;
            project_mask_volume(
                &values[frame_offset..frame_offset + size_z * plane_len],
                size_z,
                height,
                width,
                view_plane,
                projection,
            )
        }
    }
}

fn project_mask_volume(
    values: &[u32],
    size_z: usize,
    height: usize,
    width: usize,
    view_plane: ViewPlane,
    projection: FrameProjection,
) -> Result<FrameData<u32>> {
    match view_plane {
        ViewPlane::XY => {
            let plane_len = height * width;
            match projection {
                FrameProjection::Max => {
                    let mut pixels = vec![0u32; plane_len];
                    for z in 0..size_z {
                        let start = z * plane_len;
                        for (index, value) in values[start..start + plane_len].iter().enumerate() {
                            pixels[index] = pixels[index].max(*value);
                        }
                    }
                    Ok(FrameData {
                        width,
                        height,
                        pixels,
                    })
                }
                FrameProjection::ZSlice(z_index) => {
                    if z_index >= size_z {
                        anyhow::bail!("Z {} is out of bounds for {} plane(s)", z_index, size_z);
                    }
                    let start = z_index * plane_len;
                    Ok(FrameData {
                        width,
                        height,
                        pixels: values[start..start + plane_len].to_vec(),
                    })
                }
            }
        }
        ViewPlane::XZ => {
            let slice_y = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= height {
                        anyhow::bail!("Y {} is out of bounds for {} row(s)", index, height);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![0u32; size_z * width];
            for z in 0..size_z {
                for x in 0..width {
                    let value = if let Some(y_index) = slice_y {
                        values[z * height * width + y_index * width + x]
                    } else {
                        let mut best = 0u32;
                        for y in 0..height {
                            best = best.max(values[z * height * width + y * width + x]);
                        }
                        best
                    };
                    pixels[z * width + x] = value;
                }
            }
            Ok(FrameData {
                width,
                height: size_z,
                pixels,
            })
        }
        ViewPlane::YZ => {
            let slice_x = match projection {
                FrameProjection::Max => None,
                FrameProjection::ZSlice(index) => {
                    if index >= width {
                        anyhow::bail!("X {} is out of bounds for {} column(s)", index, width);
                    }
                    Some(index)
                }
            };
            let mut pixels = vec![0u32; size_z * height];
            for z in 0..size_z {
                for y in 0..height {
                    let value = if let Some(x_index) = slice_x {
                        values[z * height * width + y * width + x_index]
                    } else {
                        let mut best = 0u32;
                        for x in 0..width {
                            best = best.max(values[z * height * width + y * width + x]);
                        }
                        best
                    };
                    pixels[z * height + y] = value;
                }
            }
            Ok(FrameData {
                width: height,
                height: size_z,
                pixels,
            })
        }
    }
}

impl eframe::App for CellAcdcGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.persisted.data_prep_active_channel = self.data_prep.active_channel.clone();
        self.persisted.data_prep_projection_mode = self.data_prep.projection_mode;
        self.persisted.data_prep_z_index = self.data_prep.z_index;
        self.persisted.annotation_tool = self.annotation.tool;
        self.persisted.annotation_brush_radius = self.annotation.brush_radius;
        self.persisted.gui_mode = self.annotation.mode;
        self.persisted.lineage_tool = self.annotation.lineage_tool;
        self.persisted.view_plane = self.annotation.view_plane;
        self.persisted.show_all_custom_annotations =
            self.annotation.custom_annotation_toolbar.show_all;
        self.persisted.track_ioa_threshold = self.annotation.tracking_params.ioa_threshold;
        if self.pending_annotation_autosave {
            if let Some(document) = self.current_annotation_document_mut() {
                let _ = document.session.save_autosave();
            }
        }
        persist::save(storage, &self.persisted);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_active_job();
        self.request_repaint_for_active_job(ctx);
        self.flush_annotation_autosave_if_due();
        if self.persisted.route == AppRoute::Annotation {
            self.handle_gui_shortcuts(ctx);
        }

        self.draw_shell_bar(ctx);
        match self.persisted.route {
            AppRoute::Launcher => self.draw_launcher_panel(ctx),
            AppRoute::DataStructure => self.draw_data_structure_panel(ctx),
            AppRoute::DataPrep => self.draw_data_prep_panel(ctx),
            AppRoute::Segmentation => self.render_segmentation_workspace(ctx),
            AppRoute::Annotation => self.draw_annotation_panel(ctx),
            AppRoute::Utilities => self.draw_utility_panel(ctx),
            AppRoute::Help => self.draw_help_panel(ctx),
        }
    }
}

impl CellAcdcGui {
    fn handle_gui_shortcuts(&mut self, ctx: &egui::Context) {
        let actions = [
            super::state::GuiActionId::Save,
            super::state::GuiActionId::SaveAsVersion,
            super::state::GuiActionId::Undo,
            super::state::GuiActionId::Redo,
            super::state::GuiActionId::FindId,
            super::state::GuiActionId::ToolSelect,
            super::state::GuiActionId::ToolBrush,
            super::state::GuiActionId::ToolEraser,
            super::state::GuiActionId::HighlightSelectedId,
            super::state::GuiActionId::ManualTracking,
            super::state::GuiActionId::RepeatTracking,
            super::state::GuiActionId::AssignMotherToBud,
            super::state::GuiActionId::UnknownLineage,
            super::state::GuiActionId::NoLineageTool,
            super::state::GuiActionId::PropagateLineage,
        ];
        if let Some(action) =
            super::shortcuts::triggered_action(ctx, &self.persisted.shortcut_overrides, &actions)
        {
            if self.gui_action_state(action).enabled {
                self.dispatch_gui_action(action);
            }
        }
        if !matches!(
            self.annotation.mode,
            super::state::GuiMode::CustomAnnotations | super::state::GuiMode::Snapshot
        ) {
            return;
        }
        let definitions = self
            .annotation
            .custom_annotations
            .definitions
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for definition in definitions {
            let Some(shortcut) = definition.shortcut.as_deref() else {
                continue;
            };
            let Some(key) = super::shortcuts::key_from_name(shortcut) else {
                continue;
            };
            let keyboard_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, key);
            let triggered = ctx.input_mut(|input| input.consume_shortcut(&keyboard_shortcut));
            if !triggered {
                continue;
            }
            if self
                .annotation
                .active_custom_annotation
                .active_name
                .as_deref()
                == Some(definition.name.as_str())
            {
                self.annotation.active_custom_annotation.active_name = None;
            } else {
                self.annotation.active_custom_annotation.active_name = Some(definition.name);
            }
        }
    }
}

fn label_centroids(frame: &FrameData<u32>) -> std::collections::BTreeMap<u32, (usize, usize)> {
    use std::collections::BTreeMap;

    let mut accum = BTreeMap::<u32, (u64, u64, u64)>::new();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let label = frame.pixels[y * frame.width + x];
            if label == 0 {
                continue;
            }
            let entry = accum.entry(label).or_insert((0, 0, 0));
            entry.0 += x as u64;
            entry.1 += y as u64;
            entry.2 += 1;
        }
    }

    accum
        .into_iter()
        .filter_map(|(label, (sum_x, sum_y, count))| {
            if count == 0 {
                None
            } else {
                Some((label, ((sum_x / count) as usize, (sum_y / count) as usize)))
            }
        })
        .collect()
}
