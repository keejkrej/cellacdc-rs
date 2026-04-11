use super::jobs::JobHandle;
use super::persist::{self, PersistedState};
use super::state::{
    AnnotationPendingAction, AnnotationWorkspaceState, AppRoute, InspectionKey,
    LoadedMaskDocument, ProjectionMode, ViewKey,
};
use super::workspaces;
use anyhow::{anyhow, Result};
use cellacdc_rs::{
    inspect_position_frame, open_experiment_session, export_frame_image, export_frame_sequence,
    ExperimentSession, FrameData, FrameInspection, FrameInspectionConfig, FrameProjection,
    ImageExportFormat, ImportSource, MaskData, MaskEditCommand, MaskEditSession,
    MaskPathResolution, MaskRecoveryState, MaskSaveMode, OverlayRenderStyle, OverwritePolicy,
    PositionSession, RenderFrameRequest, ScaleBarStyle, SegmentationLayout,
    SegmentationParams, TimestampStyle,
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
    pub(crate) data_structure_scan_path: String,
    pub(crate) data_structure_scan_results: Vec<ImportSource>,
    pub(crate) data_structure_scan_error: Option<String>,
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
        let last_non_launcher_route = match persisted.route {
            AppRoute::Launcher => AppRoute::Segmentation,
            route => route,
        };
        let mut app = Self {
            annotation: AnnotationWorkspaceState {
                tool: persisted.annotation_tool,
                brush_radius: persisted.annotation_brush_radius,
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
            data_structure_scan_path: String::new(),
            data_structure_scan_results: Vec::new(),
            data_structure_scan_error: None,
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
        } else {
            self.persisted.z_index = self.persisted.z_index.min(size_z - 1);
        }
        self.annotation.save_as_endname = self
            .persisted
            .selected_segmentation_endname
            .clone()
            .unwrap_or_else(|| "edited".to_string());
    }

    pub(crate) fn current_projection(&self) -> FrameProjection {
        match self.persisted.projection_mode {
            ProjectionMode::Max => FrameProjection::Max,
            ProjectionMode::ZSlice => FrameProjection::ZSlice(self.persisted.z_index),
        }
    }

    pub(crate) fn current_view_key(&self) -> Option<ViewKey> {
        let position = self.selected_position()?;
        Some(ViewKey {
            position_dir: position.spec.position_dir.clone(),
            channel: self.persisted.selected_channel.clone(),
            frame_index: self.selected_frame_idx,
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
        self.clear_annotation_document();
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
        let document = self
            .current_annotation_document_mut()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let saved = document.session.save_with_mode(MaskSaveMode::Overwrite)?;
        document.revision += 1;
        self.pending_annotation_autosave = false;
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
        self.persisted.selected_segmentation_endname = Some(endname.clone());
        self.append_log(format!("Saved GUI version -> {}", saved.display()));
        self.reload_experiment();
        self.clear_annotation_document();
        self.ensure_annotation_document_loaded();
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn run_annotation_command(&mut self, command: MaskEditCommand) -> Result<()> {
        let document = self
            .current_annotation_document_mut()
            .ok_or_else(|| anyhow!("No GUI mask document is loaded"))?;
        let result = document.session.apply_command(command)?;
        if result.changed_pixels > 0 {
            document.revision += 1;
            self.pending_annotation_autosave = true;
            self.last_annotation_change_at = Some(Instant::now());
        } else {
            document.revision += 1;
        }
        self.last_error = None;
        self.invalidate_texture();
        Ok(())
    }

    pub(crate) fn annotation_undo(&mut self) {
        let mut changed = false;
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
    }

    pub(crate) fn annotation_redo(&mut self) {
        let mut changed = false;
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
                    self.current_projection(),
                )?));
            }
        }
        position.load_segmentation_frame(
            self.persisted.selected_segmentation_endname.as_deref(),
            self.selected_frame_idx,
            self.current_projection(),
        )
    }

    pub(crate) fn current_render_request(&self) -> Result<Option<RenderFrameRequest>> {
        let Some(position) = self.selected_position() else {
            return Ok(None);
        };
        let frame = position.load_channel_frame(
            &self.persisted.selected_channel,
            self.selected_frame_idx,
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
                .join(format!("{}_gui_frame_{:04}.png", position.spec.basename, self.selected_frame_idx))
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
        let start = self.annotation.export_video.start_frame.min(position.spec.size_t.saturating_sub(1));
        let end = self.annotation.export_video.end_frame.min(position.spec.size_t.saturating_sub(1));
        if end < start {
            anyhow::bail!("Video export range is invalid");
        }
        let mut requests = Vec::new();
        for frame_index in start..=end {
            let frame = position.load_channel_frame(
                &self.persisted.selected_channel,
                frame_index,
                self.current_projection(),
            )?;
            let segmentation = if self.annotation.export_video.include_overlay {
                position.load_segmentation_frame(
                    self.persisted.selected_segmentation_endname.as_deref(),
                    frame_index,
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
            let frames = export_frame_sequence(
                &requests,
                &sequence_dir,
                "frame",
                ImageExportFormat::Png,
            )?;
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
            let frames = export_frame_sequence(
                &requests,
                &output_path,
                "frame",
                ImageExportFormat::Png,
            )?;
            self.append_log(format!(
                "Exported image sequence -> {} ({} frame(s))",
                output_path.display(),
                frames.len()
            ));
        }
        Ok(())
    }

    fn try_encode_mp4(&self, sequence_dir: &std::path::Path, output_path: &std::path::Path) -> Result<bool> {
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
            project_mask_volume(values, shape[0], shape[1], shape[2], projection)
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
    projection: FrameProjection,
) -> Result<FrameData<u32>> {
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

impl eframe::App for CellAcdcGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.persisted.annotation_tool = self.annotation.tool;
        self.persisted.annotation_brush_radius = self.annotation.brush_radius;
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
        ];
        if let Some(action) =
            super::shortcuts::triggered_action(ctx, &self.persisted.shortcut_overrides, &actions)
        {
            if self.gui_action_state(action).enabled {
                self.dispatch_gui_action(action);
            }
        }
    }
}
