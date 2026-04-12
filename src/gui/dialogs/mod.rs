mod autosave_interval;
mod cell_cycle_editor;
mod cell_cycle_viewer;
mod custom_annotation_editor;
mod export_image;
mod export_video;
mod find_id;
mod lineage_review;
mod load_saved_custom_annotations;
mod overlay_labels;
mod shortcut_editor;
mod snapshot_save_scope;
mod tracking_params;
mod version_browser;

use super::app::CellAcdcGui;
use eframe::egui;

impl CellAcdcGui {
    pub(crate) fn draw_gui_dialogs(&mut self, ctx: &egui::Context) {
        self.draw_find_id_dialog(ctx);
        self.draw_shortcut_editor_dialog(ctx);
        self.draw_version_browser_dialog(ctx);
        self.draw_export_image_dialog(ctx);
        self.draw_export_video_dialog(ctx);
        self.draw_autosave_interval_dialog(ctx);
        self.draw_overlay_labels_dialog(ctx);
        self.draw_tracking_params_dialog(ctx);
        self.draw_cell_cycle_editor_dialog(ctx);
        self.draw_cell_cycle_viewer_dialog(ctx);
        self.draw_lineage_review_dialog(ctx);
        self.draw_custom_annotation_editor_dialog(ctx);
        self.draw_load_saved_custom_annotations_dialog(ctx);
        self.draw_snapshot_save_scope_dialog(ctx);
    }
}
