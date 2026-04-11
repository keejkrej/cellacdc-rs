mod autosave_interval;
mod export_image;
mod export_video;
mod find_id;
mod overlay_labels;
mod shortcut_editor;
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
    }
}
