use crate::session::FrameData;
use anyhow::{anyhow, bail, Result};
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageExportFormat {
    Png,
    Tiff,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayRenderStyle {
    pub enabled: bool,
    pub alpha: f32,
    pub selected_label: Option<u32>,
    pub highlighted_label: Option<u32>,
    pub show_labels: bool,
    pub single_channel_mode: bool,
    pub true_transparency: bool,
    pub label_color: [u8; 4],
    pub label_scale: u32,
}

impl Default for OverlayRenderStyle {
    fn default() -> Self {
        Self {
            enabled: true,
            alpha: 0.45,
            selected_label: None,
            highlighted_label: None,
            show_labels: false,
            single_channel_mode: false,
            true_transparency: false,
            label_color: [255, 255, 255, 255],
            label_scale: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleBarStyle {
    pub enabled: bool,
    pub length_um: f64,
    pub thickness_px: u32,
    pub color: [u8; 4],
    pub margin_px: u32,
}

impl Default for ScaleBarStyle {
    fn default() -> Self {
        Self {
            enabled: false,
            length_um: 5.0,
            thickness_px: 4,
            color: [255, 255, 255, 255],
            margin_px: 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimestampStyle {
    pub enabled: bool,
    pub color: [u8; 4],
}

impl Default for TimestampStyle {
    fn default() -> Self {
        Self {
            enabled: false,
            color: [255, 255, 255, 255],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayMarker {
    pub x: usize,
    pub y: usize,
    pub symbol: String,
    pub color: [u8; 4],
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderFrameRequest {
    pub frame: FrameData<f32>,
    pub segmentation: Option<FrameData<u32>>,
    pub overlay: OverlayRenderStyle,
    pub markers: Vec<OverlayMarker>,
    pub scale_bar: ScaleBarStyle,
    pub timestamp: TimestampStyle,
    pub frame_index: usize,
    pub time_seconds: Option<f64>,
    pub physical_size_x: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedFrame {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

pub fn render_frame(request: &RenderFrameRequest) -> Result<RenderedFrame> {
    if let Some(segm) = &request.segmentation {
        if segm.width != request.frame.width || segm.height != request.frame.height {
            bail!(
                "Segmentation size {}x{} does not match frame size {}x{}",
                segm.width,
                segm.height,
                request.frame.width,
                request.frame.height
            );
        }
    }
    let (min_value, max_value) = request.frame.pixels.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(*value), max_v.max(*value)),
    );
    let denom = (max_value - min_value).max(f32::EPSILON);
    let mut rgba = Vec::with_capacity(request.frame.pixels.len() * 4);
    for (index, value) in request.frame.pixels.iter().enumerate() {
        let normalized = (((*value - min_value) / denom).clamp(0.0, 1.0) * 255.0) as u8;
        let mut color = [normalized, normalized, normalized, 255];
        if request.overlay.enabled {
            if let Some(segm) = &request.segmentation {
                let label = segm.pixels[index];
                if label != 0 {
                    let is_selected = request.overlay.selected_label == Some(label);
                    let is_highlighted = request.overlay.highlighted_label == Some(label);
                    let overlay =
                        overlay_color(label, &request.overlay, is_selected, is_highlighted);
                    let alpha = if request.overlay.true_transparency {
                        (request.overlay.alpha * 0.75).clamp(0.0, 1.0)
                    } else if is_selected || is_highlighted {
                        request.overlay.alpha.max(0.82)
                    } else {
                        request.overlay.alpha
                    };
                    color[0] = blend_channel(color[0], overlay[0], alpha);
                    color[1] = blend_channel(color[1], overlay[1], alpha);
                    color[2] = blend_channel(color[2], overlay[2], alpha);
                }
            }
        }
        rgba.extend_from_slice(&color);
    }

    if request.overlay.enabled && request.overlay.show_labels {
        if let Some(segm) = &request.segmentation {
            draw_label_annotations(
                &mut rgba,
                request.frame.width,
                request.frame.height,
                segm,
                request.overlay.label_color,
                request.overlay.label_scale.max(1),
            );
        }
    }

    if !request.markers.is_empty() {
        for marker in &request.markers {
            draw_overlay_marker(&mut rgba, request.frame.width, request.frame.height, marker);
        }
    }

    if request.scale_bar.enabled {
        if let Some(physical_size_x) = request.physical_size_x {
            draw_scale_bar(
                &mut rgba,
                request.frame.width,
                request.frame.height,
                &request.scale_bar,
                physical_size_x,
            );
        }
    }

    if request.timestamp.enabled {
        let timestamp = match request.time_seconds {
            Some(time_seconds) => format!("t={time_seconds:.1}s  F={}", request.frame_index),
            None => format!("F={}", request.frame_index),
        };
        draw_text(
            &mut rgba,
            request.frame.width,
            request.frame.height,
            10,
            10,
            &timestamp,
            request.timestamp.color,
            2,
        );
    }

    Ok(RenderedFrame {
        width: request.frame.width,
        height: request.frame.height,
        rgba,
    })
}

pub fn export_frame_image(
    request: &RenderFrameRequest,
    output_path: impl AsRef<Path>,
) -> Result<PathBuf> {
    let rendered = render_frame(request)?;
    write_rendered_frame(&rendered, output_path.as_ref())?;
    Ok(output_path.as_ref().to_path_buf())
}

pub fn export_frame_sequence(
    requests: &[RenderFrameRequest],
    output_dir: impl AsRef<Path>,
    stem: &str,
    format: ImageExportFormat,
) -> Result<Vec<PathBuf>> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;
    let ext = match format {
        ImageExportFormat::Png => "png",
        ImageExportFormat::Tiff => "tiff",
    };
    let mut outputs = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let path = output_dir.join(format!("{stem}_{index:04}.{ext}"));
        export_frame_image(request, &path)?;
        outputs.push(path);
    }
    Ok(outputs)
}

fn write_rendered_frame(rendered: &RenderedFrame, output_path: &Path) -> Result<()> {
    let Some(parent) = output_path.parent() else {
        return Err(anyhow!(
            "Export path has no parent: {}",
            output_path.display()
        ));
    };
    std::fs::create_dir_all(parent)?;
    let format = match output_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => ImageFormat::Png,
        Some("tif") | Some("tiff") => ImageFormat::Tiff,
        other => bail!(
            "Unsupported export format {:?} for {}. Use .png or .tiff",
            other,
            output_path.display()
        ),
    };
    image::save_buffer_with_format(
        output_path,
        &rendered.rgba,
        rendered.width as u32,
        rendered.height as u32,
        image::ColorType::Rgba8,
        format,
    )?;
    Ok(())
}

fn overlay_color(
    label: u32,
    overlay: &OverlayRenderStyle,
    is_selected: bool,
    is_highlighted: bool,
) -> [u8; 3] {
    if overlay.single_channel_mode {
        if is_selected || is_highlighted {
            [255, 230, 120]
        } else {
            [255, 90, 90]
        }
    } else if is_selected {
        [255, 240, 120]
    } else if is_highlighted {
        [120, 220, 255]
    } else {
        let hash = label.wrapping_mul(0x9E37_79B9);
        [
            ((hash & 0xFF) as u8).max(60),
            (((hash >> 8) & 0xFF) as u8).max(60),
            (((hash >> 16) & 0xFF) as u8).max(60),
        ]
    }
}

fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    ((base as f32) * (1.0 - alpha) + (overlay as f32) * alpha).round() as u8
}

fn draw_scale_bar(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    style: &ScaleBarStyle,
    physical_size_x: f64,
) {
    if physical_size_x <= 0.0 {
        return;
    }
    let length_px = (style.length_um / physical_size_x).round().max(1.0) as usize;
    let margin = style.margin_px as usize;
    let thickness = style.thickness_px.max(1) as usize;
    let y0 = height.saturating_sub(margin + thickness);
    let x0 = width.saturating_sub(margin + length_px);
    for y in y0..(y0 + thickness).min(height) {
        for x in x0..(x0 + length_px).min(width) {
            set_rgba_pixel(rgba, width, x, y, style.color);
        }
    }
    draw_text(
        rgba,
        width,
        height,
        x0 as i32,
        y0.saturating_sub(16) as i32,
        &format!("{} um", style.length_um.round() as i32),
        style.color,
        2,
    );
}

fn draw_label_annotations(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    segmentation: &FrameData<u32>,
    color: [u8; 4],
    scale: u32,
) {
    let mut sums = BTreeMap::<u32, (usize, usize, usize)>::new();
    for y in 0..height {
        for x in 0..width {
            let label = segmentation.pixels[y * width + x];
            if label == 0 {
                continue;
            }
            let entry = sums.entry(label).or_insert((0, 0, 0));
            entry.0 += x;
            entry.1 += y;
            entry.2 += 1;
        }
    }
    for (label, (sum_x, sum_y, count)) in sums {
        if count == 0 {
            continue;
        }
        let x = (sum_x / count) as i32;
        let y = (sum_y / count) as i32;
        draw_text(rgba, width, height, x, y, &label.to_string(), color, scale);
    }
}

fn draw_overlay_marker(rgba: &mut [u8], width: usize, height: usize, marker: &OverlayMarker) {
    let radius = marker.size.max(5) as i32 / 2;
    let cx = marker.x as i32;
    let cy = marker.y as i32;
    match marker.symbol.as_str() {
        "s" => {
            for y in (cy - radius)..=(cy + radius) {
                for x in (cx - radius)..=(cx + radius) {
                    if within_bounds(x, y, width, height) {
                        set_rgba_pixel(rgba, width, x as usize, y as usize, marker.color);
                    }
                }
            }
        }
        "+" => {
            for delta in -radius..=radius {
                if within_bounds(cx + delta, cy, width, height) {
                    set_rgba_pixel(
                        rgba,
                        width,
                        (cx + delta) as usize,
                        cy as usize,
                        marker.color,
                    );
                }
                if within_bounds(cx, cy + delta, width, height) {
                    set_rgba_pixel(
                        rgba,
                        width,
                        cx as usize,
                        (cy + delta) as usize,
                        marker.color,
                    );
                }
            }
        }
        "x" => {
            for delta in -radius..=radius {
                if within_bounds(cx + delta, cy + delta, width, height) {
                    set_rgba_pixel(
                        rgba,
                        width,
                        (cx + delta) as usize,
                        (cy + delta) as usize,
                        marker.color,
                    );
                }
                if within_bounds(cx + delta, cy - delta, width, height) {
                    set_rgba_pixel(
                        rgba,
                        width,
                        (cx + delta) as usize,
                        (cy - delta) as usize,
                        marker.color,
                    );
                }
            }
        }
        "d" => {
            for dy in -radius..=radius {
                let span = radius - dy.abs();
                for dx in -span..=span {
                    if within_bounds(cx + dx, cy + dy, width, height) {
                        set_rgba_pixel(
                            rgba,
                            width,
                            (cx + dx) as usize,
                            (cy + dy) as usize,
                            marker.color,
                        );
                    }
                }
            }
        }
        "t" => {
            for dy in 0..=radius {
                let span = dy;
                for dx in -span..=span {
                    let x = cx + dx;
                    let y = cy + radius - dy;
                    if within_bounds(x, y, width, height) {
                        set_rgba_pixel(rgba, width, x as usize, y as usize, marker.color);
                    }
                }
            }
        }
        _ => {
            for y in (cy - radius)..=(cy + radius) {
                for x in (cx - radius)..=(cx + radius) {
                    let dx = x - cx;
                    let dy = y - cy;
                    if dx * dx + dy * dy > radius * radius {
                        continue;
                    }
                    if within_bounds(x, y, width, height) {
                        set_rgba_pixel(rgba, width, x as usize, y as usize, marker.color);
                    }
                }
            }
        }
    }
}

fn within_bounds(x: i32, y: i32, width: usize, height: usize) -> bool {
    x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height
}

fn draw_text(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    text: &str,
    color: [u8; 4],
    scale: u32,
) {
    let mut cursor_x = x;
    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += (4 * scale as i32).max(4);
            continue;
        }
        if let Some(bitmap) = glyph_bitmap(ch) {
            for (row_idx, row) in bitmap.iter().enumerate() {
                for (col_idx, pixel) in row.chars().enumerate() {
                    if pixel != '1' {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cursor_x + (col_idx as u32 * scale + dx) as i32;
                            let py = y + (row_idx as u32 * scale + dy) as i32;
                            if px < 0 || py < 0 {
                                continue;
                            }
                            let px = px as usize;
                            let py = py as usize;
                            if px >= width || py >= height {
                                continue;
                            }
                            set_rgba_pixel(rgba, width, px, py, color);
                        }
                    }
                }
            }
            cursor_x += ((bitmap[0].len() as u32 + 1) * scale) as i32;
        }
    }
}

fn set_rgba_pixel(rgba: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 4]) {
    let index = (y * width + x) * 4;
    if index + 3 < rgba.len() {
        rgba[index..index + 4].copy_from_slice(&color);
    }
}

fn glyph_bitmap(ch: char) -> Option<&'static [&'static str]> {
    match ch {
        '0' => Some(&["111", "101", "101", "101", "111"]),
        '1' => Some(&["010", "110", "010", "010", "111"]),
        '2' => Some(&["111", "001", "111", "100", "111"]),
        '3' => Some(&["111", "001", "111", "001", "111"]),
        '4' => Some(&["101", "101", "111", "001", "001"]),
        '5' => Some(&["111", "100", "111", "001", "111"]),
        '6' => Some(&["111", "100", "111", "101", "111"]),
        '7' => Some(&["111", "001", "010", "100", "100"]),
        '8' => Some(&["111", "101", "111", "101", "111"]),
        '9' => Some(&["111", "101", "111", "001", "111"]),
        'F' => Some(&["111", "100", "111", "100", "100"]),
        't' => Some(&["010", "111", "010", "010", "001"]),
        '=' => Some(&["000", "111", "000", "111", "000"]),
        '.' => Some(&["000", "000", "000", "000", "010"]),
        's' => Some(&["111", "100", "111", "001", "111"]),
        'u' => Some(&["000", "101", "101", "101", "111"]),
        'm' => Some(&["000", "111", "111", "101", "101"]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_request() -> RenderFrameRequest {
        RenderFrameRequest {
            frame: FrameData {
                width: 3,
                height: 3,
                pixels: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
            segmentation: Some(FrameData {
                width: 3,
                height: 3,
                pixels: vec![0, 1, 1, 0, 0, 2, 2, 2, 2],
            }),
            overlay: OverlayRenderStyle {
                enabled: true,
                show_labels: true,
                ..Default::default()
            },
            scale_bar: ScaleBarStyle::default(),
            timestamp: TimestampStyle::default(),
            frame_index: 0,
            time_seconds: Some(0.0),
            physical_size_x: Some(0.1),
            markers: Vec::new(),
        }
    }

    #[test]
    fn exports_png_frame() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("frame.png");
        export_frame_image(&sample_request(), &path)?;
        assert!(path.exists());
        Ok(())
    }
}
