use crate::mask_io::{save_mask_data, MaskData, MaskPathResolution};
use anyhow::{bail, Result};
use ndarray::ArrayD;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskDocumentPaths {
    pub source_path: PathBuf,
    pub autosave_path: PathBuf,
    pub safe_overwrite_path: PathBuf,
}

impl MaskDocumentPaths {
    pub fn from_source_path(path: impl AsRef<Path>) -> Result<Self> {
        let source_path = path.as_ref().to_path_buf();
        let parent = source_path.parent().ok_or_else(|| {
            anyhow::anyhow!("Mask source path has no parent: {}", source_path.display())
        })?;
        let recovery_dir = parent.join("recovery");
        let file_name = source_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Mask source path has no file name"))?;
        let autosave_path = recovery_dir.join(file_name);

        let stem = source_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("segm");
        let safe_name = match source_path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if !ext.is_empty() => format!("{stem}.new.{ext}"),
            _ => format!("{stem}.new"),
        };
        let safe_overwrite_path = parent.join(safe_name);

        Ok(Self {
            source_path,
            autosave_path,
            safe_overwrite_path,
        })
    }

    pub fn recovery_is_newer(&self) -> Result<bool> {
        if !self.autosave_path.exists() {
            return Ok(false);
        }
        if !self.source_path.exists() {
            return Ok(true);
        }
        let autosave_mtime = fs::metadata(&self.autosave_path)?.modified()?;
        let source_mtime = fs::metadata(&self.source_path)?.modified()?;
        Ok(autosave_mtime > source_mtime)
    }

    pub fn discard_recovery(&self) -> Result<()> {
        if self.autosave_path.exists() {
            fs::remove_file(&self.autosave_path)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskSaveMode {
    Overwrite,
    SaveAs(PathBuf),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaskRecoveryState {
    #[default]
    None,
    RecoveryAvailable,
    Recovered,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectionState {
    pub frame_index: usize,
    pub z_index: Option<usize>,
    pub selected_label: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskEditCommand {
    Paint {
        flat_indices: Vec<usize>,
        label: u32,
    },
    Erase {
        flat_indices: Vec<usize>,
    },
    ReplaceLabel {
        from: u32,
        to: u32,
    },
    DeleteLabel {
        label: u32,
    },
    SelectLabel {
        label: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskEditResult {
    pub changed_pixels: usize,
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub struct UndoStack<T> {
    limit: usize,
    past: Vec<T>,
    future: Vec<T>,
}

impl<T: Clone> UndoStack<T> {
    pub fn with_limit(limit: usize) -> Self {
        Self {
            limit,
            past: Vec::new(),
            future: Vec::new(),
        }
    }

    pub fn push_snapshot(&mut self, snapshot: &T) {
        self.past.push(snapshot.clone());
        if self.past.len() > self.limit {
            self.past.remove(0);
        }
        self.future.clear();
    }

    pub fn undo(&mut self, current: &mut T) -> bool {
        let Some(previous) = self.past.pop() else {
            return false;
        };
        self.future.push(current.clone());
        *current = previous;
        true
    }

    pub fn redo(&mut self, current: &mut T) -> bool {
        let Some(next) = self.future.pop() else {
            return false;
        };
        self.past.push(current.clone());
        *current = next;
        true
    }
}

#[derive(Debug, Clone)]
pub struct MaskEditSession {
    path: Option<PathBuf>,
    autosave_path: Option<PathBuf>,
    safe_overwrite_path: Option<PathBuf>,
    recovery_state: MaskRecoveryState,
    pub data: MaskData,
    pub selection: SelectionState,
    pub dirty: bool,
    pub undo_stack: UndoStack<ArrayD<u32>>,
}

impl MaskEditSession {
    pub fn new(path: Option<PathBuf>, data: MaskData) -> Self {
        let document_paths = path
            .as_ref()
            .and_then(|path| MaskDocumentPaths::from_source_path(path).ok());
        Self {
            path,
            autosave_path: document_paths
                .as_ref()
                .map(|paths| paths.autosave_path.clone()),
            safe_overwrite_path: document_paths
                .as_ref()
                .map(|paths| paths.safe_overwrite_path.clone()),
            recovery_state: MaskRecoveryState::None,
            data,
            selection: SelectionState::default(),
            dirty: false,
            undo_stack: UndoStack::with_limit(32),
        }
    }

    pub fn with_autosave_path(mut self, path: PathBuf) -> Self {
        self.autosave_path = Some(path);
        self
    }

    pub fn with_document_paths(mut self, paths: &MaskDocumentPaths) -> Self {
        self.path = Some(paths.source_path.clone());
        self.autosave_path = Some(paths.autosave_path.clone());
        self.safe_overwrite_path = Some(paths.safe_overwrite_path.clone());
        self
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn autosave_path(&self) -> Option<&Path> {
        self.autosave_path.as_deref()
    }

    pub fn safe_overwrite_path(&self) -> Option<&Path> {
        self.safe_overwrite_path.as_deref()
    }

    pub fn document_paths(&self) -> Option<MaskDocumentPaths> {
        match (
            self.path.clone(),
            self.autosave_path.clone(),
            self.safe_overwrite_path.clone(),
        ) {
            (Some(source_path), Some(autosave_path), Some(safe_overwrite_path)) => {
                Some(MaskDocumentPaths {
                    source_path,
                    autosave_path,
                    safe_overwrite_path,
                })
            }
            _ => None,
        }
    }

    pub fn recovery_state(&self) -> MaskRecoveryState {
        self.recovery_state
    }

    pub fn set_recovery_state(&mut self, recovery_state: MaskRecoveryState) {
        self.recovery_state = recovery_state;
    }

    pub fn apply_command(&mut self, command: MaskEditCommand) -> Result<MaskEditResult> {
        if matches!(command, MaskEditCommand::SelectLabel { .. }) {
            if let MaskEditCommand::SelectLabel { label } = command {
                self.selection.selected_label = label;
            }
            return Ok(MaskEditResult {
                changed_pixels: 0,
                dirty: self.dirty,
            });
        }

        self.undo_stack.push_snapshot(&self.data.values);
        let values = self
            .data
            .values
            .as_slice_memory_order_mut()
            .ok_or_else(|| anyhow::anyhow!("Mask data is not contiguous"))?;
        let mut changed = 0usize;
        match command {
            MaskEditCommand::Paint {
                flat_indices,
                label,
            } => {
                for index in flat_indices {
                    if let Some(pixel) = values.get_mut(index) {
                        if *pixel != label {
                            *pixel = label;
                            changed += 1;
                        }
                    }
                }
            }
            MaskEditCommand::Erase { flat_indices } => {
                for index in flat_indices {
                    if let Some(pixel) = values.get_mut(index) {
                        if *pixel != 0 {
                            *pixel = 0;
                            changed += 1;
                        }
                    }
                }
            }
            MaskEditCommand::ReplaceLabel { from, to } => {
                for pixel in values.iter_mut() {
                    if *pixel == from {
                        *pixel = to;
                        changed += 1;
                    }
                }
            }
            MaskEditCommand::DeleteLabel { label } => {
                for pixel in values.iter_mut() {
                    if *pixel == label {
                        *pixel = 0;
                        changed += 1;
                    }
                }
            }
            MaskEditCommand::SelectLabel { .. } => unreachable!(),
        }
        self.dirty |= changed > 0;
        Ok(MaskEditResult {
            changed_pixels: changed,
            dirty: self.dirty,
        })
    }

    pub fn undo(&mut self) -> bool {
        let undone = self.undo_stack.undo(&mut self.data.values);
        self.dirty |= undone;
        undone
    }

    pub fn redo(&mut self) -> bool {
        let redone = self.undo_stack.redo(&mut self.data.values);
        self.dirty |= redone;
        redone
    }

    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        save_mask_data(path, &self.data)?;
        if let Ok(paths) = MaskDocumentPaths::from_source_path(path) {
            self.path = Some(paths.source_path);
            self.autosave_path = Some(paths.autosave_path);
            self.safe_overwrite_path = Some(paths.safe_overwrite_path);
        } else {
            self.path = Some(path.to_path_buf());
        }
        self.dirty = false;
        self.recovery_state = MaskRecoveryState::None;
        Ok(())
    }

    pub fn save_autosave(&self) -> Result<PathBuf> {
        let path = if let Some(path) = self.autosave_path.as_ref() {
            path.clone()
        } else if let Some(source_path) = self.path.as_ref() {
            MaskDocumentPaths::from_source_path(source_path)?.autosave_path
        } else {
            bail!("Autosave path is not configured");
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        save_mask_data(&path, &self.data)?;
        Ok(path)
    }

    pub fn save_with_mode(&mut self, mode: MaskSaveMode) -> Result<PathBuf> {
        match mode {
            MaskSaveMode::Overwrite => self.safe_overwrite(),
            MaskSaveMode::SaveAs(path) => {
                self.save(&path)?;
                Ok(path)
            }
        }
    }

    pub fn safe_overwrite(&mut self) -> Result<PathBuf> {
        let Some(source_path) = self.path.as_ref() else {
            bail!("Mask source path is not configured");
        };
        let paths = self
            .document_paths()
            .unwrap_or(MaskDocumentPaths::from_source_path(source_path)?);
        save_mask_data(&paths.safe_overwrite_path, &self.data)?;
        if source_path.exists() {
            fs::remove_file(source_path)?;
        }
        fs::rename(&paths.safe_overwrite_path, source_path)?;
        self.dirty = false;
        self.recovery_state = MaskRecoveryState::None;
        paths.discard_recovery()?;
        self.autosave_path = Some(paths.autosave_path.clone());
        self.safe_overwrite_path = Some(paths.safe_overwrite_path.clone());
        Ok(source_path.clone())
    }

    pub fn discard_recovery(&mut self) -> Result<()> {
        if let Some(paths) = self.document_paths() {
            paths.discard_recovery()?;
        } else if let Some(path) = self.autosave_path.as_ref() {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        self.recovery_state = MaskRecoveryState::None;
        Ok(())
    }

    pub fn from_source_path(
        path: impl AsRef<Path>,
        resolution: Option<&MaskPathResolution>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let paths = MaskDocumentPaths::from_source_path(&path)?;
        let data = crate::mask_io::load_mask_data(&path, resolution)?;
        let mut session = Self::new(Some(path), data).with_document_paths(&paths);
        if paths.recovery_is_newer()? {
            session.recovery_state = MaskRecoveryState::RecoveryAvailable;
        }
        Ok(session)
    }

    pub fn load_with_recovery(
        path: impl AsRef<Path>,
        resolution: Option<&MaskPathResolution>,
        prefer_recovery: bool,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let paths = MaskDocumentPaths::from_source_path(&path)?;
        if prefer_recovery && paths.recovery_is_newer()? {
            let data = crate::mask_io::load_mask_data(&paths.autosave_path, resolution)?;
            let mut session = Self::new(Some(path), data).with_document_paths(&paths);
            session.recovery_state = MaskRecoveryState::Recovered;
            return Ok(session);
        }
        Self::from_source_path(path, resolution)
    }

    pub fn recover_from_autosave(
        path: impl AsRef<Path>,
        resolution: Option<&MaskPathResolution>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data = crate::mask_io::load_mask_data(&path, resolution)?;
        let mut session = Self::new(None, data).with_autosave_path(path);
        session.recovery_state = MaskRecoveryState::Recovered;
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask_io::SegmentationLayout;
    use ndarray::ArrayD;
    use tempfile::tempdir;

    fn sample_mask() -> MaskData {
        MaskData {
            values: ArrayD::from_shape_vec(vec![2, 2], vec![1, 1, 0, 2]).unwrap(),
            layout: SegmentationLayout::YX,
            source_path: PathBuf::from("sample_segm.npz"),
        }
    }

    #[test]
    fn edits_and_undoes_mask() -> Result<()> {
        let mut session = MaskEditSession::new(None, sample_mask());
        let result = session.apply_command(MaskEditCommand::ReplaceLabel { from: 1, to: 3 })?;
        assert_eq!(result.changed_pixels, 2);
        assert!(session.undo());
        assert_eq!(
            session
                .data
                .values
                .as_slice_memory_order()
                .unwrap()
                .to_vec(),
            vec![1, 1, 0, 2]
        );
        Ok(())
    }

    #[test]
    fn autosaves_current_mask() -> Result<()> {
        let dir = tempdir()?;
        let autosave_path = dir.path().join("autosave_segm.npz");
        let session = MaskEditSession::new(None, sample_mask()).with_autosave_path(autosave_path);
        let saved = session.save_autosave()?;
        assert!(saved.exists());
        Ok(())
    }

    #[test]
    fn safe_overwrites_current_mask() -> Result<()> {
        let dir = tempdir()?;
        let source_path = dir.path().join("demo_segm.npz");
        save_mask_data(&source_path, &sample_mask())?;

        let mut session = MaskEditSession::from_source_path(&source_path, None)?;
        session.apply_command(MaskEditCommand::ReplaceLabel { from: 1, to: 8 })?;
        let saved = session.safe_overwrite()?;
        let loaded = crate::mask_io::load_mask_data(&saved, None)?;
        assert!(loaded.values.iter().all(|value| *value != 1));
        Ok(())
    }

    #[test]
    fn detects_newer_recovery_file() -> Result<()> {
        let dir = tempdir()?;
        let source_path = dir.path().join("demo_segm.npz");
        save_mask_data(&source_path, &sample_mask())?;
        let paths = MaskDocumentPaths::from_source_path(&source_path)?;
        std::fs::create_dir_all(paths.autosave_path.parent().unwrap())?;
        std::thread::sleep(std::time::Duration::from_millis(10));
        save_mask_data(&paths.autosave_path, &sample_mask())?;

        let session = MaskEditSession::from_source_path(&source_path, None)?;
        assert_eq!(
            session.recovery_state(),
            MaskRecoveryState::RecoveryAvailable
        );
        Ok(())
    }
}
