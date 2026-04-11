use crate::mask_io::{save_mask_data, MaskData, MaskPathResolution};
use anyhow::{bail, Result};
use ndarray::ArrayD;
use std::path::{Path, PathBuf};

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
    pub data: MaskData,
    pub selection: SelectionState,
    pub dirty: bool,
    pub undo_stack: UndoStack<ArrayD<u32>>,
}

impl MaskEditSession {
    pub fn new(path: Option<PathBuf>, data: MaskData) -> Self {
        Self {
            path,
            autosave_path: None,
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

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn autosave_path(&self) -> Option<&Path> {
        self.autosave_path.as_deref()
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
        save_mask_data(path.as_ref(), &self.data)?;
        self.dirty = false;
        Ok(())
    }

    pub fn save_autosave(&self) -> Result<PathBuf> {
        let Some(path) = self.autosave_path.as_ref() else {
            bail!("Autosave path is not configured");
        };
        save_mask_data(path, &self.data)?;
        Ok(path.clone())
    }

    pub fn recover_from_autosave(
        path: impl AsRef<Path>,
        resolution: Option<&MaskPathResolution>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data = crate::mask_io::load_mask_data(&path, resolution)?;
        Ok(Self::new(None, data).with_autosave_path(path))
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
}
