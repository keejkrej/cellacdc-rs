use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingConfig {
    pub ioa_threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedSequence {
    pub frames: Vec<Vec<u32>>,
    pub labels_found: u32,
    pub disappeared: Vec<(usize, u32)>,
}

pub fn track_sequence(
    frames: &[Vec<u32>],
    height: usize,
    width: usize,
    config: &TrackingConfig,
) -> TrackedSequence {
    if frames.is_empty() {
        return TrackedSequence {
            frames: Vec::new(),
            labels_found: 0,
            disappeared: Vec::new(),
        };
    }

    let mut tracked_frames = Vec::with_capacity(frames.len());
    let mut disappeared = Vec::new();

    let first_frame = frames[0].clone();
    let mut labels_found = first_frame.iter().copied().max().unwrap_or(0);
    tracked_frames.push(first_frame.clone());

    let mut prev_frame = first_frame;
    for (frame_i, frame) in frames.iter().enumerate().skip(1) {
        let tracked_frame = track_frame(&prev_frame, frame, height, width, config);
        labels_found = labels_found.max(tracked_frame.iter().copied().max().unwrap_or(0));

        let prev_labels = unique_nonzero(prev_frame.iter().copied());
        let current_labels = unique_nonzero(tracked_frame.iter().copied());
        disappeared.extend(
            prev_labels
                .difference(&current_labels)
                .copied()
                .map(|cell_id| (frame_i - 1, cell_id)),
        );

        prev_frame = tracked_frame.clone();
        tracked_frames.push(tracked_frame);
    }

    TrackedSequence {
        frames: tracked_frames,
        labels_found,
        disappeared,
    }
}

fn track_frame(
    prev_frame: &[u32],
    frame: &[u32],
    height: usize,
    width: usize,
    config: &TrackingConfig,
) -> Vec<u32> {
    if !frame.iter().any(|label| *label != 0) {
        return frame.to_vec();
    }

    let (ioa_matrix, curr_ids, prev_ids) = calc_ioa_matrix(frame, prev_frame, height, width);
    let (old_ids, tracked_ids) = assign(&ioa_matrix, &curr_ids, &prev_ids, config.ioa_threshold);
    let unique_id = std::cmp::max(
        prev_ids.iter().copied().max().unwrap_or(0),
        curr_ids.iter().copied().max().unwrap_or(0),
    ) + 1;

    index_assignment(&old_ids, &tracked_ids, &curr_ids, frame, unique_id, true)
}

fn calc_ioa_matrix(
    frame: &[u32],
    prev_frame: &[u32],
    height: usize,
    width: usize,
) -> (Vec<Vec<f32>>, Vec<u32>, Vec<u32>) {
    let curr_ids = collect_ids(frame);
    let prev_ids = collect_ids(prev_frame);
    let curr_idx = curr_ids
        .iter()
        .enumerate()
        .map(|(idx, label)| (*label, idx))
        .collect::<BTreeMap<_, _>>();
    let prev_idx = prev_ids
        .iter()
        .enumerate()
        .map(|(idx, label)| (*label, idx))
        .collect::<BTreeMap<_, _>>();

    let mut matrix = vec![vec![0.0; prev_ids.len()]; curr_ids.len()];
    let mut prev_areas = vec![0usize; prev_ids.len()];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let prev_label = prev_frame[idx];
            if prev_label == 0 {
                continue;
            }
            let prev_col = prev_idx[&prev_label];
            prev_areas[prev_col] += 1;

            let curr_label = frame[idx];
            if curr_label == 0 {
                continue;
            }
            let curr_row = curr_idx[&curr_label];
            matrix[curr_row][prev_col] += 1.0;
        }
    }

    for row in &mut matrix {
        for (col, value) in row.iter_mut().enumerate() {
            if prev_areas[col] != 0 {
                *value /= prev_areas[col] as f32;
            }
        }
    }

    (matrix, curr_ids, prev_ids)
}

fn assign(
    ioa_matrix: &[Vec<f32>],
    curr_ids: &[u32],
    prev_ids: &[u32],
    ioa_threshold: f32,
) -> (Vec<u32>, Vec<u32>) {
    if ioa_matrix.is_empty() || prev_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut max_col_idx = Vec::with_capacity(ioa_matrix.len());
    for row in ioa_matrix {
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        for (idx, value) in row.iter().enumerate() {
            if *value > best_val {
                best_val = *value;
                best_idx = idx;
            }
        }
        max_col_idx.push(best_idx);
    }

    let mut counts = BTreeMap::<usize, usize>::new();
    for col_idx in &max_col_idx {
        *counts.entry(*col_idx).or_default() += 1;
    }

    let mut old_ids = Vec::new();
    let mut tracked_ids = Vec::new();
    for (row_idx, col_idx) in max_col_idx.into_iter().enumerate() {
        let max_ioa = ioa_matrix[row_idx][col_idx];
        if max_ioa < ioa_threshold {
            continue;
        }

        let tracked_id = prev_ids[col_idx];
        let old_id = if counts.get(&col_idx).copied().unwrap_or(0) == 1 {
            curr_ids[row_idx]
        } else {
            let mut best_row_idx = 0usize;
            let mut best_val = f32::MIN;
            for (candidate_row_idx, candidate_row) in ioa_matrix.iter().enumerate() {
                let value = candidate_row[col_idx];
                if value > best_val {
                    best_val = value;
                    best_row_idx = candidate_row_idx;
                }
            }
            curr_ids[best_row_idx]
        };

        old_ids.push(old_id);
        tracked_ids.push(tracked_id);
    }

    (old_ids, tracked_ids)
}

fn index_assignment(
    old_ids: &[u32],
    tracked_ids: &[u32],
    curr_ids: &[u32],
    frame: &[u32],
    unique_id: u32,
    assign_unique_new_ids: bool,
) -> Vec<u32> {
    let mut tracked = frame.to_vec();
    let old_ids_set = old_ids.iter().copied().collect::<BTreeSet<_>>();
    let new_untracked_ids = curr_ids
        .iter()
        .copied()
        .filter(|id| !old_ids_set.contains(id))
        .collect::<Vec<_>>();

    if !new_untracked_ids.is_empty() && assign_unique_new_ids {
        for (offset, old_id) in new_untracked_ids.iter().enumerate() {
            replace_label(&mut tracked, *old_id, unique_id + offset as u32);
        }
    } else if !new_untracked_ids.is_empty() && !tracked_ids.is_empty() {
        let tracked_ids_set = tracked_ids.iter().copied().collect::<BTreeSet<_>>();
        let new_ids_in_tracked = new_untracked_ids
            .iter()
            .copied()
            .filter(|id| tracked_ids_set.contains(id))
            .collect::<Vec<_>>();
        for (offset, old_id) in new_ids_in_tracked.iter().enumerate() {
            replace_label(&mut tracked, *old_id, unique_id + offset as u32);
        }
    }

    for (old_id, tracked_id) in old_ids.iter().zip(tracked_ids.iter()) {
        replace_label(&mut tracked, *old_id, *tracked_id);
    }

    tracked
}

fn replace_label(frame: &mut [u32], from: u32, to: u32) {
    if from == to {
        return;
    }
    for value in frame.iter_mut() {
        if *value == from {
            *value = to;
        }
    }
}

fn collect_ids(frame: &[u32]) -> Vec<u32> {
    unique_nonzero(frame.iter().copied()).into_iter().collect()
}

fn unique_nonzero(values: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
    values.into_iter().filter(|value| *value != 0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_frame_zero_labels() {
        let frames = vec![vec![
            2, 2, 0, 0, //
            2, 2, 0, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0, //
        ]];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(tracked.frames[0], frames[0]);
        assert_eq!(tracked.labels_found, 2);
    }

    #[test]
    fn tracks_objects_by_overlap_without_distance_fallback() {
        let frames = vec![
            vec![
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![
                0, 2, 2, 0, //
                0, 2, 2, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
        ];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(
            tracked.frames[1],
            vec![
                0, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ]
        );
        assert!(tracked.disappeared.is_empty());
    }

    #[test]
    fn resolves_id_collisions_with_two_pass_assignment() {
        let frames = vec![
            vec![
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![
                2, 2, 1, 1, //
                2, 2, 1, 1, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
        ];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(
            tracked.frames[1],
            vec![
                1, 1, 3, 3, //
                1, 1, 3, 3, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ]
        );
        assert_eq!(tracked.labels_found, 3);
    }

    #[test]
    fn relabels_all_new_objects_after_empty_previous_frame() {
        let frames = vec![
            vec![
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![
                1, 1, 2, 2, //
                1, 1, 2, 2, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
        ];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(
            tracked.frames[1],
            vec![
                3, 3, 4, 4, //
                3, 3, 4, 4, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ]
        );
    }

    #[test]
    fn reports_disappearance_when_object_is_lost() {
        let frames = vec![
            vec![
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 2, 2, //
                0, 0, 2, 2, //
            ],
            vec![
                3, 3, 0, 0, //
                3, 3, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
        ];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(tracked.disappeared, vec![(0, 2)]);
        assert_eq!(tracked.frames[1][0], 1);
    }

    #[test]
    fn leaves_empty_frames_unchanged() {
        let frames = vec![
            vec![
                1, 1, 0, 0, //
                1, 1, 0, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![0; 16],
        ];

        let tracked = track_sequence(&frames, 4, 4, &TrackingConfig { ioa_threshold: 0.4 });

        assert_eq!(tracked.frames[1], vec![0; 16]);
        assert_eq!(tracked.disappeared, vec![(0, 1)]);
    }
}
