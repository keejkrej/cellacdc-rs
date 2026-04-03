use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingConfig {
    pub max_distance_px: f32,
    pub min_overlap_px: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedSequence {
    pub frames: Vec<Vec<u32>>,
    pub labels_found: u32,
    pub disappeared: Vec<(usize, u32)>,
}

#[derive(Debug, Clone, Default)]
struct RegionStats {
    area: usize,
    sum_x: f64,
    sum_y: f64,
}

#[derive(Debug, Clone)]
struct Region {
    label: u32,
    area: usize,
    centroid_x: f64,
    centroid_y: f64,
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

    let mut next_id = 1u32;
    let first_frame = relabel_with_fresh_ids(&frames[0], &mut next_id);
    let mut labels_found = first_frame.iter().copied().max().unwrap_or(0);
    tracked_frames.push(first_frame.clone());

    let mut prev_frame = first_frame;
    let mut prev_regions = compute_regions(&prev_frame, height, width);

    for (frame_i, frame) in frames.iter().enumerate().skip(1) {
        let overlaps = compute_overlaps(frame, &prev_frame);
        let curr_regions = compute_regions(frame, height, width);
        let (tracked_frame, frame_disappeared) = track_frame(
            frame,
            &curr_regions,
            &prev_regions,
            &overlaps,
            &mut next_id,
            config,
        );
        labels_found = labels_found.max(tracked_frame.iter().copied().max().unwrap_or(0));
        disappeared.extend(
            frame_disappeared
                .into_iter()
                .map(|cell_id| (frame_i - 1, cell_id)),
        );
        prev_regions = compute_regions(&tracked_frame, height, width);
        prev_frame = tracked_frame.clone();
        tracked_frames.push(tracked_frame);
    }

    TrackedSequence {
        frames: tracked_frames,
        labels_found,
        disappeared,
    }
}

fn relabel_with_fresh_ids(frame: &[u32], next_id: &mut u32) -> Vec<u32> {
    let mut remap = BTreeMap::new();
    let mut tracked = Vec::with_capacity(frame.len());

    for &label in frame {
        if label == 0 {
            tracked.push(0);
            continue;
        }
        let entry = remap.entry(label).or_insert_with(|| {
            let assigned = *next_id;
            *next_id += 1;
            assigned
        });
        tracked.push(*entry);
    }

    tracked
}

fn compute_regions(frame: &[u32], height: usize, width: usize) -> Vec<Region> {
    let mut regions = BTreeMap::<u32, RegionStats>::new();

    for y in 0..height {
        for x in 0..width {
            let label = frame[y * width + x];
            if label == 0 {
                continue;
            }
            let stats = regions.entry(label).or_default();
            stats.area += 1;
            stats.sum_x += x as f64;
            stats.sum_y += y as f64;
        }
    }

    regions
        .into_iter()
        .map(|(label, stats)| Region {
            label,
            area: stats.area,
            centroid_x: stats.sum_x / stats.area as f64,
            centroid_y: stats.sum_y / stats.area as f64,
        })
        .collect()
}

fn compute_overlaps(curr_frame: &[u32], prev_frame: &[u32]) -> HashMap<u32, BTreeMap<u32, usize>> {
    let mut overlaps = HashMap::<u32, BTreeMap<u32, usize>>::new();

    for (&curr, &prev) in curr_frame.iter().zip(prev_frame.iter()) {
        if curr == 0 || prev == 0 {
            continue;
        }
        *overlaps.entry(curr).or_default().entry(prev).or_default() += 1;
    }

    overlaps
}

fn track_frame(
    curr_frame: &[u32],
    curr_regions: &[Region],
    prev_regions: &[Region],
    overlaps: &HashMap<u32, BTreeMap<u32, usize>>,
    next_id: &mut u32,
    config: &TrackingConfig,
) -> (Vec<u32>, Vec<u32>) {
    let mut assignments = HashMap::<u32, u32>::new();
    let mut used_prev_ids = BTreeSet::<u32>::new();

    let mut overlap_candidates = curr_regions
        .iter()
        .filter_map(|region| {
            let best = overlaps
                .get(&region.label)?
                .iter()
                .filter(|(_, overlap)| **overlap >= config.min_overlap_px)
                .max_by_key(|(_, overlap)| **overlap)?;
            Some((region.label, *best.0, *best.1, region.area))
        })
        .collect::<Vec<_>>();

    overlap_candidates.sort_by(|left, right| {
        right
            .2
            .cmp(&left.2)
            .then_with(|| right.3.cmp(&left.3))
            .then_with(|| left.0.cmp(&right.0))
    });

    for (curr_label, prev_label, _, _) in overlap_candidates {
        if used_prev_ids.insert(prev_label) {
            assignments.insert(curr_label, prev_label);
        }
    }

    let unmatched_curr = curr_regions
        .iter()
        .filter(|region| !assignments.contains_key(&region.label))
        .collect::<Vec<_>>();
    let unmatched_prev = prev_regions
        .iter()
        .filter(|region| !used_prev_ids.contains(&region.label))
        .collect::<Vec<_>>();

    let mut distance_candidates = Vec::new();
    for curr in &unmatched_curr {
        for prev in &unmatched_prev {
            let dx = curr.centroid_x - prev.centroid_x;
            let dy = curr.centroid_y - prev.centroid_y;
            let distance = (dx * dx + dy * dy).sqrt() as f32;
            if distance <= config.max_distance_px {
                distance_candidates.push((curr.label, prev.label, distance));
            }
        }
    }

    distance_candidates.sort_by(|left, right| {
        left.2
            .partial_cmp(&right.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });

    for (curr_label, prev_label, _) in distance_candidates {
        if assignments.contains_key(&curr_label) || used_prev_ids.contains(&prev_label) {
            continue;
        }
        assignments.insert(curr_label, prev_label);
        used_prev_ids.insert(prev_label);
    }

    let mut remap = BTreeMap::<u32, u32>::new();
    for region in curr_regions {
        let stable_id = assignments.get(&region.label).copied().unwrap_or_else(|| {
            let assigned = *next_id;
            *next_id += 1;
            assigned
        });
        remap.insert(region.label, stable_id);
    }

    let tracked_frame = curr_frame
        .iter()
        .map(|label| {
            if *label == 0 {
                0
            } else {
                *remap.get(label).expect("missing tracked label remap")
            }
        })
        .collect::<Vec<_>>();

    let disappeared = prev_regions
        .iter()
        .filter(|region| !used_prev_ids.contains(&region.label))
        .map(|region| region.label)
        .collect::<Vec<_>>();

    (tracked_frame, disappeared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_objects_by_overlap_then_distance() {
        let frames = vec![
            vec![
                0, 1, 1, 0, //
                0, 1, 1, 0, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![
                0, 0, 2, 2, //
                0, 0, 2, 2, //
                0, 0, 0, 0, //
                0, 0, 0, 0, //
            ],
            vec![
                0, 0, 0, 0, //
                0, 0, 0, 0, //
                0, 3, 3, 0, //
                0, 3, 3, 0, //
            ],
        ];

        let tracked = track_sequence(
            &frames,
            4,
            4,
            &TrackingConfig {
                max_distance_px: 3.0,
                min_overlap_px: 1,
            },
        );

        assert_eq!(tracked.labels_found, 1);
        assert!(tracked.frames.iter().all(|frame| frame.contains(&1)));
        assert!(tracked.disappeared.is_empty());
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

        let tracked = track_sequence(
            &frames,
            4,
            4,
            &TrackingConfig {
                max_distance_px: 2.0,
                min_overlap_px: 1,
            },
        );

        assert_eq!(tracked.disappeared, vec![(0, 2)]);
        assert_eq!(tracked.frames[0][0], tracked.frames[1][0]);
    }
}
