//! Progressive ADSampling-based top-1 nearest-neighbour search.
//!
//! Given a batch of partial (GEMM) distances over the first `partial_d`
//! dimensions, this routine adds the remaining dimensions in two phases —
//! horizontal-d in blocks of `H_DIM_SIZE`, then the trailing vertical-d slab
//! — pruning aggressively against the running best with the precomputed
//! pruner ratios.

use crate::adsampling::ADSamplingPruner;
use crate::common::{H_DIM_SIZE, KnnCandidate};

/// Top-1 search for a single query against `batch_n_y` centroids stored
/// row-major in `centroids[(j..j+batch_n_y) * d]` (slice passed at the right
/// offset). On entry `partial_distances` holds the raw GEMM dot products over
/// the first `partial_d` dimensions; this converts them to partial L2 distances
/// in place (using `norm_x_i` and the per-centroid partial norms `norms_y`) and
/// refines the surviving entries.
///
/// Returns `(best_candidate, initial_not_pruned)`.
#[allow(clippy::too_many_arguments)]
pub fn top1_partial_search(
    pruner: &ADSamplingPruner,
    query: &[f32],     // length d (already rotated)
    centroids: &[f32], // batch_n_y × d, row-major
    batch_n_y: usize,
    d: usize,
    vertical_d: usize,
    horizontal_d: usize,
    partial_distances: &mut [f32], // length batch_n_y; raw dots in, partial L2 out
    norm_x_i: f32,                 // query's partial squared norm
    norms_y: &[f32],               // centroids' partial squared norms, length batch_n_y
    partial_d: usize,
    prev_top_1: u32,
    prev_threshold: f32,
    centroid_global_offset: u32,
    positions_buf: &mut Vec<u32>,
) -> (KnnCandidate, usize) {
    let ratios = &pruner.ratios;
    let mut top = KnnCandidate::new(prev_top_1, prev_threshold);

    positions_buf.clear();
    let init_threshold = top.distance * ratios[partial_d];
    // Fused: convert raw dot products to partial L2 (||x||²+||c||²-2·x·c) and
    // collect the surviving candidates in a single pass over the row.
    for i in 0..batch_n_y {
        let v = -2.0 * partial_distances[i] + norm_x_i + norms_y[i];
        partial_distances[i] = v;
        if v < init_threshold {
            positions_buf.push(i as u32);
        }
    }
    let initial_not_pruned = positions_buf.len();

    // Early exit: only candidate is the previous best.
    if positions_buf.len() == 1 {
        let p = positions_buf[0];
        if centroid_global_offset + p == prev_top_1 {
            positions_buf.clear();
            return (top, initial_not_pruned);
        }
    }

    let mut current_dim = partial_d;
    let mut current_horizontal = 0;

    // Horizontal phase — H_DIM_SIZE columns at a time, starting at offset vertical_d.
    while !positions_buf.is_empty() && current_horizontal < horizontal_d {
        let block_size = H_DIM_SIZE.min(horizontal_d - current_horizontal);
        let offset = vertical_d + current_horizontal;
        let q_slice = &query[offset..offset + block_size];

        for &v_idx in positions_buf.iter() {
            let i = v_idx as usize;
            let row = &centroids[i * d + offset..i * d + offset + block_size];
            let mut acc = 0.0_f32;
            for k in 0..block_size {
                let diff = q_slice[k] - row[k];
                acc += diff * diff;
            }
            partial_distances[i] += acc;
        }

        current_horizontal += block_size;
        current_dim += block_size;
        let threshold = top.distance * ratios[current_dim];
        positions_buf.retain(|&v_idx| partial_distances[v_idx as usize] < threshold);
    }

    // Trailing vertical-d slab — columns [partial_d, vertical_d).
    if !positions_buf.is_empty() && partial_d < vertical_d {
        let dims_left = vertical_d - partial_d;
        let offset = partial_d;
        let q_slice = &query[offset..offset + dims_left];

        for &v_idx in positions_buf.iter() {
            let i = v_idx as usize;
            let row = &centroids[i * d + offset..i * d + offset + dims_left];
            let mut acc = 0.0_f32;
            for k in 0..dims_left {
                let diff = q_slice[k] - row[k];
                acc += diff * diff;
            }
            partial_distances[i] += acc;
        }

        current_dim = d;
        let threshold = top.distance * ratios[current_dim];
        positions_buf.retain(|&v_idx| partial_distances[v_idx as usize] < threshold);
    }

    // Final scan over remaining candidates.
    for &v_idx in positions_buf.iter() {
        let i = v_idx as usize;
        let dist = partial_distances[i];
        if dist < top.distance {
            top.distance = dist;
            top.index = centroid_global_offset + v_idx;
        }
    }

    positions_buf.clear();
    (top, initial_not_pruned)
}
