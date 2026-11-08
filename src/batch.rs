//! Batched nearest-neighbour search built on top of SGEMM and the pruning kernel.

use rayon::prelude::*;

use crate::adsampling::ADSamplingPruner;
use crate::common::{MINI_BATCH_SIZE, X_BATCH_SIZE, Y_BATCH_SIZE};
use crate::distance::l2_squared;
use crate::pdxearch::top1_partial_search;

/// Compute `out[i * ldc + j] = dot(x[i, :effective_d], y[j, :effective_d])`; `ldc` is the
/// row stride of `out`, at least `batch_n_y`.
fn blas_dot_products(
    x_batch: &[f32], // batch_n_x × d row-major
    y_batch: &[f32], // batch_n_y × d row-major
    batch_n_x: usize,
    batch_n_y: usize,
    d: usize,
    effective_d: usize,
    out: &mut [f32], // batch_n_x rows of batch_n_y values, ldc apart
    ldc: usize,
) {
    debug_assert!(x_batch.len() >= batch_n_x * d);
    debug_assert!(y_batch.len() >= batch_n_y * d);
    debug_assert!(ldc >= batch_n_y);
    debug_assert!(out.len() >= (batch_n_x - 1) * ldc + batch_n_y);
    // out = x[:, :effective_d] * y[:, :effective_d]^T ; both rows are d-wide.
    let gemm = |rows: usize, x: &[f32], c: &mut [f32]| {
        crate::gemm::sgemm_ld(
            false,
            true,
            rows,
            effective_d,
            batch_n_y,
            x,
            d,
            y_batch,
            d,
            c,
            ldc,
        );
    };

    // Dispatch mini-batches across Rayon threads and keep each `sgemm_ld` call
    // single-threaded so BLAS and Rayon never run competing thread pools.
    out.par_chunks_mut(MINI_BATCH_SIZE * ldc)
        .enumerate()
        .for_each(|(mb, chunk)| {
            let r = mb * MINI_BATCH_SIZE;
            let rows = MINI_BATCH_SIZE.min(batch_n_x - r);
            gemm(rows, &x_batch[r * d..(r + rows) * d], chunk);
        });
}

/// Maximum bytes per distance/product matrix in a SOAR x-tile.
const SOAR_TILE_BYTES: usize = 64 << 20;

/// Rows per x-tile of the SOAR assignment: the x batch, capped so the kept distances stay
/// within [`SOAR_TILE_BYTES`].
fn soar_tile_rows(n_x: usize, n_y: usize) -> usize {
    (SOAR_TILE_BYTES / (4 * n_y.max(1)))
        .clamp(1, X_BATCH_SIZE)
        .min(n_x.max(1))
}

/// Primary and SOAR secondary centroid per vector, from two SGEMM passes.
///
/// Pass 1 computes every distance `D[r][c] = ‖x‖² + ‖c‖² − 2⟨x_r, c⟩` of an x-tile and keeps
/// it; the primary `c1` is its argmin, chosen exactly as [`find_nearest_neighbor`] chooses it.
/// With `r̂1 = (x − C[c1]) / ‖x − C[c1]‖` (zero when the vector sits on its centroid, which
/// reduces the choice to the second nearest) and `a = ⟨x, r̂1⟩`, pass 2 computes
/// `P[r][c] = ⟨r̂1, c⟩` and the secondary minimizes `D[r][c] + lambda · (a − P[r][c])²` over
/// `c ≠ c1`. With a single centroid the secondary is `u32::MAX`. `buf` is caller-owned scratch,
/// grown as needed and reused across calls.
pub fn find_soar_assignments(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    lambda: f32,
    out_primary: &mut [u32],
    out_secondary: &mut [u32],
    buf: &mut Vec<f32>,
) {
    debug_assert!(x.len() >= n_x * d);
    debug_assert!(y.len() >= n_y * d);
    debug_assert!(norms_x.len() >= n_x && norms_y.len() >= n_y);
    for v in out_primary.iter_mut().take(n_x) {
        *v = 0;
    }
    for v in out_secondary.iter_mut().take(n_x) {
        *v = u32::MAX;
    }
    if n_x == 0 || n_y == 0 {
        return;
    }

    let tile = soar_tile_rows(n_x, n_y);
    let y_tile = Y_BATCH_SIZE.min(n_y);
    // Centroid-major tiles expose parallel work even when many centroids leave few x rows.
    // The two matrices each stay within SOAR_TILE_BYTES (except a single oversized row).
    let matrix_len = tile * n_y;
    let res_len = tile * d;
    let required = 2 * matrix_len + res_len + tile;
    if buf.len() < required {
        buf.resize(required, 0.0);
    }
    let (dist, rest) = buf.split_at_mut(matrix_len);
    let (prod, rest) = rest.split_at_mut(matrix_len);
    let (residuals, rest) = rest.split_at_mut(res_len);
    let (along, _) = rest.split_at_mut(tile);

    let mut i = 0;
    while i < n_x {
        let t = tile.min(n_x - i);
        let x_tile = &x[i * d..(i + t) * d];
        let primary = &mut out_primary[i..i + t];
        let secondary = &mut out_secondary[i..i + t];
        let block_len = t * y_tile;

        dist[..t * n_y]
            .par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(block, distances)| {
                let j = block * y_tile;
                let cols = distances.len() / t;
                blas_dot_products(
                    x_tile,
                    &y[j * d..(j + cols) * d],
                    t,
                    cols,
                    d,
                    d,
                    distances,
                    cols,
                );
                for (r, row) in distances.chunks_mut(cols).enumerate() {
                    for (c, value) in row.iter_mut().enumerate() {
                        *value = -2.0 * *value + norms_x[i + r] + norms_y[j + c];
                    }
                }
            });
        primary.par_iter_mut().enumerate().for_each(|(r, idx)| {
            let mut best = f32::MAX;
            for (block, distances) in dist[..t * n_y].chunks(block_len).enumerate() {
                let cols = distances.len() / t;
                let row = &distances[r * cols..(r + 1) * cols];
                let mut min_v = f32::MAX;
                let mut min_c = 0;
                for (c, &value) in row.iter().enumerate() {
                    if value < min_v {
                        min_v = value;
                        min_c = c;
                    }
                }
                // Preserve nearest-neighbor's ordered tile reduction, including its clamp.
                if min_v < best {
                    best = min_v.max(0.0);
                    *idx = (block * y_tile + min_c) as u32;
                }
            }
        });
        if n_y == 1 {
            i += t;
            continue;
        }

        // Unit residuals to the primary, and each vector's projection on its own.
        let chosen: &[u32] = primary;
        residuals[..t * d]
            .par_chunks_mut(d)
            .zip(along[..t].par_iter_mut())
            .enumerate()
            .for_each(|(r, (residual, a))| {
                let xr = &x_tile[r * d..(r + 1) * d];
                let centroid = &y[chosen[r] as usize * d..][..d];
                let mut norm = 0.0_f32;
                for k in 0..d {
                    let v = xr[k] - centroid[k];
                    residual[k] = v;
                    norm += v * v;
                }
                let norm = norm.sqrt();
                if norm > 0.0 && norm.is_finite() {
                    let inv = 1.0 / norm;
                    for v in residual.iter_mut() {
                        *v *= inv;
                    }
                } else {
                    residual.fill(0.0);
                }
                *a = xr.iter().zip(residual.iter()).map(|(p, q)| p * q).sum();
            });

        prod[..t * n_y]
            .par_chunks_mut(block_len)
            .zip(dist[..t * n_y].par_chunks(block_len))
            .enumerate()
            .for_each(|(block, (costs, distances))| {
                let j = block * y_tile;
                let cols = costs.len() / t;
                blas_dot_products(
                    &residuals[..t * d],
                    &y[j * d..(j + cols) * d],
                    t,
                    cols,
                    d,
                    d,
                    costs,
                    cols,
                );
                for (r, (row, distances)) in costs
                    .chunks_mut(cols)
                    .zip(distances.chunks(cols))
                    .enumerate()
                {
                    for (c, (value, distance)) in row.iter_mut().zip(distances).enumerate() {
                        let gap = along[r] - *value;
                        *value = if (j + c) as u32 == chosen[r] {
                            f32::MAX
                        } else {
                            distance + lambda * gap * gap
                        };
                    }
                }
            });
        secondary.par_iter_mut().enumerate().for_each(|(r, idx)| {
            let mut best = f32::MAX;
            for (block, costs) in prod[..t * n_y].chunks(block_len).enumerate() {
                let cols = costs.len() / t;
                for (c, &value) in costs[r * cols..(r + 1) * cols].iter().enumerate() {
                    if value < best {
                        best = value;
                        *idx = (block * y_tile + c) as u32;
                    }
                }
            }
        });
        i += t;
    }
}

/// Grow `buf` so one SGEMM batch (up to `X_BATCH_SIZE × Y_BATCH_SIZE`, clamped
/// to the problem size) fits. The buffer is caller-owned and reused across
/// calls, so this avoids re-allocating (and page-faulting) it every iteration.
/// The GEMM writes the used region with beta=0, so stale contents are fine.
fn ensure_buf(buf: &mut Vec<f32>, n_x: usize, n_y: usize) {
    let required = X_BATCH_SIZE.min(n_x) * Y_BATCH_SIZE.min(n_y);
    if buf.len() < required {
        buf.resize(required, 0.0);
    }
}

/// Brute-force top-1 nearest-centroid search via batched SGEMM.
pub fn find_nearest_neighbor(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    out_knn: &mut [u32],
    out_distances: &mut [f32],
    buf: &mut Vec<f32>,
) {
    for v in out_distances.iter_mut().take(n_x) {
        *v = f32::MAX;
    }
    for v in out_knn.iter_mut().take(n_x) {
        *v = 0;
    }

    ensure_buf(buf, n_x, n_y);

    let mut i = 0;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x_p = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y_p = &y[j * d..(j + batch_n_y) * d];

            blas_dot_products(
                batch_x_p,
                batch_y_p,
                batch_n_x,
                batch_n_y,
                d,
                d,
                &mut buf[..batch_n_x * batch_n_y],
                batch_n_y,
            );

            let dist_window = &mut out_distances[i..i + batch_n_x];
            let knn_window = &mut out_knn[i..i + batch_n_x];
            let nx = norms_x;
            let ny = norms_y;
            let j_off = j;

            buf[..batch_n_x * batch_n_y]
                .par_chunks_mut(batch_n_y)
                .zip(dist_window.par_iter_mut())
                .zip(knn_window.par_iter_mut())
                .enumerate()
                .for_each(|(r, ((row, best_d), best_idx))| {
                    let norm_x_i = nx[i + r];
                    let mut min_v = f32::MAX;
                    let mut min_c: u32 = 0;
                    for c in 0..batch_n_y {
                        let v = -2.0 * row[c] + norm_x_i + ny[j_off + c];
                        row[c] = v;
                        if v < min_v {
                            min_v = v;
                            min_c = c as u32;
                        }
                    }
                    if min_v < *best_d {
                        *best_d = min_v.max(0.0);
                        *best_idx = j_off as u32 + min_c;
                    }
                });

            j += batch_n_y;
        }
        i += batch_n_x;
    }
}

/// k-nearest neighbours brute-force via batched SGEMM.
pub fn find_k_nearest_neighbors(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    k: usize,
    out_knn: &mut [u32],       // n_x × k
    out_distances: &mut [f32], // n_x × k
) {
    for v in out_distances.iter_mut() {
        *v = f32::MAX;
    }
    for v in out_knn.iter_mut() {
        *v = u32::MAX;
    }

    let mut buf = vec![0.0_f32; X_BATCH_SIZE * Y_BATCH_SIZE];

    let mut i = 0;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x_p = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y_p = &y[j * d..(j + batch_n_y) * d];

            blas_dot_products(
                batch_x_p,
                batch_y_p,
                batch_n_x,
                batch_n_y,
                d,
                d,
                &mut buf[..batch_n_x * batch_n_y],
                batch_n_y,
            );

            // Per row: convert dot to L2 then merge into the current top-k for that query.
            // We split out_distances/out_knn into row-sized chunks for safe parallelism.
            let i_base = i;
            let j_base = j;
            let bn_y = batch_n_y;
            let nx = norms_x;
            let ny = norms_y;

            let buf_slice = &mut buf[..batch_n_x * bn_y];
            let dist_window = &mut out_distances[i_base * k..(i_base + batch_n_x) * k];
            let knn_window = &mut out_knn[i_base * k..(i_base + batch_n_x) * k];

            buf_slice
                .par_chunks_mut(bn_y)
                .zip(dist_window.par_chunks_mut(k))
                .zip(knn_window.par_chunks_mut(k))
                .enumerate()
                .for_each(|(r, ((row, dist_row), knn_row))| {
                    let norm_x_i = nx[i_base + r];
                    for c in 0..bn_y {
                        row[c] = -2.0 * row[c] + norm_x_i + ny[j_base + c];
                    }

                    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(k + bn_y);
                    for ki in 0..k {
                        if dist_row[ki] < f32::MAX {
                            candidates.push((dist_row[ki], knn_row[ki]));
                        }
                    }
                    for c in 0..bn_y {
                        candidates.push((row[c], (j_base + c) as u32));
                    }
                    let actual_k = k.min(candidates.len());
                    candidates
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    for ki in 0..actual_k {
                        dist_row[ki] = candidates[ki].0.max(0.0);
                        knn_row[ki] = candidates[ki].1;
                    }
                    for ki in actual_k..k {
                        dist_row[ki] = f32::MAX;
                        knn_row[ki] = u32::MAX;
                    }
                });

            j += batch_n_y;
        }
        i += batch_n_x;
    }
}

/// GEMM + ADSampling/PDX pruning. `norms_x_partial` and `norms_y_partial` are
/// squared L2 norms restricted to the first `partial_d` dimensions.
/// `out_knn` / `out_distances` come in seeded from the previous iteration and
/// are refined in place.
pub fn find_nearest_neighbor_with_pruning(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    vertical_d: usize,
    horizontal_d: usize,
    norms_x_partial: &[f32],
    norms_y_partial: &[f32],
    out_knn: &mut [u32],
    out_distances: &mut [f32],
    pruner: &ADSamplingPruner,
    partial_d: usize,
    out_not_pruned_counts: &mut [usize],
    buf: &mut Vec<f32>,
) {
    ensure_buf(buf, n_x, n_y);

    let mut i = 0;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x_p = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y_p = &y[j * d..(j + batch_n_y) * d];

            blas_dot_products(
                batch_x_p,
                batch_y_p,
                batch_n_x,
                batch_n_y,
                d,
                partial_d,
                &mut buf[..batch_n_x * batch_n_y],
                batch_n_y,
            );

            let i_base = i;
            let j_base = j;
            let bn_y = batch_n_y;
            let nx = norms_x_partial;
            let ny = norms_y_partial;

            let buf_slice = &mut buf[..batch_n_x * bn_y];
            let knn_window = &mut out_knn[i_base..i_base + batch_n_x];
            let dist_window = &mut out_distances[i_base..i_base + batch_n_x];
            let npc_window = &mut out_not_pruned_counts[i_base..i_base + batch_n_x];

            buf_slice
                .par_chunks_mut(bn_y)
                .zip(knn_window.par_iter_mut())
                .zip(dist_window.par_iter_mut())
                .zip(npc_window.par_iter_mut())
                .enumerate()
                // Reuse one `positions` scratch buffer per worker thread instead
                // of allocating a Vec per row; top1_partial_search clears it.
                .for_each_init(
                    || Vec::<u32>::with_capacity(bn_y),
                    |positions, (r, (((row, knn), dist), npc))| {
                        let i_idx = i_base + r;
                        let norm_x_i = nx[i_idx];
                        // Convert raw dot products to partial L2 distances. Iterate
                        // over zipped slices (no bounds checks, no aliasing) so the
                        // autovectorizer emits a SIMD FMA loop — the Rust equivalent
                        // of the C++ `SKM_VECTORIZE_LOOP` conversion pass.
                        for (rv, &ny_c) in row.iter_mut().zip(&ny[j_base..j_base + bn_y]) {
                            *rv = norm_x_i + ny_c - 2.0 * *rv;
                        }
                        let data_p = &x[i_idx * d..i_idx * d + d];
                        let prev_assignment = *knn;
                        let dist_to_prev = if j_base == 0 {
                            let other_start = prev_assignment as usize * d;
                            l2_squared(&y[other_start..other_start + d], data_p)
                        } else {
                            *dist
                        };

                        let (assignment, local_not_pruned) = top1_partial_search(
                            pruner,
                            data_p,
                            batch_y_p,
                            bn_y,
                            d,
                            vertical_d,
                            horizontal_d,
                            row,
                            partial_d,
                            prev_assignment,
                            dist_to_prev,
                            j_base as u32,
                            positions,
                        );
                        *npc += local_not_pruned;
                        *knn = assignment.index;
                        *dist = assignment.distance.max(0.0);
                    },
                );

            j += batch_n_y;
        }
        i += batch_n_x;
    }
}
