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

/// Primary and SOAR secondary centroid per vector: [`find_soar_replica_assignments`] with
/// `replicas = 2`. With a single centroid the secondary is `u32::MAX`.
pub fn find_soar_assignments(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    lambda: f32,
    candidates: usize,
    out_primary: &mut [u32],
    out_secondary: &mut [u32],
    buf: &mut Vec<f32>,
    ids: &mut Vec<u32>,
) {
    find_soar_replica_assignments(
        x,
        y,
        n_x,
        n_y,
        d,
        norms_x,
        norms_y,
        lambda,
        2,
        candidates,
        out_primary,
        out_secondary,
        buf,
        ids,
    );
}

/// Primary and `replicas − 1` SOAR secondaries per vector.
///
/// The primary is the nearest centroid, chosen exactly as [`find_nearest_neighbor`] chooses it.
/// Each secondary is chosen among the row's `candidates` nearest centroids by
/// `‖x − c‖² + lambda · Σ_i ⟨x − c, r̂_i⟩²` over the unit residuals `r̂_i` to every preceding
/// assignment, skipping centroids already chosen; a zero residual contributes no penalty and
/// ties go to the lowest centroid index. `candidates` is clamped to `1..=n_y`, and
/// `candidates = n_y` is the exhaustive rule. Secondaries are row-major, `replicas − 1` per
/// vector, `u32::MAX` where none is available.
///
/// One SGEMM pass over every centroid gives the distances, and each row keeps its `candidates`
/// nearest. Every secondary then costs `candidates · d` multiply-adds per row over the row's
/// gathered shortlist, with no further pass over the centroid table. `buf` and `ids` are
/// caller-owned scratch, grown as needed and reused across calls: one x-batch by y-batch product
/// block, the same as [`find_nearest_neighbor`]'s, plus the shortlist of every row in the x-batch.
pub fn find_soar_replica_assignments(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    lambda: f32,
    replicas: usize,
    candidates: usize,
    out_primary: &mut [u32],
    out_secondary: &mut [u32],
    buf: &mut Vec<f32>,
    ids: &mut Vec<u32>,
) {
    assert!(replicas > 0);
    let secondaries = replicas - 1;
    assert!(out_primary.len() >= n_x);
    assert!(out_secondary.len() >= n_x * secondaries);
    debug_assert!(x.len() >= n_x * d);
    debug_assert!(y.len() >= n_y * d);
    debug_assert!(norms_x.len() >= n_x && norms_y.len() >= n_y);
    for v in out_primary.iter_mut().take(n_x) {
        *v = 0;
    }
    for v in out_secondary.iter_mut().take(n_x * secondaries) {
        *v = u32::MAX;
    }
    if n_x == 0 || n_y == 0 {
        return;
    }

    let m = candidates.clamp(1, n_y);
    let usable = secondaries.min(m - 1);
    let tile = X_BATCH_SIZE.min(n_x);
    let block_len = tile * Y_BATCH_SIZE.min(n_y);
    let required = block_len + tile * m + tile;
    if buf.len() < required {
        buf.resize(required, 0.0);
    }
    if ids.len() < tile * m {
        ids.resize(tile * m, u32::MAX);
    }
    let (block, rest) = buf.split_at_mut(block_len);
    let (near_dist, rest) = rest.split_at_mut(tile * m);
    let best = &mut rest[..tile];
    let near_ids = &mut ids[..tile * m];

    let mut i = 0;
    while i < n_x {
        let t = tile.min(n_x - i);
        let x_tile = &x[i * d..(i + t) * d];
        let primary = &mut out_primary[i..i + t];
        let near_dist = &mut near_dist[..t * m];
        let near_ids = &mut near_ids[..t * m];
        let best = &mut best[..t];
        near_dist.fill(f32::MAX);
        near_ids.fill(u32::MAX);
        best.fill(f32::MAX);

        // Pass 1: distances to every centroid, one y-batch at a time, as find_nearest_neighbor.
        let mut j = 0;
        while j < n_y {
            let cols = Y_BATCH_SIZE.min(n_y - j);
            let products = &mut block[..t * cols];
            blas_dot_products(
                x_tile,
                &y[j * d..(j + cols) * d],
                t,
                cols,
                d,
                d,
                products,
                cols,
            );
            products
                .par_chunks(cols)
                .zip(near_dist.par_chunks_mut(m))
                .zip(near_ids.par_chunks_mut(m))
                .zip(best.par_iter_mut())
                .zip(primary.par_iter_mut())
                .enumerate()
                .for_each(|(r, ((((row, dist), id), best), primary))| {
                    let norm_x = norms_x[i + r];
                    let mut min_v = f32::MAX;
                    let mut min_c = 0;
                    for (c, &dot) in row.iter().enumerate() {
                        let v = -2.0 * dot + norm_x + norms_y[j + c];
                        if v < min_v {
                            min_v = v;
                            min_c = c;
                        }
                        if v < dist[0] {
                            push_nearest(dist, id, v, (j + c) as u32);
                        }
                    }
                    // Nearest-neighbour's ordered reduction over y-batches, clamp included.
                    if min_v < *best {
                        *best = min_v.max(0.0);
                        *primary = (j + min_c) as u32;
                    }
                });
            j += cols;
        }

        if usable > 0 {
            out_secondary[i * secondaries..(i + t) * secondaries]
                .par_chunks_mut(secondaries)
                .zip(near_dist.par_chunks(m))
                .zip(near_ids.par_chunks(m))
                .zip(primary.par_iter())
                .enumerate()
                .for_each_init(
                    || Shortlist::new(m, d),
                    |shortlist, (r, (((chosen, dist), id), &primary))| {
                        shortlist.choose(
                            &x_tile[r * d..(r + 1) * d],
                            y,
                            d,
                            lambda,
                            primary,
                            dist,
                            id,
                            &mut chosen[..usable],
                        );
                    },
                );
        }
        i += t;
    }
}

/// Whether `(a, ia)` orders after `(b, ib)`: by distance, then by centroid index.
#[inline]
fn after(a: f32, ia: u32, b: f32, ib: u32) -> bool {
    a > b || (a == b && ia > ib)
}

/// Replaces the root of the max-heap `dist` / `ids`, ordered by [`after`], with `(v, c)` and
/// restores the heap; the caller has checked `v < dist[0]`. The heap holds the row's nearest
/// centroids seen so far, `(f32::MAX, u32::MAX)` in slots not yet filled. Centroids arrive in
/// index order, so of equally distant centroids the lowest indices are kept.
fn push_nearest(dist: &mut [f32], ids: &mut [u32], v: f32, c: u32) {
    let m = dist.len();
    let mut slot = 0;
    loop {
        let left = 2 * slot + 1;
        if left >= m {
            break;
        }
        let right = left + 1;
        let child = if right < m && after(dist[right], ids[right], dist[left], ids[left]) {
            right
        } else {
            left
        };
        if !after(dist[child], ids[child], v, c) {
            break;
        }
        dist[slot] = dist[child];
        ids[slot] = ids[child];
        slot = child;
    }
    dist[slot] = v;
    ids[slot] = c;
}

/// Per-thread scratch for choosing one row's secondaries from its shortlist.
struct Shortlist {
    /// The shortlisted centroids, gathered once per row and read by every replica.
    rows: Vec<f32>,
    /// `(centroid, distance)` per shortlisted centroid, in centroid order.
    entries: Vec<(u32, f32)>,
    ids: Vec<u32>,
    cost: Vec<f32>,
    products: Vec<f32>,
    residual: Vec<f32>,
}

impl Shortlist {
    fn new(m: usize, d: usize) -> Self {
        Shortlist {
            rows: vec![0.0; m * d],
            entries: Vec::with_capacity(m),
            ids: vec![0; m],
            cost: vec![0.0; m],
            products: vec![0.0; m],
            residual: vec![0.0; d],
        }
    }

    /// Fills `out` with the row's secondaries in order, leaving `u32::MAX` once no shortlisted
    /// centroid is left. `dist` / `ids` are the row's shortlist heap from pass 1.
    fn choose(
        &mut self,
        x: &[f32],
        y: &[f32],
        d: usize,
        lambda: f32,
        primary: u32,
        dist: &[f32],
        ids: &[u32],
        out: &mut [u32],
    ) {
        // Gathered in centroid order, so the reads ascend through the centroid table.
        self.entries.clear();
        self.entries.extend(
            ids.iter()
                .zip(dist)
                .filter(|&(&c, _)| c != u32::MAX)
                .map(|(&c, &v)| (c, v)),
        );
        self.entries.sort_unstable_by_key(|&(c, _)| c);
        let len = self.entries.len();
        if len == 0 {
            return;
        }
        for (slot, &(c, v)) in self.entries.iter().enumerate() {
            self.rows[slot * d..(slot + 1) * d].copy_from_slice(&y[c as usize * d..][..d]);
            self.ids[slot] = c;
            self.cost[slot] = v;
        }
        let ids = &self.ids[..len];
        let cost = &mut self.cost[..len];
        let products = &mut self.products[..len];
        let mut previous = primary;
        for k in 0..out.len() {
            let along = unit_residual(x, &y[previous as usize * d..][..d], &mut self.residual);
            crate::gemm::sgemm_ld(
                false,
                true,
                1,
                d,
                len,
                &self.residual,
                d,
                &self.rows[..len * d],
                d,
                products,
                len,
            );
            let mut pick = (f32::MAX, u32::MAX);
            for ((value, &product), &c) in cost.iter_mut().zip(products.iter()).zip(ids) {
                let gap = along - product;
                *value += lambda * gap * gap;
                let taken = c == primary || out[..k].contains(&c);
                if !taken
                    && *value < f32::MAX
                    && (*value < pick.0 || (*value == pick.0 && c < pick.1))
                {
                    pick = (*value, c);
                }
            }
            if pick.1 == u32::MAX {
                break;
            }
            out[k] = pick.1;
            previous = pick.1;
        }
    }
}

/// Writes the unit residual `(x − c) / ‖x − c‖` into `residual`, all zeros when the norm is zero
/// or not finite, and returns `⟨x, residual⟩`.
fn unit_residual(x: &[f32], c: &[f32], residual: &mut [f32]) -> f32 {
    let mut norm = 0.0_f32;
    for k in 0..x.len() {
        let v = x[k] - c[k];
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
    x.iter().zip(residual.iter()).map(|(p, q)| p * q).sum()
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

#[cfg(test)]
pub(crate) mod reference {
    use super::*;

    /// Maximum bytes per distance/product matrix in an x-tile of the reference.
    const SOAR_TILE_BYTES: usize = 256 << 20;

    /// Rows per x-tile of the reference.
    fn soar_tile_rows(n_x: usize, n_y: usize) -> usize {
        (SOAR_TILE_BYTES / (4 * n_y.max(1)))
            .clamp(1, X_BATCH_SIZE)
            .min(n_x.max(1))
    }

    /// The exhaustive rule as one SGEMM over every centroid per replica: the reference that
    /// `candidates = n_y` must reproduce.
    pub(super) fn exact_soar_replica_assignments(
        x: &[f32],
        y: &[f32],
        n_x: usize,
        n_y: usize,
        d: usize,
        norms_x: &[f32],
        norms_y: &[f32],
        lambda: f32,
        replicas: usize,
        out_primary: &mut [u32],
        out_secondary: &mut [u32],
        buf: &mut Vec<f32>,
    ) {
        assert!(replicas > 0);
        let secondaries = replicas - 1;
        assert!(out_primary.len() >= n_x);
        assert!(out_secondary.len() >= n_x * secondaries);
        debug_assert!(x.len() >= n_x * d);
        debug_assert!(y.len() >= n_y * d);
        debug_assert!(norms_x.len() >= n_x && norms_y.len() >= n_y);
        for v in out_primary.iter_mut().take(n_x) {
            *v = 0;
        }
        for v in out_secondary.iter_mut().take(n_x * secondaries) {
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
            let secondary = &mut out_secondary[i * secondaries..(i + t) * secondaries];
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
            if n_y == 1 || secondaries == 0 {
                i += t;
                continue;
            }

            for replica in 0..secondaries.min(n_y - 1) {
                residuals[..t * d]
                    .par_chunks_mut(d)
                    .zip(along[..t].par_iter_mut())
                    .enumerate()
                    .for_each(|(r, (residual, a))| {
                        let xr = &x_tile[r * d..(r + 1) * d];
                        let chosen = if replica == 0 {
                            primary[r]
                        } else {
                            secondary[r * secondaries + replica - 1]
                        };
                        let centroid = &y[chosen as usize * d..][..d];
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
                    .zip(dist[..t * n_y].par_chunks_mut(block_len))
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
                            .zip(distances.chunks_mut(cols))
                            .enumerate()
                        {
                            for (c, (value, distance)) in row.iter_mut().zip(distances).enumerate()
                            {
                                let gap = along[r] - *value;
                                *distance += lambda * gap * gap;
                                let centroid = (j + c) as u32;
                                *value = if centroid == primary[r]
                                    || secondary[r * secondaries..r * secondaries + replica]
                                        .contains(&centroid)
                                {
                                    f32::MAX
                                } else {
                                    *distance
                                };
                            }
                        }
                    });
                secondary
                    .par_chunks_mut(secondaries)
                    .enumerate()
                    .for_each(|(r, assignments)| {
                        let mut best = f32::MAX;
                        for (block, costs) in prod[..t * n_y].chunks(block_len).enumerate() {
                            let cols = costs.len() / t;
                            for (c, &value) in costs[r * cols..(r + 1) * cols].iter().enumerate() {
                                if value < best {
                                    best = value;
                                    assignments[replica] = (block * y_tile + c) as u32;
                                }
                            }
                        }
                    });
            }
            i += t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{generate_random_vectors, squared_norms};

    /// With `candidates = n_y` the shortlist holds every centroid, so the assignments equal the
    /// exhaustive reference's, for rows on a centroid (zero residual) and duplicate centroids
    /// (ties) included. A vendor BLAS rounds the products of identical centroids by column
    /// position, so there an exact tie may resolve to another copy of the same centroid.
    #[test]
    fn every_centroid_as_candidate_reproduces_the_exhaustive_rule() {
        let (n, d) = (600, 24);
        for k in [1, 2, 5, 300] {
            let mut centroids = generate_random_vectors(k, d, -1.0, 1.0, 17 + k as u64);
            for (copy, of) in [(2, 0), (4, 1), (7, 0), (200, 3)] {
                if copy < k {
                    centroids.copy_within(of * d..(of + 1) * d, copy * d);
                }
            }
            let mut data = generate_random_vectors(n, d, -1.0, 1.0, 29 + k as u64);
            for row in 0..20 {
                let c = row % k;
                data[row * d..(row + 1) * d].copy_from_slice(&centroids[c * d..(c + 1) * d]);
            }
            let norms_x = squared_norms(&data, n, d);
            let norms_y = squared_norms(&centroids, k, d);
            for replicas in [2, 3, 4] {
                let s = replicas - 1;
                let (mut primary, mut secondary) = (vec![0; n], vec![0; n * s]);
                let (mut want_primary, mut want_secondary) = (vec![0; n], vec![0; n * s]);
                find_soar_replica_assignments(
                    &data,
                    &centroids,
                    n,
                    k,
                    d,
                    &norms_x,
                    &norms_y,
                    2.0,
                    replicas,
                    k,
                    &mut primary,
                    &mut secondary,
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
                reference::exact_soar_replica_assignments(
                    &data,
                    &centroids,
                    n,
                    k,
                    d,
                    &norms_x,
                    &norms_y,
                    2.0,
                    replicas,
                    &mut want_primary,
                    &mut want_secondary,
                    &mut Vec::new(),
                );
                #[cfg(not(feature = "blas"))]
                {
                    assert_eq!(primary, want_primary, "k {k}, replicas {replicas}");
                    assert_eq!(secondary, want_secondary, "k {k}, replicas {replicas}");
                }
                let centroid = |c: u32| (c != u32::MAX).then(|| &centroids[c as usize * d..][..d]);
                let pairs = primary.iter().zip(&want_primary);
                for (at, (&got, &want)) in pairs
                    .chain(secondary.iter().zip(&want_secondary))
                    .enumerate()
                {
                    assert_eq!(
                        centroid(got),
                        centroid(want),
                        "k {k}, replicas {replicas}, slot {at}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
