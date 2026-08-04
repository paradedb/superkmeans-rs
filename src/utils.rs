//! Timing, blob data generation, brute-force reference routines, and small
//! shared numeric kernels.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;
use std::time::Instant;

/// Simple stopwatch matching the C++ TicToc API.
pub struct TicToc {
    accum_ns: u128,
    start: Instant,
}

impl Default for TicToc {
    fn default() -> Self {
        Self::new()
    }
}

impl TicToc {
    pub fn new() -> Self {
        Self {
            accum_ns: 0,
            start: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.accum_ns = 0;
        self.start = Instant::now();
    }

    pub fn tic(&mut self) {
        self.start = Instant::now();
    }

    pub fn toc(&mut self) {
        self.accum_ns += self.start.elapsed().as_nanos();
    }

    pub fn milliseconds(&self) -> f64 {
        self.accum_ns as f64 / 1.0e6
    }
}

/// Generate synthetic clusterable data (scikit-learn-style make_blobs).
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    normalize: bool,
    cluster_std: f32,
    center_spread: f32,
    random_state: u64,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(random_state);
    let center_dist = Normal::new(0.0_f32, center_spread).unwrap();

    let mut centers = vec![0.0_f32; n_centers * n_features];
    for value in centers.iter_mut() {
        *value = center_dist.sample(&mut rng);
    }

    let mut data = vec![0.0_f32; n_samples * n_features];
    let point_dist = Normal::new(0.0_f32, cluster_std).unwrap();
    let cluster_dist = Uniform::new(0_usize, n_centers);

    data.par_chunks_mut(n_features)
        .enumerate()
        .for_each(|(i, row)| {
            let mut local_rng = ChaCha8Rng::seed_from_u64(random_state.wrapping_add(i as u64 + 1));
            let center_idx = cluster_dist.sample(&mut local_rng) * n_features;
            let center = &centers[center_idx..center_idx + n_features];
            for j in 0..n_features {
                row[j] = center[j] + point_dist.sample(&mut local_rng);
            }
        });

    if normalize {
        data.par_chunks_mut(n_features).for_each(|row| {
            let mut norm_sq = 0.0_f32;
            for &v in row.iter() {
                norm_sq += v * v;
            }
            let inv = 1.0 / norm_sq.sqrt().max(f32::EPSILON);
            for v in row.iter_mut() {
                *v *= inv;
            }
        });
    }

    data
}

pub fn generate_random_vectors(
    n: usize,
    d: usize,
    min_val: f32,
    max_val: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Uniform::new(min_val, max_val);
    let mut output = vec![0.0_f32; n * d];
    for v in output.iter_mut() {
        *v = dist.sample(&mut rng);
    }
    output
}

pub fn ceil_to_multiple(x: u32, m: u32) -> u32 {
    if m == 0 { x } else { x.div_ceil(m) * m }
}

pub fn is_power_of_two(x: u32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

/// Reference brute-force search used in tests and for validation.
pub fn compute_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        s += diff * diff;
    }
    s
}

pub fn compute_norms_row_major(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    data.par_chunks(d)
        .zip(out.par_iter_mut())
        .for_each(|(row, n_out)| {
            let mut s = 0.0_f32;
            for &v in row {
                s += v * v;
            }
            *n_out = s;
        });
    out
}

/// Matrix size below which a sequential pass beats spreading the rows over rayon,
/// for the kernels that *reduce* each row.
///
/// `cargo bench --bench utils` puts rayon's fork-join at 13-15 µs when entered
/// from a non-worker thread, which is what training does, while a sequential row
/// reduction holds a steady ~2.2 Gelem/s at every size. Parallelism therefore only
/// pays once a matrix is large enough to cover that fixed cost, measured at
/// somewhere between 49K and 98K elements.
///
/// Calibrated for those non-worker calls. Crossing into the pool once at the top
/// of training instead would move the crossover down.
const REDUCTION_PARALLEL_MIN_ELEMENTS: usize = 64 * 1024;

/// The same crossover for kernels that only *scale* each element, which sits 16x
/// higher.
///
/// Two effects compound. A scale runs sequentially at ~17 Gelem/s, so a given
/// element count buys far less time to cover the dispatch with; and it is memory
/// bound, so spreading it over rayon returns ~1.5x rather than the ~5x a reduction
/// gets. Recovering 13 µs out of a 1.5x saving takes ~800K elements, which is
/// where the measured crossover sits.
const SCALE_PARALLEL_MIN_ELEMENTS: usize = 1024 * 1024;

/// Total squared distance the centroids moved this iteration, summed over the
/// first `n_clusters` rows of two `n_clusters × d` row-major matrices.
///
/// This is k-means' convergence signal: callers compare it against their
/// tolerance to decide whether to stop.
///
/// Dispatches on matrix size between [`centroid_shift_sequential`] and
/// [`centroid_shift_parallel`], which agree only to within f32 reduction order.
/// Both are public so benchmarks can measure where the crossover lies.
pub fn centroid_shift(
    new_centroids: &[f32],
    prev_centroids: &[f32],
    n_clusters: usize,
    d: usize,
) -> f32 {
    if n_clusters * d < REDUCTION_PARALLEL_MIN_ELEMENTS {
        centroid_shift_sequential(new_centroids, prev_centroids, n_clusters, d)
    } else {
        centroid_shift_parallel(new_centroids, prev_centroids, n_clusters, d)
    }
}

/// Single-threaded [`centroid_shift`].
pub fn centroid_shift_sequential(
    new_centroids: &[f32],
    prev_centroids: &[f32],
    n_clusters: usize,
    d: usize,
) -> f32 {
    new_centroids
        .chunks(d)
        .zip(prev_centroids.chunks(d))
        .take(n_clusters)
        .map(|(new_row, prev_row)| row_shift(new_row, prev_row))
        .sum()
}

/// One rayon work item per centroid [`centroid_shift`].
pub fn centroid_shift_parallel(
    new_centroids: &[f32],
    prev_centroids: &[f32],
    n_clusters: usize,
    d: usize,
) -> f32 {
    new_centroids
        .par_chunks(d)
        .zip(prev_centroids.par_chunks(d))
        .take(n_clusters)
        .map(|(new_row, prev_row)| row_shift(new_row, prev_row))
        .sum()
}

/// Replace each row of an `n_rows × d` row-major matrix of sums with its
/// arithmetic mean, dividing row `i` by `counts[i]`.
///
/// Rows with a zero count keep their accumulated value — there is no mean to
/// take — so a caller that cares about empty clusters has to repair them.
///
/// Dispatches on matrix size between [`mean_rows_by_count_sequential`] and
/// [`mean_rows_by_count_parallel`], against a much higher bar than the reduction
/// kernels use — see [`SCALE_PARALLEL_MIN_ELEMENTS`].
pub fn mean_rows_by_count(rows: &mut [f32], counts: &[u32], d: usize) {
    if counts.len() * d < SCALE_PARALLEL_MIN_ELEMENTS {
        mean_rows_by_count_sequential(rows, counts, d);
    } else {
        mean_rows_by_count_parallel(rows, counts, d);
    }
}

/// Single-threaded [`mean_rows_by_count`].
pub fn mean_rows_by_count_sequential(rows: &mut [f32], counts: &[u32], d: usize) {
    rows.chunks_mut(d)
        .zip(counts.iter())
        .for_each(|(row, &count)| mean_row(row, count));
}

/// One rayon work item per row [`mean_rows_by_count`].
pub fn mean_rows_by_count_parallel(rows: &mut [f32], counts: &[u32], d: usize) {
    rows.par_chunks_mut(d)
        .zip(counts.par_iter())
        .for_each(|(row, &count)| mean_row(row, count));
}

#[inline]
fn mean_row(row: &mut [f32], count: u32) {
    if count == 0 {
        return;
    }
    // One divide and `d` multiplies rather than `d` divides.
    let mult = 1.0 / count as f32;
    for v in row.iter_mut() {
        *v *= mult;
    }
}

/// Scale each of the first `n_rows` rows of an `n_rows × d` row-major matrix to
/// unit L2 norm, projecting it onto the unit sphere.
///
/// An all-zero row stays zero rather than becoming NaN.
///
/// Dispatches on matrix size between [`normalize_rows_l2_sequential`] and
/// [`normalize_rows_l2_parallel`].
pub fn normalize_rows_l2(rows: &mut [f32], n_rows: usize, d: usize) {
    if n_rows * d < REDUCTION_PARALLEL_MIN_ELEMENTS {
        normalize_rows_l2_sequential(rows, n_rows, d);
    } else {
        normalize_rows_l2_parallel(rows, n_rows, d);
    }
}

/// Single-threaded [`normalize_rows_l2`].
pub fn normalize_rows_l2_sequential(rows: &mut [f32], n_rows: usize, d: usize) {
    rows.chunks_mut(d).take(n_rows).for_each(normalize_row_l2);
}

/// One rayon work item per row [`normalize_rows_l2`].
pub fn normalize_rows_l2_parallel(rows: &mut [f32], n_rows: usize, d: usize) {
    rows.par_chunks_mut(d)
        .take(n_rows)
        .for_each(normalize_row_l2);
}

#[inline]
fn normalize_row_l2(row: &mut [f32]) {
    let mut sum = 0.0_f32;
    for v in row.iter() {
        sum += v * v;
    }
    let norm = 1.0 / sum.sqrt().max(f32::EPSILON);
    for v in row.iter_mut() {
        *v *= norm;
    }
}

#[inline]
fn row_shift(new_row: &[f32], prev_row: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for k in 0..new_row.len() {
        let diff = new_row[k] - prev_row[k];
        acc += diff * diff;
    }
    acc
}

pub fn find_nearest_neighbor_brute_force(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    out_knn: &mut [u32],
    out_distances: &mut [f32],
) {
    for i in 0..n_x {
        let mut best_dist = f32::MAX;
        let mut best_idx = 0u32;
        for j in 0..n_y {
            let mut dist = 0.0_f32;
            for k in 0..d {
                let diff = x[i * d + k] - y[j * d + k];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = j as u32;
            }
        }
        out_knn[i] = best_idx;
        out_distances[i] = best_dist;
    }
}

pub fn random_vec_seeded(rng: &mut impl Rng, n: usize, low: f32, high: f32) -> Vec<f32> {
    let dist = Uniform::new(low, high);
    (0..n).map(|_| dist.sample(rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_shift_sums_squared_row_displacement() {
        // Two rows of d=2 moving by (1,2) and (0,1): 5 + 1.
        let new = [1.0, 2.0, 5.0, 5.0];
        let prev = [0.0, 0.0, 5.0, 4.0];
        assert_eq!(centroid_shift_sequential(&new, &prev, 2, 2), 6.0);
        assert_eq!(centroid_shift_parallel(&new, &prev, 2, 2), 6.0);
    }

    #[test]
    fn centroid_shift_ignores_rows_past_n_clusters() {
        // Callers pass whole centroid buffers, which outlive the cluster count
        // they are currently training: the tail must not reach the sum.
        let new = [1.0, 2.0, 100.0, 100.0];
        let prev = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(centroid_shift_sequential(&new, &prev, 1, 2), 5.0);
        assert_eq!(centroid_shift_parallel(&new, &prev, 1, 2), 5.0);
    }

    #[test]
    fn mean_rows_by_count_divides_each_row_by_its_own_count() {
        let sums = [3.0, 6.0, 8.0, 12.0];
        let counts = [3, 4];

        let mut rows = sums;
        mean_rows_by_count_sequential(&mut rows, &counts, 2);
        assert_eq!(rows, [1.0, 2.0, 2.0, 3.0]);

        let mut parallel = sums;
        mean_rows_by_count_parallel(&mut parallel, &counts, 2);
        assert_eq!(parallel, [1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_rows_by_count_leaves_uncounted_rows_untouched() {
        // An empty cluster has no mean; `split_clusters` reseeds it instead.
        let sums = [3.0, 6.0, 7.0, 9.0];
        let counts = [3, 0];

        let mut rows = sums;
        mean_rows_by_count_sequential(&mut rows, &counts, 2);
        assert_eq!(rows, [1.0, 2.0, 7.0, 9.0]);

        let mut parallel = sums;
        mean_rows_by_count_parallel(&mut parallel, &counts, 2);
        assert_eq!(parallel, [1.0, 2.0, 7.0, 9.0]);
    }

    #[test]
    fn normalize_rows_l2_scales_each_row_to_unit_norm() {
        let rows = [3.0, 4.0, 0.0, 2.0];

        let mut sequential = rows;
        normalize_rows_l2_sequential(&mut sequential, 2, 2);
        assert_eq!(sequential, [0.6, 0.8, 0.0, 1.0]);

        let mut parallel = rows;
        normalize_rows_l2_parallel(&mut parallel, 2, 2);
        assert_eq!(parallel, [0.6, 0.8, 0.0, 1.0]);
    }

    #[test]
    fn normalize_rows_l2_leaves_zero_rows_at_zero() {
        // A cluster that summed to nothing must not turn into NaN.
        let rows = [0.0, 0.0, 3.0, 4.0];

        let mut sequential = rows;
        normalize_rows_l2_sequential(&mut sequential, 1, 2);
        assert_eq!(sequential, [0.0, 0.0, 3.0, 4.0]);

        let mut parallel = rows;
        normalize_rows_l2_parallel(&mut parallel, 1, 2);
        assert_eq!(parallel, [0.0, 0.0, 3.0, 4.0]);
    }
}
