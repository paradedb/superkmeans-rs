//! Timing, blob data generation, and brute-force reference routines.

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
