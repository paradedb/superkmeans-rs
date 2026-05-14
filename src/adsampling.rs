//! ADSampling-style random rotation + per-dimension pruning thresholds.
//!
//! The original C++ uses Eigen's HouseholderQR. We re-implement Householder QR
//! directly in pure Rust (it runs once at construction).

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use crate::common::KnnCandidate;
use crate::gemm;

pub struct ADSamplingPruner {
    pub num_dimensions: usize,
    pub ratios: Vec<f32>,
    pub epsilon0: f32,
    /// Random orthonormal d × d rotation matrix, row-major.
    matrix: Vec<f32>,
}

impl ADSamplingPruner {
    pub fn new(num_dimensions: usize, epsilon0: f32, seed: u64) -> Self {
        let matrix = random_orthonormal_matrix(num_dimensions, seed);
        let mut me = Self {
            num_dimensions,
            ratios: Vec::new(),
            epsilon0,
            matrix,
        };
        me.initialize_ratios();
        me
    }

    pub fn set_epsilon0(&mut self, eps0: f32) {
        self.epsilon0 = eps0;
        self.initialize_ratios();
    }

    fn initialize_ratios(&mut self) {
        let d = self.num_dimensions;
        let mut ratios = vec![0.0_f32; d + 1];
        for i in 0..=d {
            ratios[i] = compute_ratio(i, d, self.epsilon0);
        }
        self.ratios = ratios;
    }

    /// Threshold = best_distance * ratio(current_dim).
    #[inline]
    pub fn pruning_threshold(&self, best: &KnnCandidate, current_dim: usize) -> f32 {
        best.distance * self.ratios[current_dim]
    }

    /// out = vectors * matrix^T
    pub fn rotate(&self, vectors: &[f32], out: &mut [f32], n: usize) {
        let d = self.num_dimensions;
        debug_assert!(vectors.len() >= n * d);
        debug_assert!(out.len() >= n * d);
        unsafe {
            matrixmultiply::sgemm(
                n,
                d,
                d,
                1.0,
                vectors.as_ptr(),
                d as isize,
                1,
                self.matrix.as_ptr(),
                1,
                d as isize,
                0.0,
                out.as_mut_ptr(),
                d as isize,
                1,
            );
        }
    }

    /// out = rotated * matrix   (inverse of `rotate` for orthonormal matrix).
    pub fn unrotate(&self, rotated: &[f32], out: &mut [f32], n: usize) {
        let d = self.num_dimensions;
        gemm::sgemm_row_major(n, d, d, rotated, &self.matrix, out);
    }
}

fn compute_ratio(visited: usize, total: usize, epsilon0: f32) -> f32 {
    if visited == 0 || visited == total {
        return 1.0;
    }
    let v = visited as f64;
    let t = total as f64;
    let eps = epsilon0 as f64;
    let r = (v / t) * (1.0 + eps / v.sqrt()) * (1.0 + eps / v.sqrt());
    r as f32
}

fn random_orthonormal_matrix(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();

    // Column-major working storage: a[c][r] is element (r, c).
    let mut a: Vec<Vec<f32>> = (0..d)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let mut q: Vec<Vec<f32>> = (0..d)
        .map(|c| {
            let mut col = vec![0.0_f32; d];
            col[c] = 1.0;
            col
        })
        .collect();

    for j in 0..d {
        // Reflector for the sub-column A[j..d, j].
        let mut norm_sq = 0.0_f32;
        for i in j..d {
            norm_sq += a[j][i] * a[j][i];
        }
        let norm = norm_sq.sqrt();
        if norm < 1.0e-12 {
            continue;
        }
        let alpha = if a[j][j] >= 0.0 { -norm } else { norm };

        let m = d - j;
        let mut v = vec![0.0_f32; m];
        v[0] = a[j][j] - alpha;
        for i in 1..m {
            v[i] = a[j][i + j];
        }
        let mut v_norm_sq = 0.0_f32;
        for &vi in &v {
            v_norm_sq += vi * vi;
        }
        if v_norm_sq < 1.0e-30 {
            continue;
        }
        let inv = 2.0 / v_norm_sq;

        // A := H A  (H acts on rows j..d).
        for c in j..d {
            let mut dot = 0.0_f32;
            for i in 0..m {
                dot += v[i] * a[c][i + j];
            }
            let scale = inv * dot;
            for i in 0..m {
                a[c][i + j] -= scale * v[i];
            }
        }

        // Q := Q H  (H acts on cols j..d, all rows).
        for r in 0..d {
            let mut dot = 0.0_f32;
            for i in 0..m {
                dot += q[j + i][r] * v[i];
            }
            let scale = inv * dot;
            for i in 0..m {
                q[j + i][r] -= scale * v[i];
            }
        }
    }

    let mut out = vec![0.0_f32; d * d];
    for c in 0..d {
        for r in 0..d {
            out[r * d + c] = q[c][r];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_is_orthonormal() {
        let d = 16;
        let pruner = ADSamplingPruner::new(d, 1.5, 42);
        // Check Q Q^T = I by sampling
        let mut acc = vec![0.0_f32; d * d];
        for r in 0..d {
            for c in 0..d {
                let mut s = 0.0_f32;
                for k in 0..d {
                    s += pruner.matrix[r * d + k] * pruner.matrix[c * d + k];
                }
                acc[r * d + c] = s;
            }
        }
        for r in 0..d {
            for c in 0..d {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (acc[r * d + c] - expected).abs() < 1.0e-4,
                    "QQ^T[{}, {}] = {} (expected {})",
                    r,
                    c,
                    acc[r * d + c],
                    expected
                );
            }
        }
    }

    #[test]
    fn unrotate_inverts_rotate() {
        let d = 16;
        let n = 4;
        let pruner = ADSamplingPruner::new(d, 1.5, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        let input: Vec<f32> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        let mut rotated = vec![0.0_f32; n * d];
        let mut restored = vec![0.0_f32; n * d];
        pruner.rotate(&input, &mut rotated, n);
        pruner.unrotate(&rotated, &mut restored, n);
        for i in 0..n * d {
            assert!(
                (input[i] - restored[i]).abs() < 1.0e-3,
                "mismatch at {}: {} vs {}",
                i,
                input[i],
                restored[i]
            );
        }
    }
}
