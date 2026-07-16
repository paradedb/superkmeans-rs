//! Core SuperKMeans algorithm: BLAS+pruning k-means.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Uniform, WeightedIndex};
use rayon::prelude::*;

use crate::adsampling::ADSamplingPruner;
use crate::batch;
use crate::common::{
    CENTROID_PERTURBATION_EPS, DIMENSION_THRESHOLD_FOR_PRUNING, MIN_PARTIAL_D,
    N_CLUSTERS_THRESHOLD_FOR_PRUNING, PRUNER_INITIAL_THRESHOLD, RECALL_CONVERGENCE_PATIENCE,
    X_BATCH_SIZE, Y_BATCH_SIZE,
};
use crate::layout;

/// Configuration parameters for SuperKMeans clustering.
#[derive(Clone, Debug)]
pub struct SuperKMeansConfig {
    pub iters: u32,
    pub sampling_fraction: f32,
    pub max_points_per_cluster: u32,
    pub n_threads: u32,
    pub seed: u64,
    pub use_blas_only: bool,
    pub tol: f32,
    pub recall_tol: f32,
    pub early_termination: bool,
    pub sample_queries: bool,
    pub objective_k: usize,
    pub ann_explore_fraction: f32,
    pub min_not_pruned_pct: f32,
    pub max_not_pruned_pct: f32,
    pub adjustment_factor_for_partial_d: f32,
    pub unrotate_centroids: bool,
    pub verbose: bool,
    pub angular: bool,
    pub suppress_warnings: bool,
    pub data_already_rotated: bool,
    /// Use the cuVS-style aggressive small-cluster rebalancing during
    /// consolidation. Hierarchical training opts into this.
    pub use_aggressive_split: bool,
}

impl Default for SuperKMeansConfig {
    fn default() -> Self {
        Self {
            iters: 10,
            sampling_fraction: 0.3,
            max_points_per_cluster: 256,
            n_threads: 0,
            seed: 42,
            use_blas_only: false,
            tol: 1.0e-4,
            recall_tol: 0.005,
            early_termination: true,
            sample_queries: false,
            objective_k: 100,
            ann_explore_fraction: 0.01,
            min_not_pruned_pct: 0.03,
            max_not_pruned_pct: 0.05,
            adjustment_factor_for_partial_d: 0.20,
            unrotate_centroids: true,
            verbose: false,
            angular: false,
            suppress_warnings: false,
            data_already_rotated: false,
            use_aggressive_split: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperKMeansIterationStats {
    pub iteration: usize,
    pub objective: f32,
    pub shift: f32,
    pub split: usize,
    pub recall: f32,
    pub not_pruned_pct: f32,
    pub partial_d: u32,
    pub is_gemm_only: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ClusterBalanceStats {
    pub mean: f32,
    pub geometric_mean: f32,
    pub stdev: f32,
    pub cv: f32,
    pub min: usize,
    pub max: usize,
}

/// Trained SuperKMeans state.
pub struct SuperKMeans {
    pub d: usize,
    pub n_clusters: usize,
    pub config: SuperKMeansConfig,
    pub n_threads: usize,

    pub pruner: ADSamplingPruner,

    // Row-major centroids (this iteration & previous iteration).
    pub horizontal_centroids: Vec<f32>,
    pub prev_centroids: Vec<f32>,

    // Per-cluster size and per-sample assignment/distance.
    pub cluster_sizes: Vec<u32>,
    pub assignments: Vec<u32>,
    pub distances: Vec<f32>,

    // Pre-computed norms; may be either full or partial-d depending on the
    // most recent kernel that wrote them.
    pub data_norms: Vec<f32>,
    pub centroid_norms: Vec<f32>,

    pub sampled_indices: Vec<usize>,

    // Geometry.
    pub vertical_d: usize,
    pub horizontal_d: usize,
    pub partial_d: u32,

    // Iteration state.
    pub trained: bool,
    pub n_split: usize,
    pub n_samples: usize,
    pub centroids_to_explore: usize,
    pub prev_cost: f32,
    pub cost: f32,
    pub shift: f32,
    pub recall: f32,

    pub iteration_stats: Vec<SuperKMeansIterationStats>,
}

impl SuperKMeans {
    pub fn new(n_clusters: usize, dimensionality: usize) -> Self {
        Self::with_config(n_clusters, dimensionality, SuperKMeansConfig::default())
    }

    pub fn with_config(
        n_clusters: usize,
        dimensionality: usize,
        mut config: SuperKMeansConfig,
    ) -> Self {
        assert!(n_clusters > 0, "n_clusters must be positive");
        assert!(dimensionality > 0, "dimensionality must be positive");
        assert!(config.iters > 0, "iters must be positive");
        assert!(
            config.sampling_fraction > 0.0,
            "sampling_fraction must be positive"
        );
        assert!(
            config.sampling_fraction <= 1.0,
            "sampling_fraction must be <= 1.0"
        );

        if config.data_already_rotated {
            config.unrotate_centroids = false;
        }
        let n_threads = if config.n_threads == 0 {
            rayon::current_num_threads()
        } else {
            config.n_threads as usize
        };
        let pruner = ADSamplingPruner::new(dimensionality, PRUNER_INITIAL_THRESHOLD, config.seed);
        let split = layout::get_dimension_split(dimensionality);
        Self {
            d: dimensionality,
            n_clusters,
            config,
            n_threads,
            pruner,
            horizontal_centroids: Vec::new(),
            prev_centroids: Vec::new(),
            cluster_sizes: Vec::new(),
            assignments: Vec::new(),
            distances: Vec::new(),
            data_norms: Vec::new(),
            centroid_norms: Vec::new(),
            sampled_indices: Vec::new(),
            vertical_d: split.vertical_d,
            horizontal_d: split.horizontal_d,
            partial_d: 0,
            trained: false,
            n_split: 0,
            n_samples: 0,
            centroids_to_explore: 0,
            prev_cost: 0.0,
            cost: 0.0,
            shift: 0.0,
            recall: 0.0,
            iteration_stats: Vec::new(),
        }
    }

    /// Train the model and return row-major centroids (n_clusters × d).
    pub fn train(&mut self, data: &[f32], n: usize) -> Vec<f32> {
        self.train_with_queries(data, n, &[], 0)
    }

    pub fn train_with_queries(
        &mut self,
        data: &[f32],
        n: usize,
        queries: &[f32],
        n_queries: usize,
    ) -> Vec<f32> {
        assert!(n > 0, "n must be positive");
        assert!(!self.trained, "The clustering has already been trained");
        assert!(
            n >= self.n_clusters,
            "n must be >= n_clusters ({} < {})",
            n,
            self.n_clusters
        );
        if n_queries > 0 && queries.is_empty() && !self.config.sample_queries {
            panic!("Queries must be provided if n_queries > 0 and sample_queries is false");
        }

        self.iteration_stats.clear();
        self.n_samples = self.compute_n_vectors_to_sample(n);
        assert!(
            self.n_samples >= self.n_clusters,
            "Not enough samples to train (n_samples={}, n_clusters={})",
            self.n_samples,
            self.n_clusters
        );

        let d = self.d;
        let n_clusters = self.n_clusters;
        let n_samples = self.n_samples;

        self.horizontal_centroids = vec![0.0_f32; n_clusters * d];
        self.prev_centroids = vec![0.0_f32; n_clusters * d];
        self.cluster_sizes = vec![0_u32; n_clusters];
        self.assignments = vec![0_u32; n_samples];
        self.distances = vec![0.0_f32; n_samples];
        self.data_norms = vec![0.0_f32; n_samples];
        self.centroid_norms = vec![0.0_f32; n_clusters];

        self.partial_d = (MIN_PARTIAL_D).max((self.vertical_d as u32) / 2);
        if self.partial_d as usize > self.vertical_d {
            self.partial_d = self.vertical_d as u32;
        }

        if self.config.verbose {
            println!("Front dimensions (d') = {}", self.partial_d);
            println!("Trailing dimensions (d'') = {}", d - self.vertical_d);
        }

        // Sample initial centroids (Forgy) and rotate them into prev_centroids/horizontal_centroids.
        let rotate = !self.config.data_already_rotated;
        self.generate_centroids(data, n, rotate);

        if self.config.verbose {
            println!("Sampling data...");
        }
        let data_to_cluster = self.sample_and_rotate_vectors(data, n, rotate);

        // horizontal_centroids currently holds the unrotated Forgy samples.
        // Rotate (or copy) into prev_centroids.
        self.rotate_or_copy_into_prev_centroids(rotate);

        // Compute full norms for first GEMM iteration.
        self.data_norms = compute_norms_row_major(&data_to_cluster, n_samples, d, true);
        self.centroid_norms = compute_norms_row_major(&self.prev_centroids, n_clusters, d, true);

        // Optional recall tracking — skipped for the minimal port; the algorithm runs
        // identically without queries (early-termination on shift/cost still works).
        let _ = queries;
        let _ = n_queries;

        let always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING
            || self.config.use_blas_only
            || n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        let mut partial_norms_computed = false;
        let mut best_recall = 0.0_f32;
        let mut iters_without_improvement: usize = 0;

        let mut not_pruned_counts = vec![0_usize; n_samples];

        for iter_idx in 0..self.config.iters {
            let use_gemm_only = (iter_idx == 0) || always_gemm_only;
            if !use_gemm_only && !partial_norms_computed {
                self.data_norms = compute_partial_norms_row_major(
                    &data_to_cluster,
                    n_samples,
                    d,
                    self.partial_d as usize,
                );
                partial_norms_computed = true;
            }
            self.run_iteration(
                &data_to_cluster,
                iter_idx,
                iter_idx == 0,
                use_gemm_only,
                &mut not_pruned_counts,
            );

            if self.config.early_termination
                && self.should_stop_early(
                    false,
                    &mut best_recall,
                    &mut iters_without_improvement,
                    iter_idx,
                )
            {
                break;
            }
        }

        self.trained = true;
        self.get_output_centroids(self.config.unrotate_centroids)
    }

    /// Brute-force assignment: returns the index of the nearest centroid for
    /// each vector (data and centroids assumed to share the same domain).
    pub fn assign(&self, vectors: &[f32], centroids: &[f32], n_vectors: usize) -> Vec<u32> {
        let d = self.d;
        let n_centroids = centroids.len() / d;
        let mut result_assignments = vec![0_u32; n_vectors];
        let mut result_distances = vec![0.0_f32; n_vectors];
        let vector_norms = compute_norms_row_major(vectors, n_vectors, d, true);
        let centroid_norms = compute_norms_row_major(centroids, n_centroids, d, true);
        batch::find_nearest_neighbor(
            vectors,
            centroids,
            n_vectors,
            n_centroids,
            d,
            &vector_norms,
            &centroid_norms,
            &mut result_assignments,
            &mut result_distances,
        );
        result_assignments
    }

    /// Fast assignment of training points using pruning.
    pub fn assign_training_points(
        &mut self,
        vectors: &[f32],
        centroids: &[f32],
        n_vectors: usize,
    ) -> Vec<u32> {
        assert!(
            self.trained,
            "assign_training_points requires training first"
        );
        let d = self.d;
        let n_centroids = centroids.len() / d;

        // Fall-back path: when pruning isn't beneficial, defer to brute force.
        if self.config.use_blas_only
            || d < DIMENSION_THRESHOLD_FOR_PRUNING
            || n_centroids <= N_CLUSTERS_THRESHOLD_FOR_PRUNING
        {
            if !self.config.suppress_warnings {
                eprintln!(
                    "WARNING: AssignTrainingPoints cannot use pruning, falling back to brute force"
                );
            }
            return self.assign(vectors, centroids, n_vectors);
        }

        self.partial_d = (MIN_PARTIAL_D).max((self.vertical_d as u32) / 2);

        let mut not_pruned_counts = vec![0_usize; n_vectors];
        let mut data_buffer = vec![0.0_f32; n_vectors * d];
        let data_p: &[f32] = if self.config.data_already_rotated {
            vectors
        } else {
            self.pruner.rotate(vectors, &mut data_buffer, n_vectors);
            &data_buffer
        };

        // Norms from the rotated centroids (the pruning space), not the passed-in
        // `centroids` (unrotated; used only by the brute-force fallback).
        let centroid_partial_norms = compute_partial_norms_row_major(
            &self.horizontal_centroids,
            n_centroids,
            d,
            self.partial_d as usize,
        );

        let mut result_assignments = vec![0_u32; n_vectors];
        let mut result_distances = vec![0.0_f32; n_vectors];

        if self.config.sampling_fraction == 1.0 {
            let data_partial_norms =
                compute_partial_norms_row_major(data_p, n_vectors, d, self.partial_d as usize);
            // Seed assignments from training run; for sampling_fraction=1, training assignments
            // match the natural indices.
            result_assignments
                .copy_from_slice(&self.assignments[..n_vectors.min(self.assignments.len())]);
            batch::find_nearest_neighbor_with_pruning(
                data_p,
                &self.horizontal_centroids,
                n_vectors,
                n_centroids,
                d,
                self.vertical_d,
                self.horizontal_d,
                &data_partial_norms,
                &centroid_partial_norms,
                &mut result_assignments,
                &mut result_distances,
                &self.pruner,
                self.partial_d as usize,
                &mut not_pruned_counts,
            );
            return result_assignments;
        }

        // For non-full sampling, seed assignments with a distribution proportional to
        // the trained cluster sizes (matches the FAISS-style heuristic in the C++).
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed.wrapping_add(1));
        if self.config.sampling_fraction > 0.8 {
            let n_samples = self.n_samples;
            for cur in 0..n_samples {
                let orig = self.sampled_indices[cur];
                if orig < n_vectors {
                    result_assignments[orig] = self.assignments[cur];
                }
            }
            let weights: Vec<u32> = self
                .cluster_sizes
                .iter()
                .copied()
                .map(|s| s.max(1))
                .collect();
            let weighted = WeightedIndex::new(&weights).expect("non-empty weights");
            for cur in n_samples..n_vectors {
                let orig = if cur < self.sampled_indices.len() {
                    self.sampled_indices[cur]
                } else {
                    cur
                };
                if orig < n_vectors {
                    result_assignments[orig] = weighted.sample(&mut rng) as u32;
                }
            }

            let data_partial_norms =
                compute_partial_norms_row_major(data_p, n_vectors, d, self.partial_d as usize);
            batch::find_nearest_neighbor_with_pruning(
                data_p,
                &self.horizontal_centroids,
                n_vectors,
                n_centroids,
                d,
                self.vertical_d,
                self.horizontal_d,
                &data_partial_norms,
                &centroid_partial_norms,
                &mut result_assignments,
                &mut result_distances,
                &self.pruner,
                self.partial_d as usize,
                &mut not_pruned_counts,
            );
            return result_assignments;
        }

        // Very small sampling fractions: meso-cluster the centroids to seed assignments.
        let new_n_centroids = ((n_centroids as f64).sqrt() as usize).max(1);
        let tmp_config = SuperKMeansConfig {
            iters: 10,
            sampling_fraction: 1.0,
            use_blas_only: false,
            verbose: self.config.verbose,
            suppress_warnings: self.config.suppress_warnings,
            seed: self.config.seed,
            angular: self.config.angular,
            data_already_rotated: self.config.data_already_rotated,
            ..Default::default()
        };
        let mut tmp_kmeans = SuperKMeans::with_config(new_n_centroids, d, tmp_config);
        let meso_centroids = tmp_kmeans.train(centroids, n_centroids);
        let meso_assignments = tmp_kmeans.assign(vectors, &meso_centroids, n_vectors);
        let centroids_to_meso = tmp_kmeans.assign(centroids, &meso_centroids, n_centroids);
        let mut meso_to_original = vec![0_u32; new_n_centroids];
        for c in 0..n_centroids {
            meso_to_original[centroids_to_meso[c] as usize] = c as u32;
        }
        let n_samples = self.n_samples;
        for cur in 0..n_samples {
            let orig = self.sampled_indices[cur];
            if orig < n_vectors {
                result_assignments[orig] = self.assignments[cur];
            }
        }
        for cur in n_samples..n_vectors {
            let orig = if cur < self.sampled_indices.len() {
                self.sampled_indices[cur]
            } else {
                cur
            };
            if orig < n_vectors {
                result_assignments[orig] = meso_to_original[meso_assignments[orig] as usize];
            }
        }

        let data_partial_norms =
            compute_partial_norms_row_major(data_p, n_vectors, d, self.partial_d as usize);
        batch::find_nearest_neighbor_with_pruning(
            data_p,
            &self.horizontal_centroids,
            n_vectors,
            n_centroids,
            d,
            self.vertical_d,
            self.horizontal_d,
            &data_partial_norms,
            &centroid_partial_norms,
            &mut result_assignments,
            &mut result_distances,
            &self.pruner,
            self.partial_d as usize,
            &mut not_pruned_counts,
        );

        result_assignments
    }

    // ----------- internals -----------

    pub(crate) fn run_iteration(
        &mut self,
        data: &[f32],
        iter_idx: u32,
        is_first_iter: bool,
        gemm_only: bool,
        not_pruned_counts: &mut [usize],
    ) {
        let d = self.d;
        let n_clusters = self.n_clusters;
        let n_samples = self.n_samples;

        if !is_first_iter {
            std::mem::swap(&mut self.horizontal_centroids, &mut self.prev_centroids);
        }

        if gemm_only {
            self.centroid_norms =
                compute_norms_row_major(&self.prev_centroids, n_clusters, d, true);
            batch::find_nearest_neighbor(
                data,
                &self.prev_centroids,
                n_samples,
                n_clusters,
                d,
                &self.data_norms,
                &self.centroid_norms,
                &mut self.assignments,
                &mut self.distances,
            );
            self.horizontal_centroids.iter_mut().for_each(|v| *v = 0.0);
            self.cluster_sizes.iter_mut().for_each(|v| *v = 0);
        } else {
            // Partial norms for pruning.
            self.centroid_norms = compute_partial_norms_row_major(
                &self.prev_centroids,
                n_clusters,
                d,
                self.partial_d as usize,
            );
            for v in not_pruned_counts.iter_mut() {
                *v = 0;
            }
            batch::find_nearest_neighbor_with_pruning(
                data,
                &self.prev_centroids,
                n_samples,
                n_clusters,
                d,
                self.vertical_d,
                self.horizontal_d,
                &self.data_norms,
                &self.centroid_norms,
                &mut self.assignments,
                &mut self.distances,
                &self.pruner,
                self.partial_d as usize,
                not_pruned_counts,
            );
            self.horizontal_centroids.iter_mut().for_each(|v| *v = 0.0);
            self.cluster_sizes.iter_mut().for_each(|v| *v = 0);
        }

        self.update_centroids(data, n_samples, n_clusters);

        let mut avg_not_pruned_pct = -1.0_f32;
        let old_partial_d = self.partial_d;
        if !gemm_only {
            let (avg, changed) = self.tune_partial_d(not_pruned_counts, n_samples, n_clusters);
            avg_not_pruned_pct = avg;
            if changed {
                self.data_norms =
                    compute_partial_norms_row_major(data, n_samples, d, self.partial_d as usize);
            }
        }

        self.consolidate_centroids(n_samples, n_clusters);
        self.compute_cost();
        self.compute_shift(n_clusters);

        let stats = SuperKMeansIterationStats {
            iteration: (iter_idx + 1) as usize,
            objective: self.cost,
            shift: self.shift,
            split: self.n_split,
            recall: self.recall,
            not_pruned_pct: if gemm_only { -1.0 } else { avg_not_pruned_pct },
            partial_d: if gemm_only { 0 } else { old_partial_d },
            is_gemm_only: gemm_only,
        };
        if self.config.verbose {
            let improvement = if iter_idx > 0 {
                1.0 - (self.cost / self.prev_cost.max(f32::EPSILON))
            } else {
                0.0
            };
            print!(
                "Iteration {}/{} | Objective: {:.4} | Objective improvement: {:.4} | Shift: {:.4} | Split: {}",
                iter_idx + 1,
                self.config.iters,
                self.cost,
                improvement,
                self.shift,
                self.n_split
            );
            if gemm_only {
                println!(" [BLAS-only]");
            } else {
                println!(
                    " | Not Pruned %: {:.4} | d': {} -> {}",
                    avg_not_pruned_pct * 100.0,
                    old_partial_d,
                    self.partial_d
                );
            }
        }
        self.iteration_stats.push(stats);
    }

    pub(crate) fn update_centroids(&mut self, data: &[f32], n_samples: usize, n_clusters: usize) {
        let d = self.d;
        // For correctness with rayon, parallelise over centroids: each thread
        // handles a contiguous slice c0..c1 of clusters and scans all samples,
        // matching the C++ kernel.
        let nt = self.n_threads.max(1);
        let hc_ptr = self.horizontal_centroids.as_mut_ptr() as usize;
        let cs_ptr = self.cluster_sizes.as_mut_ptr() as usize;
        let assignments = &self.assignments;
        rayon::scope(|s| {
            for rank in 0..nt {
                let c0 = n_clusters * rank / nt;
                let c1 = n_clusters * (rank + 1) / nt;
                s.spawn(move |_| {
                    let hc = hc_ptr as *mut f32;
                    let cs = cs_ptr as *mut u32;
                    for i in 0..n_samples {
                        let ci = assignments[i] as usize;
                        if ci >= c0 && ci < c1 {
                            unsafe {
                                *cs.add(ci) = (*cs.add(ci)) + 1;
                            }
                            let vector = &data[i * d..(i + 1) * d];
                            unsafe {
                                let row = hc.add(ci * d);
                                for j in 0..d {
                                    *row.add(j) += vector[j];
                                }
                            }
                        }
                    }
                });
            }
        });
    }

    pub(crate) fn consolidate_centroids(&mut self, n_samples: usize, n_clusters: usize) {
        let d = self.d;
        self.horizontal_centroids
            .par_chunks_mut(d)
            .zip(self.cluster_sizes.par_iter())
            .for_each(|(row, &size)| {
                if size == 0 {
                    return;
                }
                let mult = 1.0 / size as f32;
                for v in row.iter_mut() {
                    *v *= mult;
                }
            });

        self.split_clusters(n_samples, n_clusters);

        if self.config.angular {
            self.postprocess_centroids(n_clusters);
        }
    }

    pub(crate) fn split_clusters(&mut self, n_samples: usize, n_clusters: usize) {
        self.n_split = 0;
        let d = self.d;
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let uniform = Uniform::new(0.0_f32, 1.0_f32);

        // Empty-cluster handling: find a donor and copy + symmetric-perturb.
        for ci in 0..n_clusters {
            if self.cluster_sizes[ci] != 0 {
                continue;
            }
            let mut cj = 0usize;
            loop {
                let size_j = self.cluster_sizes[cj] as f32;
                let denom = (n_samples as f32 - n_clusters as f32).max(1.0);
                let p = (size_j - 1.0) / denom;
                let r = uniform.sample(&mut rng);
                if r < p {
                    break;
                }
                cj = (cj + 1) % n_clusters;
            }
            let (left, right) = split_two_rows_mut(&mut self.horizontal_centroids, ci, cj, d);
            left.copy_from_slice(right);
            for j in 0..d {
                if j % 2 == 0 {
                    left[j] *= 1.0 + CENTROID_PERTURBATION_EPS;
                    right[j] *= 1.0 - CENTROID_PERTURBATION_EPS;
                } else {
                    left[j] *= 1.0 - CENTROID_PERTURBATION_EPS;
                    right[j] *= 1.0 + CENTROID_PERTURBATION_EPS;
                }
            }
            let half = self.cluster_sizes[cj] / 2;
            self.cluster_sizes[ci] = half;
            self.cluster_sizes[cj] -= half;
            self.n_split += 1;
        }

        // Optional aggressive cuVS-style balancing: pull small clusters towards
        // points from large ones.
        if !self.config.use_aggressive_split {
            return;
        }
        const CENTER_ADJUSTMENT_WEIGHT: f32 = 7.0;
        const BALANCING_THRESHOLD: f32 = 0.25;
        let average_size = (n_samples / n_clusters.max(1)) as f32;
        let threshold_size = (average_size * BALANCING_THRESHOLD) as u32;
        for ci in 0..n_clusters {
            let csize = self.cluster_sizes[ci];
            if csize == 0 || csize as f32 > threshold_size as f32 {
                continue;
            }
            let mut large_idx = 0usize;
            loop {
                let large_size = self.cluster_sizes[large_idx];
                if (large_size as f32) < average_size {
                    large_idx = (large_idx + 1) % n_clusters;
                    continue;
                }
                let p = (large_size as f32 - average_size + 1.0)
                    / ((n_samples as f32) - average_size * (n_clusters as f32) + n_clusters as f32);
                let r = uniform.sample(&mut rng);
                if r < p {
                    break;
                }
                large_idx = (large_idx + 1) % n_clusters;
            }
            let (small_row, large_row) =
                split_two_rows_mut(&mut self.horizontal_centroids, ci, large_idx, d);
            let wc = (csize as f32).min(CENTER_ADJUSTMENT_WEIGHT);
            let wd = 1.0_f32;
            for j in 0..d {
                let v = (wc * small_row[j] + wd * large_row[j]) / (wc + wd);
                small_row[j] = v;
            }
            self.n_split += 1;
        }
    }

    pub(crate) fn postprocess_centroids(&mut self, n_clusters: usize) {
        let d = self.d;
        self.horizontal_centroids
            .par_chunks_mut(d)
            .take(n_clusters)
            .for_each(|row| {
                let mut sum = 0.0_f32;
                for v in row.iter() {
                    sum += v * v;
                }
                let norm = 1.0 / sum.sqrt().max(f32::EPSILON);
                for v in row.iter_mut() {
                    *v *= norm;
                }
            });
    }

    pub(crate) fn compute_cost(&mut self) {
        self.prev_cost = self.cost;
        self.cost = self.distances.iter().sum::<f32>();
    }

    pub(crate) fn compute_shift(&mut self, n_clusters: usize) {
        let d = self.d;
        let total: f32 = self
            .horizontal_centroids
            .par_chunks(d)
            .zip(self.prev_centroids.par_chunks(d))
            .take(n_clusters)
            .map(|(new_row, prev_row)| {
                let mut acc = 0.0_f32;
                for k in 0..d {
                    let diff = new_row[k] - prev_row[k];
                    acc += diff * diff;
                }
                acc
            })
            .sum();
        self.shift = total;
    }

    pub(crate) fn tune_partial_d(
        &mut self,
        not_pruned_counts: &[usize],
        n_samples: usize,
        n_y: usize,
    ) -> (f32, bool) {
        let sum: f64 = not_pruned_counts
            .iter()
            .take(n_samples)
            .map(|&v| v as f64)
            .sum();
        let avg = (sum / (n_samples as f64 * n_y as f64)) as f32;
        let old_partial_d = self.partial_d;
        if avg > self.config.max_not_pruned_pct {
            let increase = ((self.partial_d as f32)
                * self.config.adjustment_factor_for_partial_d
                * 2.0) as u32;
            self.partial_d = (self.partial_d + increase.max(1)).min(self.vertical_d as u32);
        } else if avg < self.config.min_not_pruned_pct {
            let decrease =
                ((self.partial_d as f32) * self.config.adjustment_factor_for_partial_d) as u32;
            self.partial_d = (self.partial_d.saturating_sub(decrease.max(1))).max(MIN_PARTIAL_D);
        }
        (avg, old_partial_d != self.partial_d)
    }

    pub(crate) fn should_stop_early(
        &mut self,
        tracking_recall: bool,
        best_recall: &mut f32,
        iters_without_improvement: &mut usize,
        iter_idx: u32,
    ) -> bool {
        if self.shift < self.config.tol {
            if self.config.verbose {
                println!(
                    "Converged at iteration {} (shift {} < tol {})",
                    iter_idx + 1,
                    self.shift,
                    self.config.tol
                );
            }
            return true;
        }
        if iter_idx > 0 {
            let cost_delta = self.cost / self.prev_cost.max(f32::EPSILON);
            if cost_delta > 1.0 - self.config.tol {
                if self.config.verbose {
                    println!(
                        "Converged at iteration {} (cost improved by only {})",
                        iter_idx + 1,
                        1.0 - cost_delta
                    );
                }
                return true;
            }
        }
        if tracking_recall {
            let improvement = self.recall - *best_recall;
            if improvement > self.config.recall_tol {
                *best_recall = self.recall;
                *iters_without_improvement = 0;
            } else {
                *iters_without_improvement += 1;
                if *iters_without_improvement >= RECALL_CONVERGENCE_PATIENCE {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn compute_n_vectors_to_sample(&self, n: usize) -> usize {
        if self.config.sampling_fraction == 1.0 {
            return n;
        }
        let by_clusters = self.n_clusters * self.config.max_points_per_cluster as usize;
        let by_n = ((n as f64) * self.config.sampling_fraction as f64).floor() as usize;
        by_n.min(by_clusters)
    }

    pub(crate) fn generate_centroids(&mut self, data: &[f32], n: usize, rotate: bool) {
        let d = self.d;
        let n_clusters = self.n_clusters;
        if self.horizontal_centroids.len() < n_clusters * d {
            self.horizontal_centroids.resize(n_clusters * d, 0.0);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        for i in 0..n_clusters {
            let src = indices[i] * d;
            let dst = i * d;
            self.horizontal_centroids[dst..dst + d].copy_from_slice(&data[src..src + d]);
        }

        if rotate {
            let mut rotated = vec![0.0_f32; n_clusters * d];
            self.pruner
                .rotate(&self.horizontal_centroids, &mut rotated, n_clusters);
            self.horizontal_centroids[..n_clusters * d].copy_from_slice(&rotated);
        }
    }

    pub(crate) fn sample_and_rotate_vectors(
        &mut self,
        data: &[f32],
        n: usize,
        rotate: bool,
    ) -> Vec<f32> {
        let d = self.d;
        let n_samples = self.n_samples;

        if n_samples < n {
            if self.config.verbose {
                println!("Sampling {} vectors", n_samples);
            }
            let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
            self.sampled_indices = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                self.sampled_indices.swap(i, j);
            }
            let mut samples = vec![0.0_f32; n_samples * d];
            samples.par_chunks_mut(d).enumerate().for_each(|(i, dst)| {
                let src = self.sampled_indices[i] * d;
                dst.copy_from_slice(&data[src..src + d]);
            });
            if rotate {
                let mut rotated = vec![0.0_f32; n_samples * d];
                self.pruner.rotate(&samples, &mut rotated, n_samples);
                rotated
            } else {
                samples
            }
        } else {
            // No sampling.
            self.sampled_indices = (0..n).collect();
            if self.config.verbose {
                println!("Using {} vectors", n_samples);
            }
            if rotate {
                let mut rotated = vec![0.0_f32; n_samples * d];
                self.pruner
                    .rotate(&data[..n_samples * d], &mut rotated, n_samples);
                rotated
            } else {
                data[..n_samples * d].to_vec()
            }
        }
    }

    /// Copy horizontal_centroids -> prev_centroids, applying rotation if needed.
    fn rotate_or_copy_into_prev_centroids(&mut self, rotate: bool) {
        let d = self.d;
        let n_clusters = self.n_clusters;
        if rotate {
            // horizontal_centroids is already rotated (generate_centroids did it).
            // We just need to mirror into prev_centroids.
            self.prev_centroids[..n_clusters * d]
                .copy_from_slice(&self.horizontal_centroids[..n_clusters * d]);
        } else {
            self.prev_centroids[..n_clusters * d]
                .copy_from_slice(&self.horizontal_centroids[..n_clusters * d]);
        }
    }

    pub(crate) fn get_output_centroids(&self, unrotate: bool) -> Vec<f32> {
        let d = self.d;
        let n_clusters = self.n_clusters;
        if unrotate {
            let mut out = vec![0.0_f32; n_clusters * d];
            self.pruner
                .unrotate(&self.horizontal_centroids, &mut out, n_clusters);
            out
        } else {
            self.horizontal_centroids[..n_clusters * d].to_vec()
        }
    }

    /// Cluster-balance summary stats over the assignments.
    pub fn cluster_balance_stats(
        assignments: &[u32],
        n_samples: usize,
        n_clusters: usize,
    ) -> ClusterBalanceStats {
        let mut sizes = vec![0_usize; n_clusters];
        for i in 0..n_samples {
            sizes[assignments[i] as usize] += 1;
        }
        let mean = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let mut log_sum = 0.0_f32;
        let mut non_zero = 0;
        for &s in &sizes {
            if s > 0 {
                log_sum += (s as f32).ln();
                non_zero += 1;
            }
        }
        let geometric_mean = if non_zero > 0 {
            (log_sum / non_zero as f32).exp()
        } else {
            0.0
        };
        let sq_sum: usize = sizes.iter().map(|&s| s * s).sum();
        let stdev = (sq_sum as f32 / sizes.len() as f32 - mean * mean)
            .max(0.0)
            .sqrt();
        let cv = if mean != 0.0 { stdev / mean } else { 0.0 };
        let min = *sizes.iter().min().unwrap_or(&0);
        let max = *sizes.iter().max().unwrap_or(&0);
        ClusterBalanceStats {
            mean,
            geometric_mean,
            stdev,
            cv,
            min,
            max,
        }
    }
}

pub(crate) fn compute_norms_row_major(data: &[f32], n: usize, d: usize, _full: bool) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    out.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let row = &data[i * d..(i + 1) * d];
        let mut s = 0.0_f32;
        for &v in row {
            s += v * v;
        }
        *dst = s;
    });
    out
}

pub(crate) fn compute_partial_norms_row_major(
    data: &[f32],
    n: usize,
    d: usize,
    partial_d: usize,
) -> Vec<f32> {
    let p = partial_d.min(d);
    let mut out = vec![0.0_f32; n];
    out.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let row = &data[i * d..i * d + p];
        let mut s = 0.0_f32;
        for &v in row {
            s += v * v;
        }
        *dst = s;
    });
    out
}

/// Borrow two disjoint rows of a row-major matrix mutably.
fn split_two_rows_mut(data: &mut [f32], a: usize, b: usize, d: usize) -> (&mut [f32], &mut [f32]) {
    assert!(a != b, "split_two_rows_mut requires distinct rows");
    if a < b {
        let (left, right) = data.split_at_mut(b * d);
        (&mut left[a * d..(a + 1) * d], &mut right[..d])
    } else {
        let (left, right) = data.split_at_mut(a * d);
        (&mut right[..d], &mut left[b * d..(b + 1) * d])
    }
}

// Suppress unused-constant warnings for re-export crumbs.
const _: () = {
    let _ = X_BATCH_SIZE;
    let _ = Y_BATCH_SIZE;
};
