//! Hierarchical SuperKMeans: 3-phase clustering for very large k.

use rayon::prelude::*;

use crate::common::{
    DIMENSION_THRESHOLD_FOR_PRUNING, HIERARCHICAL_PRUNER_INITIAL_THRESHOLD, MIN_PARTIAL_D,
    N_CLUSTERS_THRESHOLD_FOR_PRUNING,
};
use crate::layout;
use crate::superkmeans::{
    SuperKMeans, SuperKMeansConfig, SuperKMeansIterationStats, compute_norms_row_major,
    compute_partial_norms_row_major,
};

/// Hierarchical-specific config — wraps SuperKMeansConfig with extra phase iters.
#[derive(Clone, Debug)]
pub struct HierarchicalSuperKMeansConfig {
    pub base: SuperKMeansConfig,
    pub iters_mesoclustering: u32,
    pub iters_fineclustering: u32,
    pub iters_refinement: u32,
}

impl Default for HierarchicalSuperKMeansConfig {
    fn default() -> Self {
        let mut base = SuperKMeansConfig::default();
        base.sampling_fraction = 1.0;
        base.use_aggressive_split = true;
        Self {
            base,
            iters_mesoclustering: 3,
            iters_fineclustering: 5,
            iters_refinement: 0,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct HierarchicalSuperKMeansIterationStats {
    pub mesoclustering: Vec<SuperKMeansIterationStats>,
    pub fineclustering: Vec<SuperKMeansIterationStats>,
    pub refinement: Vec<SuperKMeansIterationStats>,
}

pub struct HierarchicalSuperKMeans {
    pub base: SuperKMeans,
    pub config: HierarchicalSuperKMeansConfig,
    pub iteration_stats: HierarchicalSuperKMeansIterationStats,
    pub n_mesoclusters: usize,
}

impl HierarchicalSuperKMeans {
    pub fn new(n_clusters: usize, dimensionality: usize) -> Self {
        Self::with_config(
            n_clusters,
            dimensionality,
            HierarchicalSuperKMeansConfig::default(),
        )
    }

    pub fn with_config(
        n_clusters: usize,
        dimensionality: usize,
        config: HierarchicalSuperKMeansConfig,
    ) -> Self {
        assert!(
            config.iters_mesoclustering > 0,
            "iters_mesoclustering must be positive"
        );
        assert!(
            config.iters_fineclustering > 0,
            "iters_fineclustering must be positive"
        );

        let mut base_config = config.base.clone();
        base_config.use_aggressive_split = true;
        let mut base = SuperKMeans::with_config(n_clusters, dimensionality, base_config);
        // The hierarchical variant uses a less aggressive pruning ratio. Only the
        // per-dimension ratios differ from the base pruner — the random rotation
        // matrix is a function of (dimensionality, seed), both identical here — so
        // adjust epsilon0 in place instead of rebuilding it, which would repeat the
        // O(d³) Householder QR for a bit-identical matrix.
        base.pruner
            .set_epsilon0(HIERARCHICAL_PRUNER_INITIAL_THRESHOLD);

        if n_clusters <= 128 && !config.base.suppress_warnings {
            eprintln!(
                "WARNING: n_clusters <= 128 is not recommended for HierarchicalSuperKMeans. \
                 Consider using at least 128 clusters."
            );
        }

        Self {
            base,
            config,
            iteration_stats: HierarchicalSuperKMeansIterationStats::default(),
            n_mesoclusters: 0,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) -> Vec<f32> {
        assert!(n > 0, "n must be positive");
        assert!(!self.base.trained, "already trained");
        assert!(
            n >= self.base.n_clusters,
            "n must be >= n_clusters ({} < {})",
            n,
            self.base.n_clusters
        );

        let d = self.base.d;
        let n_clusters = self.base.n_clusters;
        self.n_mesoclusters = (n_clusters as f64).sqrt().round() as usize;

        self.iteration_stats = HierarchicalSuperKMeansIterationStats::default();

        let split = layout::get_dimension_split(d);
        self.base.vertical_d = split.vertical_d;
        self.base.horizontal_d = split.horizontal_d;

        self.base.n_samples = self.compute_n_to_sample(n);
        assert!(
            self.base.n_samples >= n_clusters,
            "Not enough samples to train"
        );

        // Allocate top-level buffers.
        self.base.horizontal_centroids = vec![0.0_f32; n_clusters * d];
        self.base.prev_centroids = vec![0.0_f32; n_clusters * d];
        self.base.cluster_sizes = vec![0_u32; n_clusters];
        self.base.assignments = vec![0_u32; self.base.n_samples];
        self.base.distances = vec![0.0_f32; self.base.n_samples];
        self.base.data_norms = vec![0.0_f32; self.base.n_samples];
        self.base.centroid_norms = vec![0.0_f32; n_clusters];

        let mut final_assignments = vec![0_u32; self.base.n_samples];
        let mut final_centroids = vec![0.0_f32; n_clusters * d];

        self.base.partial_d = (MIN_PARTIAL_D).max((self.base.vertical_d as u32) / 2);
        if self.base.partial_d as usize > self.base.vertical_d {
            self.base.partial_d = self.base.vertical_d as u32;
        }
        let initial_partial_d = self.base.partial_d;

        // ----- Phase 1: mesoclustering -----
        if self.base.config.verbose {
            println!(
                "\n=== PHASE 1: MESOCLUSTERING (k={} clusters) ===",
                self.n_mesoclusters
            );
        }

        let rotate = !self.base.config.data_already_rotated;
        // Generate mesocluster initial centroids.
        self.base.n_clusters = self.n_mesoclusters; // temporarily for generate_centroids
        self.base.generate_centroids(data, n, rotate);
        // Restore for storage sizing semantics; n_clusters drives storage but generate_centroids
        // only writes the first n_mesoclusters rows.
        self.base.n_clusters = n_clusters;

        let data_to_cluster = self.base.sample_and_rotate_vectors(data, n, rotate);

        // Mirror horizontal_centroids -> prev_centroids for first iteration.
        self.base.prev_centroids[..self.n_mesoclusters * d]
            .copy_from_slice(&self.base.horizontal_centroids[..self.n_mesoclusters * d]);

        self.base.data_norms =
            compute_norms_row_major(&data_to_cluster, self.base.n_samples, d, true);
        self.base.centroid_norms =
            compute_norms_row_major(&self.base.prev_centroids, self.n_mesoclusters, d, true);

        let immutable_data_norms = self.base.data_norms.clone();

        let mut not_pruned_counts = vec![0_usize; self.base.n_samples];

        let always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING
            || self.base.config.use_blas_only
            || self.n_mesoclusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        let mut partial_norms_computed = false;
        let mut best_recall = 0.0;
        let mut iters_without_improvement: usize = 0;

        // Mesoclustering loop. Run with n_clusters = n_mesoclusters.
        self.base.n_clusters = self.n_mesoclusters;
        for iter_idx in 0..self.config.iters_mesoclustering {
            let use_gemm_only = iter_idx == 0 || always_gemm_only;
            if !use_gemm_only && !partial_norms_computed {
                self.base.data_norms = compute_partial_norms_row_major(
                    &data_to_cluster,
                    self.base.n_samples,
                    d,
                    self.base.partial_d as usize,
                );
                partial_norms_computed = true;
            }
            self.base.run_iteration(
                &data_to_cluster,
                iter_idx,
                iter_idx == 0,
                use_gemm_only,
                &mut not_pruned_counts,
            );
            if let Some(stat) = self.base.iteration_stats.pop() {
                self.iteration_stats.mesoclustering.push(stat);
            }
            if self.base.config.early_termination
                && self.base.should_stop_early(
                    false,
                    &mut best_recall,
                    &mut iters_without_improvement,
                    iter_idx,
                )
            {
                break;
            }
        }
        self.base.n_clusters = n_clusters;

        // Snapshot mesocluster state.
        let mesocluster_sizes: Vec<u32> = self.base.cluster_sizes[..self.n_mesoclusters].to_vec();
        let mesocluster_assignments: Vec<u32> =
            self.base.assignments[..self.base.n_samples].to_vec();

        // Build partitioned index for the fine-clustering phase.
        let mut mesocluster_offsets = vec![0_usize; self.n_mesoclusters + 1];
        for k in 0..self.n_mesoclusters {
            mesocluster_offsets[k + 1] = mesocluster_offsets[k] + mesocluster_sizes[k] as usize;
        }
        let mut mesocluster_indices_flat = vec![0_usize; self.base.n_samples];
        {
            let mut next_to_write = mesocluster_offsets.clone();
            for i in 0..self.base.n_samples {
                let cid = mesocluster_assignments[i] as usize;
                mesocluster_indices_flat[next_to_write[cid]] = i;
                next_to_write[cid] += 1;
            }
        }

        // ----- Phase 2: fine clustering -----
        if self.base.config.verbose {
            println!(
                "\n=== PHASE 2: FINE-CLUSTERING (subdividing {} mesoclusters into total {} clusters) ===",
                self.n_mesoclusters, n_clusters
            );
        }

        let fine_clusters_nums = self.arrange_fine_clusters(
            n_clusters,
            self.n_mesoclusters,
            self.base.n_samples,
            &mesocluster_sizes,
        );

        let max_mesocluster_size = *mesocluster_sizes.iter().max().unwrap_or(&0) as usize;
        let mut mesocluster_buffer = vec![0.0_f32; max_mesocluster_size * d];
        let mut assignments_indirection = vec![0_u32; max_mesocluster_size];

        let mut fineclusters_offset: usize = 0;
        for k in 0..self.n_mesoclusters {
            let n_fine = fine_clusters_nums[k];
            if n_fine == 0 {
                continue;
            }
            self.base.partial_d = initial_partial_d;
            let mesocluster_size = mesocluster_sizes[k] as usize;
            self.base.n_samples = mesocluster_size;

            self.compact_mesocluster_to_buffer(
                mesocluster_size,
                &data_to_cluster,
                &mut mesocluster_buffer,
                &mut assignments_indirection,
                &mesocluster_indices_flat
                    [mesocluster_offsets[k]..mesocluster_offsets[k] + mesocluster_size],
                &immutable_data_norms,
            );

            // Seed centroids by sampling fine ones from this mesocluster.
            self.base.n_clusters = n_fine;
            self.base.generate_centroids(
                &mesocluster_buffer[..mesocluster_size * d],
                mesocluster_size,
                false,
            );
            self.base.n_clusters = n_clusters;
            self.base.prev_centroids[..n_fine * d]
                .copy_from_slice(&self.base.horizontal_centroids[..n_fine * d]);
            self.base.centroid_norms =
                compute_norms_row_major(&self.base.prev_centroids, n_fine, d, true);

            let fine_always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING
                || self.base.config.use_blas_only
                || n_fine <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
            let mut fine_partial_norms_computed = false;
            let mut fine_best_recall = 0.0;
            let mut fine_iwi: usize = 0;
            self.base.n_clusters = n_fine;
            for fine_iter_idx in 0..self.config.iters_fineclustering {
                let use_gemm_only = fine_iter_idx == 0 || fine_always_gemm_only;
                if !use_gemm_only && !fine_partial_norms_computed {
                    self.base.data_norms = compute_partial_norms_row_major(
                        &mesocluster_buffer[..mesocluster_size * d],
                        mesocluster_size,
                        d,
                        self.base.partial_d as usize,
                    );
                    fine_partial_norms_computed = true;
                }
                self.base.run_iteration(
                    &mesocluster_buffer[..mesocluster_size * d],
                    fine_iter_idx,
                    fine_iter_idx == 0,
                    use_gemm_only,
                    &mut not_pruned_counts[..mesocluster_size],
                );
                if let Some(stat) = self.base.iteration_stats.pop() {
                    self.iteration_stats.fineclustering.push(stat);
                }
                if self.base.config.early_termination
                    && self.base.should_stop_early(
                        false,
                        &mut fine_best_recall,
                        &mut fine_iwi,
                        fine_iter_idx,
                    )
                {
                    break;
                }
            }
            self.base.n_clusters = n_clusters;

            // Translate assignments back to global indices.
            self.translate_assignments(
                &assignments_indirection[..mesocluster_size],
                &mut final_assignments,
                &self.base.assignments[..mesocluster_size],
                fineclusters_offset as u32,
            );

            // Copy mesocluster-local centroids into the global slot.
            final_centroids[fineclusters_offset * d..(fineclusters_offset + n_fine) * d]
                .copy_from_slice(&self.base.horizontal_centroids[..n_fine * d]);

            fineclusters_offset += n_fine;
        }

        // ----- Phase 3: refinement -----
        if self.base.config.verbose {
            println!(
                "\n=== PHASE 3: REFINEMENT (fine-tuning all {} clusters) ===",
                n_clusters
            );
        }
        self.base.n_samples = immutable_data_norms.len();
        self.base.partial_d = (MIN_PARTIAL_D).max((self.base.vertical_d as u32) / 3);

        // Move final_centroids into horizontal_centroids/prev_centroids.
        self.base.horizontal_centroids[..n_clusters * d]
            .copy_from_slice(&final_centroids[..n_clusters * d]);
        self.base.prev_centroids[..n_clusters * d]
            .copy_from_slice(&final_centroids[..n_clusters * d]);
        self.base.assignments[..self.base.n_samples]
            .copy_from_slice(&final_assignments[..self.base.n_samples]);
        self.base.centroid_norms =
            compute_norms_row_major(&self.base.prev_centroids, n_clusters, d, true);

        let refinement_always_gemm_only =
            d < DIMENSION_THRESHOLD_FOR_PRUNING || n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        let mut refinement_partial_norms_computed = false;
        for refinement_iter_idx in 0..self.config.iters_refinement {
            if !refinement_always_gemm_only && !refinement_partial_norms_computed {
                self.base.data_norms = compute_partial_norms_row_major(
                    &data_to_cluster,
                    self.base.n_samples,
                    d,
                    self.base.partial_d as usize,
                );
                refinement_partial_norms_computed = true;
            }
            self.base.run_iteration(
                &data_to_cluster,
                refinement_iter_idx,
                false,
                refinement_always_gemm_only,
                &mut not_pruned_counts,
            );
            if let Some(stat) = self.base.iteration_stats.pop() {
                self.iteration_stats.refinement.push(stat);
            }
        }

        self.base.trained = true;
        self.base
            .get_output_centroids(self.base.config.unrotate_centroids)
    }

    pub fn assign(&self, vectors: &[f32], centroids: &[f32], n_vectors: usize) -> Vec<u32> {
        self.base.assign(vectors, centroids, n_vectors)
    }

    fn compute_n_to_sample(&self, n: usize) -> usize {
        if self.base.config.sampling_fraction == 1.0 {
            n
        } else {
            ((n as f64) * self.base.config.sampling_fraction as f64).floor() as usize
        }
    }

    fn arrange_fine_clusters(
        &self,
        n_clusters: usize,
        n_mesoclusters: usize,
        n_samples: usize,
        mesocluster_sizes: &[u32],
    ) -> Vec<usize> {
        let mut out = vec![0_usize; n_mesoclusters];
        let mut n_clusters_remaining = n_clusters;
        let mut n_nonempty_remaining = mesocluster_sizes.iter().filter(|&&s| s > 0).count();
        let mut n_samples_remaining = n_samples;
        for i in 0..n_mesoclusters {
            if i < n_mesoclusters - 1 {
                if mesocluster_sizes[i] == 0 {
                    out[i] = 0;
                } else {
                    n_nonempty_remaining -= 1;
                    let proportion = (n_clusters_remaining as f64 * mesocluster_sizes[i] as f64)
                        / n_samples_remaining as f64;
                    let mut allocated = proportion.round() as usize;
                    if n_clusters_remaining >= n_nonempty_remaining {
                        allocated = allocated.min(n_clusters_remaining - n_nonempty_remaining);
                    }
                    out[i] = allocated.max(1);
                }
            } else {
                out[i] = n_clusters_remaining;
            }
            n_clusters_remaining = n_clusters_remaining.saturating_sub(out[i]);
            n_samples_remaining = n_samples_remaining.saturating_sub(mesocluster_sizes[i] as usize);
        }
        out
    }

    fn compact_mesocluster_to_buffer(
        &mut self,
        mesocluster_size: usize,
        data: &[f32],
        buffer: &mut [f32],
        indirection: &mut [u32],
        meso_indices: &[usize],
        immutable_data_norms: &[f32],
    ) {
        let d = self.base.d;
        let dn = &mut self.base.data_norms;
        // `data_norms` is reassigned to a shorter vec whenever a mesocluster's
        // fine clustering recomputes partial norms (length = that mesocluster's
        // size). A subsequent, larger mesocluster would then overflow it here,
        // so ensure it can hold this mesocluster's per-point norms.
        if dn.len() < mesocluster_size {
            dn.resize(mesocluster_size, 0.0);
        }
        buffer
            .par_chunks_mut(d)
            .take(mesocluster_size)
            .zip(indirection.par_iter_mut().take(mesocluster_size))
            .enumerate()
            .for_each(|(j, (slot, ind))| {
                let i = meso_indices[j];
                *ind = i as u32;
                slot.copy_from_slice(&data[i * d..(i + 1) * d]);
            });
        for j in 0..mesocluster_size {
            let i = meso_indices[j];
            dn[j] = immutable_data_norms[i];
        }
    }

    fn translate_assignments(
        &self,
        indirection: &[u32],
        out: &mut [u32],
        input: &[u32],
        cluster_offset: u32,
    ) {
        for i in 0..indirection.len() {
            let orig = indirection[i] as usize;
            out[orig] = input[i] + cluster_offset;
        }
    }
}
