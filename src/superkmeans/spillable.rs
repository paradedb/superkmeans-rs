use std::io::{self, Read, Seek, SeekFrom, Write};

use super::*;
use crate::spill::{
    SpillOptions, TempStorage, TryDataset, invalid, row_offset, spool, visit_partition,
};

impl SuperKMeans {
    /// Train using bounded work buffers and caller-provided temporary storage.
    /// The source is consumed once. Per-row assignments are not retained in RAM.
    pub fn train_spillable<D: TryDataset + ?Sized, S: TempStorage>(
        &mut self,
        data: &mut D,
        storage: &mut S,
        options: SpillOptions,
    ) -> io::Result<Vec<f32>> {
        if self.trained {
            return Err(invalid("the clustering has already been trained"));
        }
        options.batch_rows(self.d, self.n_clusters, storage)?;
        let mut source = spool(
            data,
            self.d,
            storage,
            options,
            &self.pruner,
            !self.config.data_already_rotated,
        )?;
        self.train_spillable_partition(&mut source.file, 0, source.n, storage, options)?;
        Ok(self.finish_train())
    }

    pub(crate) fn train_spillable_partition<S: TempStorage>(
        &mut self,
        source: &mut S::File,
        start: usize,
        n: usize,
        storage: &mut S,
        options: SpillOptions,
    ) -> io::Result<(S::File, Vec<usize>)> {
        let (d, k) = (self.d, self.n_clusters);
        if n < k || n > u32::MAX as usize {
            return Err(invalid(
                "training row count must be at least k and at most u32::MAX",
            ));
        }
        let batch_rows = options.batch_rows(d, k, storage)?;
        self.n_samples = n;
        self.iteration_stats.clear();
        self.horizontal_centroids = vec![0.0; k * d];
        self.prev_centroids = vec![0.0; k * d];
        self.cluster_sizes = vec![0; k];
        self.assignments = Vec::new();
        self.distances = Vec::new();
        self.data_norms = Vec::new();
        self.partial_d = self.initial_partial_d();
        self.cost = 0.0;
        self.prev_cost = 0.0;

        let mut seen = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        visit_partition(source, start, n, d, batch_rows, |batch| {
            for row in batch.rows() {
                let slot = if seen < k {
                    seen
                } else {
                    rng.gen_range(0..=seen)
                };
                if slot < k {
                    self.horizontal_centroids[slot * d..(slot + 1) * d].copy_from_slice(row);
                }
                seen += 1;
            }
            Ok(())
        })?;
        self.prev_centroids
            .copy_from_slice(&self.horizontal_centroids);

        let mut labels = storage.create()?;
        let mut gemm_buf = Vec::new();
        let mut assignments = vec![0_u32; n.min(batch_rows)];
        let mut distances = vec![0.0_f32; assignments.len()];
        let mut not_pruned_counts = vec![0_usize; assignments.len()];
        let mut counts = vec![0; k];
        let always_gemm = d < DIMENSION_THRESHOLD_FOR_PRUNING
            || self.config.use_blas_only
            || k <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        let mut best_recall = 0.0;
        let mut iters_without_improvement = 0;

        for iter in 0..self.config.iters {
            let gemm_only = iter == 0 || always_gemm;
            if iter > 0 {
                std::mem::swap(&mut self.horizontal_centroids, &mut self.prev_centroids);
            }
            self.horizontal_centroids.fill(0.0);
            self.cluster_sizes.fill(0);
            self.centroid_norms = if gemm_only {
                squared_norms(&self.prev_centroids, k, d)
            } else {
                squared_norms_partial(&self.prev_centroids, k, d, self.partial_d as usize)
            };
            labels.rewind()?;
            let mut offset = 0;
            let mut cost = 0.0_f32;
            let mut not_pruned_sum = 0.0_f64;
            visit_partition(source, start, n, d, batch_rows, |batch| {
                let rows = batch.n();
                let assign = &mut assignments[..rows];
                let dist = &mut distances[..rows];
                if iter > 0 {
                    labels.read_exact(bytemuck::cast_slice_mut(assign))?;
                }
                let norms = if gemm_only {
                    squared_norms(batch.as_slice(), rows, d)
                } else {
                    squared_norms_partial(batch.as_slice(), rows, d, self.partial_d as usize)
                };
                if gemm_only {
                    batch::find_nearest_neighbor(
                        batch.as_slice(),
                        &self.prev_centroids,
                        rows,
                        k,
                        d,
                        &norms,
                        &self.centroid_norms,
                        assign,
                        dist,
                        &mut gemm_buf,
                    );
                } else {
                    not_pruned_counts[..rows].fill(0);
                    batch::find_nearest_neighbor_with_pruning(
                        batch.as_slice(),
                        &self.prev_centroids,
                        rows,
                        k,
                        d,
                        self.vertical_d,
                        self.horizontal_d,
                        &norms,
                        &self.centroid_norms,
                        assign,
                        dist,
                        &self.pruner,
                        self.partial_d as usize,
                        &mut not_pruned_counts[..rows],
                        &mut gemm_buf,
                    );
                    not_pruned_sum += not_pruned_counts[..rows]
                        .iter()
                        .map(|&v| v as f64)
                        .sum::<f64>();
                }
                accumulate_rows_by_assignment(
                    batch.as_slice(),
                    assign,
                    &mut self.horizontal_centroids,
                    &mut self.cluster_sizes,
                    d,
                );
                for &distance in dist.iter() {
                    cost += distance;
                }
                labels.seek(SeekFrom::Start(row_offset(offset, 1)?))?;
                labels.write_all(bytemuck::cast_slice(assign))?;
                offset += rows;
                Ok(())
            })?;
            for (out, &size) in counts.iter_mut().zip(&self.cluster_sizes) {
                *out = size as usize;
            }
            let old_partial_d = self.partial_d;
            let avg = if gemm_only {
                -1.0
            } else {
                self.apply_partial_d_from_avg((not_pruned_sum / (n as f64 * k as f64)) as f32)
            };
            self.consolidate_centroids(n, k);
            self.prev_cost = self.cost;
            self.cost = cost;
            self.shift = centroid_shift(&self.horizontal_centroids, &self.prev_centroids, k, d);
            self.push_iteration_stats(iter, gemm_only, avg, old_partial_d);
            if self.config.early_termination
                && self.should_stop_early(
                    false,
                    &mut best_recall,
                    &mut iters_without_improvement,
                    iter,
                )
            {
                break;
            }
        }
        labels.rewind()?;
        Ok((labels, counts))
    }
}
