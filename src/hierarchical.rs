//! Hierarchical SuperKMeans.
//!
//! The sample is split into \(\lceil\sqrt{K}\rceil\) meso clusters, with
//! \(K = \lceil n / \texttt{max\_leaf\_size}\rceil\). Each meso cluster larger
//! than `max_leaf_size` is re-clustered into
//! \(\lceil n_i / \texttt{max\_leaf\_size}\rceil\) groups, and any group that
//! is still too large is split the same way. Training returns those fine
//! centroids. Balance comes from each split rebalancing its own undersized
//! clusters (`use_aggressive_split`).
//!
//! Splits permute rows in place, so peak memory is one copy of the training set.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use crate::adsampling::ADSamplingPruner;
use crate::common::HIERARCHICAL_PRUNER_INITIAL_THRESHOLD;
use crate::superkmeans::{SuperKMeans, SuperKMeansConfig, SuperKMeansIterationStats};
use crate::utils::squared_norms;

/// Config for hierarchical training: [`SuperKMeansConfig`] plus the split knobs.
#[derive(Clone, Debug)]
pub struct HierarchicalSuperKMeansConfig {
    /// Settings for the k-means run performed at each split.
    pub base: SuperKMeansConfig,
    /// Stop splitting once a cluster has at most this many points. The cluster
    /// count is emergent and lands near `n / max_leaf_size`.
    pub max_leaf_size: usize,
    /// Iterations for the initial \(\lceil\sqrt{K}\rceil\) meso split.
    pub iters_meso: u32,
    /// Iterations for every later split of a cluster that is still over
    /// `max_leaf_size`.
    pub iters_fine: u32,
}

impl Default for HierarchicalSuperKMeansConfig {
    fn default() -> Self {
        let mut base = SuperKMeansConfig::default();
        // Rebalancing undersized clusters at every split is what keeps the
        // fine clusters evenly sized.
        base.use_aggressive_split = true;
        Self {
            base,
            max_leaf_size: 256,
            iters_meso: 3,
            iters_fine: 5,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct HierarchicalSuperKMeansIterationStats {
    /// Flattened stats from every local k-means run during training.
    pub local_runs: Vec<SuperKMeansIterationStats>,
}

/// One cluster still waiting to be split: a contiguous row range of the
/// (permuted) training set.
struct Pending {
    offset: usize,
    len: usize,
}

/// Hierarchical k-means: a \(\lceil\sqrt{K}\rceil\) meso split, then repeated
/// splits of any cluster larger than
/// [`HierarchicalSuperKMeansConfig::max_leaf_size`].
///
/// [`Self::train`] returns the fine centroids. Their count is
/// [`SuperKMeans::n_clusters`] on [`Self::base`].
///
/// ```no_run
/// use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig};
///
/// let (n, d) = (10_000, 128);
/// let data = vec![0.0_f32; n * d];
///
/// let mut config = HierarchicalSuperKMeansConfig::default();
/// config.max_leaf_size = 64;
///
/// let mut kmeans = HierarchicalSuperKMeans::with_config(d, config);
/// let centroids = kmeans.train(&data, n);
/// assert_eq!(centroids.len(), kmeans.base.n_clusters * d);
/// ```
pub struct HierarchicalSuperKMeans {
    /// Model holding the fine centroids once [`Self::train`] has run.
    pub base: SuperKMeans,
    pub config: HierarchicalSuperKMeansConfig,
    /// Per-iteration statistics from every local k-means run.
    pub iteration_stats: HierarchicalSuperKMeansIterationStats,
    /// Shared by every per-split model to avoid repeating the O(d³)
    /// Householder QR per split.
    pruner: Arc<ADSamplingPruner>,
}

impl HierarchicalSuperKMeans {
    /// Build a trainer with [`HierarchicalSuperKMeansConfig::default`].
    pub fn new(dimensionality: usize) -> Self {
        Self::with_config(dimensionality, HierarchicalSuperKMeansConfig::default())
    }

    /// Build a trainer for `dimensionality`-dimensional vectors.
    ///
    /// # Panics
    ///
    /// Panics if `max_leaf_size`, `iters_meso`, or `iters_fine` is zero.
    pub fn with_config(dimensionality: usize, config: HierarchicalSuperKMeansConfig) -> Self {
        assert!(config.max_leaf_size >= 1, "max_leaf_size must be positive");
        assert!(config.iters_meso > 0, "iters_meso must be positive");
        assert!(config.iters_fine > 0, "iters_fine must be positive");

        let pruner = Arc::new(ADSamplingPruner::new(
            dimensionality,
            HIERARCHICAL_PRUNER_INITIAL_THRESHOLD,
            config.base.seed,
        ));

        // Placeholder until `train` replaces it with the fine-centroid model.
        let base = Self::new_superkmeans(
            2,
            dimensionality,
            &config.base,
            LocalSkmMode::Root,
            Arc::clone(&pruner),
        );

        Self {
            base,
            config,
            iteration_stats: HierarchicalSuperKMeansIterationStats::default(),
            pruner,
        }
    }

    /// Cluster `n` row-major vectors and return the row-major fine centroids.
    ///
    /// The cluster count is emergent; read it from [`Self::base`]'s
    /// `n_clusters`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero or the model has already been trained.
    pub fn train(&mut self, data: &[f32], n: usize) -> Vec<f32> {
        self.train_impl(Cow::Borrowed(data), n)
    }

    /// Like [`Self::train`], but takes ownership of `data` so the buffer can be
    /// rotated and split in place. Peak memory is then one copy of the training
    /// set rather than two.
    pub fn train_owned(&mut self, data: Vec<f32>, n: usize) -> Vec<f32> {
        self.train_impl(Cow::Owned(data), n)
    }

    fn train_impl(&mut self, data: Cow<'_, [f32]>, n: usize) -> Vec<f32> {
        assert!(n > 0, "n must be positive");
        assert!(!self.base.trained, "already trained");

        let d = self.base.d;
        self.iteration_stats = HierarchicalSuperKMeansIterationStats::default();

        let rotate = !self.config.base.data_already_rotated;
        // Only the owned path can rotate the caller's buffer in place; borrowing
        // forces a second full-size copy.
        let mut data_to_cluster = match data {
            Cow::Owned(mut owned) => {
                owned.truncate(n * d);
                if rotate {
                    self.pruner.rotate_in_place(&mut owned, n);
                }
                owned
            }
            Cow::Borrowed(borrowed) => {
                let len = n * d;
                if rotate {
                    let mut rotated = vec![0.0_f32; len];
                    self.pruner.rotate(&borrowed[..len], &mut rotated, n);
                    rotated
                } else {
                    borrowed[..len].to_vec()
                }
            }
        };

        let mut norms = squared_norms(&data_to_cluster, n, d);
        let mut scratch = PartitionScratch::new(n, d);
        let (centroids, sizes) = self.cluster(&mut data_to_cluster, &mut norms, &mut scratch);

        let n_clusters = sizes.len();
        assert!(n_clusters > 0, "clustering produced no clusters");

        let mut base = self.new_superkmeans_from_config(n_clusters, LocalSkmMode::Root);
        base.n_samples = n;
        base.horizontal_centroids = centroids;
        base.cluster_sizes = sizes;

        if base.config.verbose {
            println!(
                "HierarchicalSuperKMeans: n_clusters={}, max_leaf_size={}",
                n_clusters, self.config.max_leaf_size
            );
        }

        base.trained = true;
        let centroids = base.get_output_centroids(base.config.unrotate_centroids);
        self.base = base;
        centroids
    }

    /// Assign each of `n_vectors` vectors to its nearest centroid, returning one
    /// index per vector.
    ///
    /// Pass the centroids returned by [`Self::train`] to get fine-cluster indices.
    pub fn assign(&self, vectors: &[f32], centroids: &[f32], n_vectors: usize) -> Vec<u32> {
        self.base.assign(vectors, centroids, n_vectors)
    }

    /// Meso split, then re-cluster every group that is still over `max_leaf_size`.
    ///
    /// Returns the fine centroids (rotated) and one size per cluster. Sizes sum
    /// to the sample length, and each size is at most `max_leaf_size`.
    fn cluster(
        &mut self,
        data: &mut [f32],
        norms: &mut [f32],
        scratch: &mut PartitionScratch,
    ) -> (Vec<f32>, Vec<u32>) {
        let d = self.base.d;
        let n = norms.len();
        assert!(n > 0, "cluster requires a non-empty sample");
        debug_assert_eq!(data.len(), n * d);

        let mut centroids = Vec::new();
        let mut sizes = Vec::new();

        let k_meso = self.meso_clusters(n);
        if k_meso < 2 {
            centroids.extend(mean_rows(data, n, d));
            sizes.push(n as u32);
            return (centroids, sizes);
        }

        let mut queue = VecDeque::new();
        self.split_range(
            data,
            norms,
            0..n,
            k_meso,
            self.config.iters_meso,
            scratch,
            &mut queue,
            &mut centroids,
            &mut sizes,
        );

        while let Some(Pending { offset, len }) = queue.pop_front() {
            let k = self.fine_clusters(len);
            if k < 2 {
                let start = offset * d;
                centroids.extend(mean_rows(&data[start..start + len * d], len, d));
                sizes.push(len as u32);
                continue;
            }
            self.split_range(
                data,
                norms,
                offset..offset + len,
                k,
                self.config.iters_fine,
                scratch,
                &mut queue,
                &mut centroids,
                &mut sizes,
            );
        }

        (centroids, sizes)
    }

    /// Cluster `range` into `k` groups. Groups that fit under `max_leaf_size`
    /// are emitted; the rest go back on `queue`.
    fn split_range(
        &mut self,
        data: &mut [f32],
        norms: &mut [f32],
        range: Range<usize>,
        k: usize,
        iters: u32,
        scratch: &mut PartitionScratch,
        queue: &mut VecDeque<Pending>,
        centroids: &mut Vec<f32>,
        sizes: &mut Vec<u32>,
    ) {
        let d = self.base.d;
        let (split_centroids, assignments) = self.local_kmeans(
            &data[range.start * d..range.end * d],
            &norms[range.start..range.end],
            k,
            iters,
        );
        let (offsets, repaired) =
            self.partition_and_repair(data, norms, range.clone(), &assignments, k, scratch);

        for (cluster, bounds) in offsets.windows(2).enumerate() {
            let (start, end) = (bounds[0], bounds[1]);
            if start == end {
                continue;
            }
            let child_offset = range.start + start;
            let child_len = end - start;
            if child_len > self.config.max_leaf_size {
                queue.push_back(Pending {
                    offset: child_offset,
                    len: child_len,
                });
                continue;
            }
            if repaired {
                let start = child_offset * d;
                centroids.extend(mean_rows(&data[start..start + child_len * d], child_len, d));
            } else {
                centroids.extend_from_slice(&split_centroids[cluster * d..(cluster + 1) * d]);
            }
            sizes.push(child_len as u32);
        }
    }

    /// How many meso clusters a sample of `n` points should start with.
    ///
    /// \(\lceil\sqrt{K}\rceil\) with \(K = \lceil n / \texttt{max\_leaf\_size}\rceil\).
    /// Returns `0` when the sample already fits in one cluster.
    fn meso_clusters(&self, n: usize) -> usize {
        let target = n.div_ceil(self.config.max_leaf_size);
        if target < 2 {
            return 0;
        }
        let meso = (target as f64).sqrt().ceil() as usize;
        meso.clamp(2, target)
    }

    /// How many clusters a later split of `n_sub` points should request.
    ///
    /// \(\lceil n_i / \texttt{max\_leaf\_size}\rceil\). Returns `0` when the
    /// subset already fits in one cluster.
    fn fine_clusters(&self, n_sub: usize) -> usize {
        let k = n_sub.div_ceil(self.config.max_leaf_size);
        if k < 2 { 0 } else { k }
    }

    /// Partition `range` by `assignments`. If the clustering collapsed to fewer
    /// than two non-empty groups, bisect instead and report `repaired = true`.
    fn partition_and_repair(
        &self,
        data: &mut [f32],
        norms: &mut [f32],
        range: Range<usize>,
        assignments: &[u32],
        k: usize,
        scratch: &mut PartitionScratch,
    ) -> (Vec<usize>, bool) {
        let n_sub = range.end - range.start;
        let d = self.base.d;
        let mut offsets = {
            let mut subset = Subset {
                data: &mut data[range.start * d..range.end * d],
                norms: &mut norms[range.start..range.end],
            };
            subset.partition(assignments, k, scratch)
        };

        // The clustering collapsed onto one cluster; bisect so the
        // max_leaf_size invariant stays reachable.
        if offsets.windows(2).filter(|b| b[1] > b[0]).count() < 2 {
            offsets = vec![0, (n_sub / 2).clamp(1, n_sub - 1), n_sub];
            return (offsets, true);
        }
        (offsets, false)
    }

    fn new_superkmeans_from_config(&self, n_clusters: usize, mode: LocalSkmMode) -> SuperKMeans {
        Self::new_superkmeans(
            n_clusters,
            self.base.d,
            &self.config.base,
            mode,
            Arc::clone(&self.pruner),
        )
    }

    fn new_superkmeans(
        n_clusters: usize,
        dimensionality: usize,
        base_config: &SuperKMeansConfig,
        mode: LocalSkmMode,
        pruner: Arc<ADSamplingPruner>,
    ) -> SuperKMeans {
        let mut cfg = base_config.clone();
        match mode {
            LocalSkmMode::Root => {}
            LocalSkmMode::Split { iters } => {
                // Subsets are already in the rotated domain from the root pass.
                cfg.data_already_rotated = true;
                cfg.iters = iters;
            }
        }
        SuperKMeans::with_shared_pruner(n_clusters, dimensionality, cfg, pruner)
    }

    /// Cluster the `n × d` rows of `subset` into `k` groups, returning the
    /// centroids and one cluster index per row.
    ///
    /// `norms` holds the rows' squared norms, computed once up front; splits
    /// only reorder rows, so they stay valid. Each call runs a fresh
    /// [`SuperKMeans`] through [`SuperKMeans::run_core_loop`].
    fn local_kmeans(
        &mut self,
        subset: &[f32],
        norms: &[f32],
        k: usize,
        iters: u32,
    ) -> (Vec<f32>, Vec<u32>) {
        let d = self.base.d;
        let n = norms.len();

        let mut skm = self.new_superkmeans_from_config(k, LocalSkmMode::Split { iters });
        skm.n_samples = n;
        skm.horizontal_centroids = vec![0.0_f32; k * d];
        skm.prev_centroids = vec![0.0_f32; k * d];
        skm.cluster_sizes = vec![0_u32; k];
        skm.assignments = vec![0_u32; n];
        skm.distances = vec![0.0_f32; n];

        // Forgy init (data already in the rotated domain).
        skm.generate_centroids(subset, n, false);
        skm.prev_centroids
            .copy_from_slice(&skm.horizontal_centroids);

        // Full-width norms; the loop narrows them to `d'` once pruning engages.
        skm.data_norms = norms.to_vec();
        skm.centroid_norms = squared_norms(&skm.prev_centroids, k, d);

        skm.run_core_loop(subset, iters);

        self.iteration_stats
            .local_runs
            .append(&mut skm.iteration_stats);
        (skm.horizontal_centroids, skm.assignments)
    }
}

/// Mean of `n` contiguous row-major vectors of length `d`.
fn mean_rows(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), n * d);
    let mut centroid = vec![0.0_f32; d];
    for row in data.chunks_exact(d) {
        for (acc, v) in centroid.iter_mut().zip(row) {
            *acc += v;
        }
    }
    let inv = 1.0 / n as f32;
    for v in &mut centroid {
        *v *= inv;
    }
    centroid
}

/// The contiguous block of training rows one split owns.
///
/// `data` and `norms` are parallel — row `i` of `data` has norm `norms[i]` —
/// and [`Subset::partition`] permutes them together so a split can hand each
/// child a contiguous sub-range.
struct Subset<'a> {
    /// Row-major vectors, `n * d` floats.
    data: &'a mut [f32],
    /// Squared L2 norm of each row.
    norms: &'a mut [f32],
}

impl Subset<'_> {
    /// Number of rows. Never zero — a split always owns at least one point.
    fn len(&self) -> usize {
        self.norms.len()
    }

    fn dim(&self) -> usize {
        self.data.len() / self.len()
    }

    /// Group rows by their entry in `assignments`, in place, returning the
    /// `k + 1` offsets that delimit the clusters.
    ///
    /// Every row moves at most once: the assignment is inverted into a
    /// permutation, which is applied by rotating each of its cycles through a
    /// single held-out row. Rows keep their relative order within a cluster,
    /// which is what makes the clustering reproducible.
    fn partition(
        &mut self,
        assignments: &[u32],
        k: usize,
        scratch: &mut PartitionScratch,
    ) -> Vec<usize> {
        let (n, d) = (self.len(), self.dim());
        debug_assert_eq!(assignments.len(), n);

        let mut offsets = vec![0_usize; k + 1];
        for &c in assignments {
            offsets[c as usize + 1] += 1;
        }
        for c in 0..k {
            offsets[c + 1] += offsets[c];
        }

        let cursor = &mut scratch.cursor;
        cursor.clear();
        cursor.extend(offsets[..k].iter().map(|&o| o as u32));
        let dest = &mut scratch.dest[..n];
        for (row, &c) in assignments.iter().enumerate() {
            dest[row] = cursor[c as usize];
            cursor[c as usize] += 1;
        }

        let visited = &mut scratch.visited[..n];
        visited.fill(false);
        let held = &mut scratch.row[..d];
        for start in 0..n {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            if dest[start] as usize == start {
                continue;
            }

            held.copy_from_slice(&self.data[start * d..(start + 1) * d]);
            let mut held_norm = self.norms[start];
            // Walk the cycle, leaving each row at its destination and picking up
            // whatever it displaced, until the chain closes back on `start`.
            let mut cur = start;
            loop {
                let next = dest[cur] as usize;
                if next == start {
                    break;
                }
                visited[next] = true;
                held.swap_with_slice(&mut self.data[next * d..(next + 1) * d]);
                std::mem::swap(&mut held_norm, &mut self.norms[next]);
                cur = next;
            }
            self.data[start * d..(start + 1) * d].copy_from_slice(held);
            self.norms[start] = held_norm;
        }

        offsets
    }
}

/// Buffers shared by every in-place partition.
///
/// One set sized for the full sample suffices: only one partition runs at a time.
struct PartitionScratch {
    /// Destination row of each row, i.e. the permutation being applied.
    dest: Vec<u32>,
    /// Next free row of each cluster while inverting the assignment.
    cursor: Vec<u32>,
    /// Cycle bookkeeping for the permutation.
    visited: Vec<bool>,
    /// The row held out while a cycle rotates.
    row: Vec<f32>,
}

impl PartitionScratch {
    fn new(n: usize, d: usize) -> Self {
        Self {
            dest: vec![0_u32; n],
            cursor: Vec::new(),
            visited: vec![false; n],
            row: vec![0.0_f32; d],
        }
    }
}

/// How to configure a SuperKMeans instance used by hierarchical training.
enum LocalSkmMode {
    /// Placeholder / final model (respects the caller's rotation flags).
    Root,
    /// Per-split clustering over an already-rotated contiguous subset.
    Split { iters: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::make_blobs;

    /// Default config with only the size cap overridden.
    fn config(max_leaf_size: usize) -> HierarchicalSuperKMeansConfig {
        HierarchicalSuperKMeansConfig {
            max_leaf_size,
            ..Default::default()
        }
    }

    fn train(n: usize, d: usize, cfg: HierarchicalSuperKMeansConfig) -> HierarchicalSuperKMeans {
        let data = make_blobs(n, d, 8, true, 1.0, 5.0, 42);
        let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);
        let _centroids = kmeans.train(&data, n);
        kmeans
    }

    #[test]
    fn meso_split_is_sqrt_of_the_target_leaf_count() {
        let kmeans = HierarchicalSuperKMeans::with_config(4, config(100));
        // K = ceil(10_000 / 100) = 100, sqrt = 10.
        assert_eq!(kmeans.meso_clusters(10_000), 10);
        // A meso of 2_000 points asks for ceil(2_000 / 100) = 20 fine clusters.
        assert_eq!(kmeans.fine_clusters(2_000), 20);
        assert_eq!(kmeans.meso_clusters(100), 0);
        assert_eq!(kmeans.fine_clusters(100), 0);
    }

    #[test]
    fn cluster_sizes_cover_the_sample_and_respect_the_cap() {
        let mut cfg = config(32);
        cfg.iters_meso = 3;
        cfg.iters_fine = 4;
        cfg.base.early_termination = false;

        let n = 500usize;
        let kmeans = train(n, 16, cfg);

        assert!(kmeans.base.n_clusters >= 2);
        assert_eq!(kmeans.base.cluster_sizes.len(), kmeans.base.n_clusters);
        let total: u32 = kmeans.base.cluster_sizes.iter().sum();
        assert_eq!(total as usize, n);
        assert!(kmeans.base.cluster_sizes.iter().all(|&s| s > 0 && s <= 32));
    }

    /// The fine pass has to run: stopping after the meso split would leave
    /// clusters far above the cap, and the cluster count well below
    /// `n / max_leaf_size`.
    #[test]
    fn fine_pass_runs_on_oversized_meso_clusters() {
        let (n, max_leaf_size) = (10_000usize, 100usize);
        let kmeans = train(n, 16, config(max_leaf_size));
        let target = n.div_ceil(max_leaf_size);

        assert!(
            kmeans.base.n_clusters >= target,
            "stopping after the meso split cannot cover {n} points with {} clusters",
            kmeans.base.n_clusters
        );
        assert!(
            kmeans
                .base
                .cluster_sizes
                .iter()
                .all(|&s| s as usize <= max_leaf_size)
        );
    }

    /// Evenly sized clusters have to fall out of the split loop rather than an
    /// explicit penalty. Coefficient of variation over cluster sizes measures
    /// it: 0 is perfectly even, 1.0 means the spread is as large as the mean.
    #[test]
    fn cluster_sizes_stay_balanced() {
        let (n, d, max_leaf_size) = (20_000usize, 32usize, 100usize);
        let data = make_blobs(n, d, 40, true, 1.0, 10.0, 7);

        let mut kmeans = HierarchicalSuperKMeans::with_config(d, config(max_leaf_size));
        let _ = kmeans.train(&data, n);

        let sizes: Vec<f64> = kmeans
            .base
            .cluster_sizes
            .iter()
            .map(|&s| s as f64)
            .collect();
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let var = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        let cv = var.sqrt() / mean;

        assert!(
            cv < 0.6,
            "cluster sizes too uneven: cv={cv:.3} over {} clusters (mean={mean:.1})",
            sizes.len()
        );
    }

    /// One iteration per split is a valid, if crude, setting.
    #[test]
    fn one_iteration_per_split_is_allowed() {
        let mut cfg = config(32);
        cfg.iters_meso = 1;
        cfg.iters_fine = 1;
        let kmeans = train(200, 8, cfg);
        assert!(kmeans.base.n_clusters >= 200 / 32);
    }

    /// Centroids have to be informative. Unit-norm points average a squared
    /// distance near 1.0 from an uninformative centroid, so nearest-centroid
    /// distortion near that means the splits are partitioning by something
    /// other than distance — which is exactly what an ill-scaled balance
    /// penalty did.
    #[test]
    fn centroids_carry_information() {
        let (n, d) = (2_000usize, 32usize);
        let data = make_blobs(n, d, 40, true, 1.0, 10.0, 3);

        let mut kmeans = HierarchicalSuperKMeans::with_config(d, config(25));
        let centroids = kmeans.train(&data, n);
        let assignments = kmeans.assign(&data, &centroids, n);

        let mut total = 0.0f64;
        for (i, &cluster) in assignments.iter().enumerate() {
            let c = &centroids[cluster as usize * d..(cluster as usize + 1) * d];
            let x = &data[i * d..(i + 1) * d];
            let sq: f32 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
            total += sq as f64;
        }
        let distortion = total / n as f64;

        assert!(
            distortion < 0.5,
            "distortion {distortion} means centroids carry no information"
        );
    }

    #[test]
    fn cluster_count_tracks_the_requested_granularity() {
        let (n, d, max_leaf_size) = (2_000usize, 16usize, 20usize);

        let kmeans = train(n, d, config(max_leaf_size));
        let ideal = n.div_ceil(max_leaf_size);

        assert!(
            kmeans.base.n_clusters >= ideal,
            "cannot cover {n} points with {} clusters of at most {max_leaf_size}",
            kmeans.base.n_clusters
        );
        assert!(
            kmeans.base.n_clusters <= ideal * 2,
            "cluster count {} overshoots the {ideal} implied by max_leaf_size",
            kmeans.base.n_clusters
        );
    }

    #[test]
    fn sample_at_the_cap_stays_a_single_cluster() {
        let kmeans = train(64, 8, config(64));
        assert_eq!(kmeans.base.n_clusters, 1);
        assert_eq!(kmeans.base.cluster_sizes, vec![64]);
    }

    #[test]
    fn assign_maps_vectors_onto_fine_centroids() {
        let (n, d) = (400usize, 16usize);
        let data = make_blobs(n, d, 8, true, 1.0, 5.0, 42);
        let mut kmeans = HierarchicalSuperKMeans::with_config(d, config(32));
        let centroids = kmeans.train(&data, n);

        let assignments = kmeans.assign(&data, &centroids, n);
        assert_eq!(assignments.len(), n);
        assert!(
            assignments
                .iter()
                .all(|&a| (a as usize) < kmeans.base.n_clusters),
            "assignment outside the cluster range"
        );
    }

    #[test]
    fn training_is_deterministic_for_a_fixed_seed() {
        let cfg = config(32);
        let (n, d) = (300usize, 16usize);
        let data = make_blobs(n, d, 6, true, 1.0, 5.0, 11);

        let mut first = HierarchicalSuperKMeans::with_config(d, cfg.clone());
        let first_centroids = first.train(&data, n);
        let mut second = HierarchicalSuperKMeans::with_config(d, cfg);
        let second_centroids = second.train(&data, n);

        assert_eq!(first.base.n_clusters, second.base.n_clusters);
        assert_eq!(first_centroids, second_centroids);
    }

    #[test]
    fn partition_groups_rows_and_keeps_the_arrays_in_lockstep() {
        let (n, d, k) = (9usize, 2usize, 3usize);
        // Row i is [i, i] with norm 2i², which ties data and norms together:
        // any row that lands with the wrong norm is visible.
        let mut data: Vec<f32> = (0..n).flat_map(|i| [i as f32, i as f32]).collect();
        let mut norms: Vec<f32> = (0..n).map(|i| 2.0 * (i * i) as f32).collect();
        let assignments: Vec<u32> = vec![2, 0, 1, 2, 0, 1, 2, 0, 1];

        let mut subset = Subset {
            data: &mut data,
            norms: &mut norms,
        };
        let mut scratch = PartitionScratch::new(n, d);
        let offsets = subset.partition(&assignments, k, &mut scratch);

        assert_eq!(offsets, vec![0, 3, 6, 9]);
        // Stable within-cluster order: cluster 0 ← rows 1,4,7; 1 ← 2,5,8; 2 ← 0,3,6.
        let expected_src = [1usize, 4, 7, 2, 5, 8, 0, 3, 6];
        for (row, &src) in expected_src.iter().enumerate() {
            assert_eq!(data[row * d..(row + 1) * d], [src as f32, src as f32]);
            assert_eq!(norms[row], 2.0 * (src * src) as f32);
        }
    }

    /// Rows move by rotating the permutation's cycles through a single held-out
    /// row, so a slip loses or duplicates training points rather than merely
    /// misordering them. Interleaved clusters make the cycles long, which is
    /// where such a bug shows up.
    #[test]
    fn partition_preserves_every_row_under_long_cycles() {
        let (n, d, k) = (257usize, 3usize, 5usize);
        let value = |row: usize, j: usize| (row * d + j) as f32 * 0.5;

        let mut data: Vec<f32> = (0..n)
            .flat_map(|i| (0..d).map(move |j| value(i, j)))
            .collect();
        let mut norms: Vec<f32> = (0..n)
            .map(|i| (0..d).map(|j| value(i, j) * value(i, j)).sum())
            .collect();
        let assignments: Vec<u32> = (0..n).map(|i| ((i * 7) % k) as u32).collect();

        let mut subset = Subset {
            data: &mut data,
            norms: &mut norms,
        };
        let mut scratch = PartitionScratch::new(n, d);
        let offsets = subset.partition(&assignments, k, &mut scratch);

        let mut seen = vec![false; n];
        for row in 0..n {
            // Recover the original row id from the encoded payload.
            let src = (data[row * d] / 0.5) as usize / d;
            assert!(!seen[src], "row {src} survived twice");
            seen[src] = true;

            for j in 0..d {
                assert_eq!(data[row * d + j], value(src, j));
            }
            let expected: f32 = (0..d).map(|j| value(src, j) * value(src, j)).sum();
            assert_eq!(norms[row], expected);

            let c = assignments[src] as usize;
            assert!(
                (offsets[c]..offsets[c + 1]).contains(&row),
                "row {src} landed outside cluster {c}"
            );
        }
        assert!(seen.iter().all(|&s| s), "the partition dropped rows");
    }

    /// Building an `ADSamplingPruner` is an O(d³) Householder QR, so every
    /// model created during training must adopt the shared rotation matrix
    /// rather than computing its own.
    #[test]
    fn training_reuses_one_rotation_matrix() {
        let mut cfg = config(16);
        cfg.iters_meso = 2;
        cfg.iters_fine = 2;

        let kmeans = train(256, 8, cfg);
        assert!(
            Arc::ptr_eq(&kmeans.pruner, &kmeans.base.pruner),
            "fine model rebuilt its own pruner instead of sharing one"
        );
    }

    #[test]
    fn centroids_match_n_clusters() {
        let mut cfg = config(64);
        cfg.iters_meso = 3;
        cfg.iters_fine = 3;
        cfg.base.unrotate_centroids = true;

        let n = 512usize;
        let d = 32usize;
        let data = make_blobs(n, d, 10, true, 1.0, 5.0, 1);
        let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);
        let centroids = kmeans.train(&data, n);
        assert_eq!(centroids.len(), kmeans.base.n_clusters * d);
        assert!(kmeans.base.n_clusters >= n.div_ceil(64));
    }
}
