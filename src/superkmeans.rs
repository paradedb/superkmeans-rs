//! Core SuperKMeans algorithm: BLAS+pruning k-means.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Uniform};

use crate::adsampling::ADSamplingPruner;
use crate::batch;
use crate::common::{
    CENTROID_PERTURBATION_EPS, DIMENSION_THRESHOLD_FOR_PRUNING, MIN_PARTIAL_D,
    N_CLUSTERS_THRESHOLD_FOR_PRUNING, PRUNER_INITIAL_THRESHOLD, X_BATCH_SIZE,
};
use crate::dataset::{self, Contiguous, Dataset};
use crate::layout;
use crate::matrix::Matrix;
use crate::utils::{
    accumulate_rows_by_assignment, centroid_shift, mean_rows_by_count, normalize_rows_l2,
    squared_norms, squared_norms_partial, sum_rows_by_assignment,
};

/// Configuration parameters for SuperKMeans clustering.
///
/// There is no sampling knob: [`SuperKMeans::train`] clusters exactly the rows
/// it is handed. Subsample before calling if you want to train on less data.
#[derive(Clone, Debug)]
pub struct SuperKMeansConfig {
    pub iters: u32,
    pub n_threads: u32,
    pub seed: u64,
    pub use_blas_only: bool,
    pub tol: f32,
    pub early_termination: bool,
    pub min_not_pruned_pct: f32,
    pub max_not_pruned_pct: f32,
    pub adjustment_factor_for_partial_d: f32,
    pub unrotate_centroids: bool,
    pub verbose: bool,
    pub angular: bool,
    pub suppress_warnings: bool,
    pub data_already_rotated: bool,
    /// Revive undersized clusters as well as empty ones.
    ///
    /// An empty cluster is always revived: it takes a slightly perturbed copy
    /// of a donor cluster's centroid, with donors drawn without replacement in
    /// proportion to their size. With this set, clusters holding fewer than a
    /// quarter of the mean cluster size are revived the same way on every pass
    /// except the last. Hierarchical training opts into this.
    pub use_aggressive_split: bool,
}

impl Default for SuperKMeansConfig {
    fn default() -> Self {
        Self {
            iters: 10,
            n_threads: 0,
            seed: 42,
            use_blas_only: false,
            tol: 1.0e-4,
            early_termination: true,
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

/// ADSampling pruning statistics for an iteration that took the pruned path.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PruningStats {
    /// Active ADSampling prefix dimensionality used during the E-step.
    pub partial_d: u32,
    /// Fraction of sample-centroid pairs that survived pruning, in `[0, 1]`
    /// (not a percentage, despite the name).
    pub not_pruned_pct: f32,
}

/// Statistics for one training pass: an assignment step followed by a
/// centroid update.
///
/// [`SuperKMeansResult::iteration_stats`] holds one entry per pass, in order.
#[derive(Clone, Debug, Default)]
pub struct SuperKMeansIterationStats {
    /// 1-based number of this pass within its training run.
    pub iteration: usize,
    /// Number of rows the run was trained on.
    pub n_samples: usize,
    /// Number of clusters the run was asked for.
    pub n_clusters: usize,
    /// Sum of squared distances from each sample to the centroid it was assigned
    /// to in this pass's E-step (the previous pass's centroids).
    pub sse: f64,
    /// Mean squared distance (`sse / n_samples`).
    pub mse: f64,
    /// Total squared displacement of centroids from the previous iteration.
    pub shift: f32,
    /// Number of strictly empty clusters (`size == 0`) revived by splitting a donor.
    pub empty_revived: usize,
    /// Number of non-empty clusters below a quarter of the mean cluster size
    /// revived by splitting a donor. Always zero when
    /// [`SuperKMeansConfig::use_aggressive_split`] is off, and on the last
    /// configured pass.
    pub small_revived: usize,
    /// ADSampling pruning statistics (`None` on exact GEMM passes).
    pub pruning: Option<PruningStats>,
}

impl SuperKMeansIterationStats {
    /// Total number of clusters revived by splitting a donor this iteration.
    pub fn revived(&self) -> usize {
        self.empty_revived + self.small_revived
    }

    /// Whether this iteration used the exact unpruned GEMM path.
    pub fn is_gemm_only(&self) -> bool {
        self.pruning.is_none()
    }
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

/// A reusable SuperKMeans trainer for `d`-dimensional vectors.
///
/// The model owns what depends only on `d` and the config: the ADSampling
/// rotation, an O(d³) QR on first use, and the pruning layout. Training takes
/// `&self` and returns a [`SuperKMeansResult`], so one model clusters any
/// number of datasets and cluster counts, from several threads at once.
pub struct SuperKMeans {
    d: usize,
    config: SuperKMeansConfig,
    pruner: ADSamplingPruner,
    vertical_d: usize,
    horizontal_d: usize,
}

/// What one [`SuperKMeans::train`] call produced.
#[derive(Clone, Debug)]
pub struct SuperKMeansResult {
    /// Row-major centroids, `n_clusters × d`.
    pub centroids: Vec<f32>,
    /// The cluster each training row landed on in the last pass, in input order.
    pub assignments: Vec<u32>,
    /// Rows per cluster after the last pass. A revived cluster takes half of
    /// its donor's count.
    pub cluster_sizes: Vec<u32>,
    /// One entry per pass, in order.
    pub iteration_stats: Vec<SuperKMeansIterationStats>,
    /// The centroids in the pruning domain, when [`Self::centroids`] left it.
    rotated_centroids: Option<Vec<f32>>,
}

impl SuperKMeansResult {
    fn rotated_centroids(&self) -> &[f32] {
        self.rotated_centroids.as_deref().unwrap_or(&self.centroids)
    }
}

/// One Lloyd run: its inputs and the centroid state each pass updates.
///
/// Centroids live in the pruning domain throughout.
pub(crate) struct Run<'a> {
    skm: &'a SuperKMeans,
    k: usize,
    pub(crate) n: usize,
    iters: u32,

    /// This pass's centroids, row-major.
    pub(crate) horizontal_centroids: Vec<f32>,
    /// The centroids the current pass assigns against.
    prev_centroids: Vec<f32>,
    /// Norms of `prev_centroids`, full-width or `d'`-wide to match the pass.
    centroid_norms: Vec<f32>,
    pub(crate) cluster_sizes: Vec<u32>,
    pub(crate) assignments: Vec<u32>,

    partial_d: u32,
    /// 0-based index of the pass in progress, or of the last one run.
    iteration: u32,
    pub(crate) stats: Vec<SuperKMeansIterationStats>,

    /// SGEMM output scratch, grown once and reused across passes.
    gemm_buf: Vec<f32>,
}

/// Assign row-major vectors to their nearest centroids without constructing a trainer.
///
/// Vectors and centroids must share the same coordinate space. This uses exact,
/// unpruned nearest-neighbor search and does not build a rotation matrix.
///
/// # Panics
///
/// Panics if the two matrices have different dimensionality.
pub fn assign(vectors: Matrix<'_>, centroids: Matrix<'_>) -> Vec<u32> {
    assert_eq!(
        vectors.d(),
        centroids.d(),
        "vectors and centroids must share d"
    );
    let n_vectors = vectors.n();
    let n_centroids = centroids.n();
    let d = vectors.d();
    let mut result_assignments = vec![0_u32; n_vectors];
    let mut result_distances = vec![0.0_f32; n_vectors];
    let vector_norms = squared_norms(vectors.as_slice(), n_vectors, d);
    let centroid_norms = squared_norms(centroids.as_slice(), n_centroids, d);
    let mut buf = Vec::new();
    batch::find_nearest_neighbor(
        vectors.as_slice(),
        centroids.as_slice(),
        n_vectors,
        n_centroids,
        d,
        &vector_norms,
        &centroid_norms,
        &mut result_assignments,
        &mut result_distances,
        &mut buf,
    );
    result_assignments
}

/// Assign row-major vectors to their nearest centroid and to a SOAR secondary without
/// constructing a trainer: the secondary minimizes `‖x − c‖² + lambda · ⟨x − c, r̂1⟩²` over the
/// other centroids, `r̂1` being the unit residual to the nearest, so the two cells miss
/// different queries. The secondary is `u32::MAX` with a single centroid.
///
/// # Panics
///
/// Panics if the two matrices have different dimensionality.
pub fn assign_soar(
    vectors: Matrix<'_>,
    centroids: Matrix<'_>,
    lambda: f32,
) -> (Vec<u32>, Vec<u32>) {
    assign_soar_replicas(vectors, centroids, lambda, 2)
}

/// Assign to a primary and up to `replicas - 1` distinct SOAR secondaries per vector.
/// Each secondary minimizes distance plus `lambda` times the sum of squared projections
/// onto all preceding unit residuals. Secondaries are row-major; missing slots are `u32::MAX`.
///
/// # Panics
/// Panics if `replicas` is zero or the matrices have different dimensionality.
pub fn assign_soar_replicas(
    vectors: Matrix<'_>,
    centroids: Matrix<'_>,
    lambda: f32,
    replicas: usize,
) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(
        vectors.d(),
        centroids.d(),
        "vectors and centroids must share d"
    );
    assert!(replicas > 0);
    let (n_vectors, n_centroids, d) = (vectors.n(), centroids.n(), vectors.d());
    let mut primary = vec![0_u32; n_vectors];
    let mut secondary = vec![u32::MAX; n_vectors * (replicas - 1)];
    batch::find_soar_replica_assignments(
        vectors.as_slice(),
        centroids.as_slice(),
        n_vectors,
        n_centroids,
        d,
        &squared_norms(vectors.as_slice(), n_vectors, d),
        &squared_norms(centroids.as_slice(), n_centroids, d),
        lambda,
        replicas,
        &mut primary,
        &mut secondary,
        &mut Vec::new(),
    );
    (primary, secondary)
}

/// Primary and SOAR secondaries for any number of vector batches against one centroid set.
///
/// The primary is the nearest centroid; each secondary minimizes the cumulative cost of
/// [`assign_soar_replicas`] over the row's `candidates` nearest non-primary centroids, so it costs
/// `candidates · d` multiply-adds per row whatever the centroid count. With every centroid a
/// candidate, the output is [`assign_soar_replicas`]'s up to cost ties within float rounding.
pub struct SoarAssigner<'a> {
    centroids: Matrix<'a>,
    norms: Vec<f32>,
    lambda: f32,
    replicas: usize,
    candidates: usize,
}

impl<'a> SoarAssigner<'a> {
    /// An assigner for `centroids` with alignment weight `lambda` and `replicas` cells per
    /// vector (the primary included). `candidates` bounds the search for each secondary to that
    /// many nearest centroids; `candidates >= centroids.n() - 1` is the exhaustive rule.
    ///
    /// # Panics
    ///
    /// Panics if `replicas` is zero.
    pub fn with_candidates(
        centroids: Matrix<'a>,
        lambda: f32,
        replicas: usize,
        candidates: usize,
    ) -> Self {
        assert!(replicas > 0);
        let norms = squared_norms(centroids.as_slice(), centroids.n(), centroids.d());
        Self {
            centroids,
            norms,
            lambda,
            replicas,
            candidates,
        }
    }

    /// One primary per vector and `replicas - 1` row-major secondaries per vector: distinct,
    /// `u32::MAX` where fewer than `replicas - 1` candidates remain.
    ///
    /// # Panics
    ///
    /// Panics if `vectors.d()` differs from the centroids.
    pub fn assign(&self, vectors: Matrix<'_>) -> (Vec<u32>, Vec<u32>) {
        let d = self.centroids.d();
        assert_eq!(
            vectors.d(),
            d,
            "vectors dimensionality {} does not match the centroids ({d})",
            vectors.d()
        );
        let n_vectors = vectors.n();
        let n_centroids = self.centroids.n();
        if self.replicas < 2 || n_centroids < 2 {
            // No secondary is possible: the nearest centroid alone.
            return assign_soar_replicas(vectors, self.centroids, self.lambda, self.replicas);
        }
        let mut primary = vec![0_u32; n_vectors];
        let mut secondary = vec![u32::MAX; n_vectors * (self.replicas - 1)];
        batch::find_soar_replica_assignments_shortlist(
            vectors.as_slice(),
            self.centroids.as_slice(),
            n_vectors,
            n_centroids,
            d,
            &squared_norms(vectors.as_slice(), n_vectors, d),
            &self.norms,
            self.lambda,
            self.replicas,
            self.candidates,
            &mut primary,
            &mut secondary,
        );
        (primary, secondary)
    }
}

impl SuperKMeans {
    /// A model with [`SuperKMeansConfig::default`].
    pub fn new(dimensionality: usize) -> Self {
        Self::with_config(dimensionality, SuperKMeansConfig::default())
    }

    /// # Panics
    ///
    /// Panics if `dimensionality` or [`SuperKMeansConfig::iters`] is zero.
    pub fn with_config(dimensionality: usize, config: SuperKMeansConfig) -> Self {
        Self::with_pruning_threshold(dimensionality, config, PRUNER_INITIAL_THRESHOLD)
    }

    /// [`Self::with_config`] with the pruner's starting threshold picked by
    /// the caller.
    pub(crate) fn with_pruning_threshold(
        dimensionality: usize,
        mut config: SuperKMeansConfig,
        threshold: f32,
    ) -> Self {
        assert!(dimensionality > 0, "dimensionality must be positive");
        assert!(config.iters > 0, "iters must be positive");
        if config.data_already_rotated {
            config.unrotate_centroids = false;
        }
        let split = layout::get_dimension_split(dimensionality);
        Self {
            d: dimensionality,
            pruner: ADSamplingPruner::new(dimensionality, threshold, config.seed),
            config,
            vertical_d: split.vertical_d,
            horizontal_d: split.horizontal_d,
        }
    }

    pub fn d(&self) -> usize {
        self.d
    }

    /// The config as training applies it:
    /// [`SuperKMeansConfig::data_already_rotated`] turns
    /// [`SuperKMeansConfig::unrotate_centroids`] off.
    pub fn config(&self) -> &SuperKMeansConfig {
        &self.config
    }

    /// Cluster `data` into `n_clusters` groups.
    ///
    /// A [`Matrix`] or [`Vectors`](crate::Vectors) is trained in memory, and an
    /// owned [`Vectors`](crate::Vectors) is rotated in place. Any other
    /// [`Dataset`] is streamed on the calling thread.
    ///
    /// ```
    /// use superkmeans::{IterDataset, Matrix, SuperKMeans, make_blobs};
    ///
    /// let (n, d) = (2_000, 32);
    /// let data = make_blobs(n, d, 10, true, 1.0, 10.0, 42);
    /// let kmeans = SuperKMeans::new(d);
    ///
    /// let coarse = kmeans.train(&mut Matrix::new(&data, n, d), 10).unwrap();
    /// assert_eq!(coarse.centroids.len(), 10 * d);
    ///
    /// // The same model streams a different cluster count.
    /// let fine = kmeans
    ///     .train(&mut IterDataset::new(Matrix::new(&data, n, d).chunks(256)), 40)
    ///     .unwrap();
    /// assert_eq!(fine.assignments.len(), n);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the first error from [`Dataset::for_each_batch`]. [`Matrix`] and
    /// [`Vectors`](crate::Vectors) do not scan, so they return [`Ok`].
    ///
    /// # Panics
    ///
    /// Panics if `n_clusters` is zero, the sample has fewer rows than
    /// `n_clusters`, a batch's dimensionality differs from the model, or two
    /// streaming passes yield different row counts.
    pub fn train<D: Dataset + ?Sized>(
        &self,
        data: &mut D,
        n_clusters: usize,
    ) -> Result<SuperKMeansResult, D::Error> {
        assert!(n_clusters > 0, "n_clusters must be positive");
        let run = match data.contiguous() {
            Some(block) => self.train_block(block, n_clusters),
            None => self.train_stream(data, n_clusters, self.config.iters, self.rotates())?,
        };
        let rotated_centroids = self
            .config
            .unrotate_centroids
            .then(|| run.horizontal_centroids.clone());
        Ok(SuperKMeansResult {
            centroids: self.output_centroids(run.horizontal_centroids),
            assignments: run.assignments,
            cluster_sizes: run.cluster_sizes,
            iteration_stats: run.stats,
            rotated_centroids,
        })
    }

    /// Lloyd over one contiguous block, seeded with a Forgy sample of its rows.
    fn train_block(&self, block: Contiguous<'_>, k: usize) -> Run<'_> {
        let d = self.d;
        let sample = block.as_matrix();
        let n = sample.n();
        self.expect_d(sample.d());
        expect_enough_rows(n, k);

        if self.config.verbose {
            println!("Front dimensions (d') = {}", self.initial_partial_d());
            println!("Trailing dimensions (d'') = {}", d - self.vertical_d);
        }

        let seeds = forgy_sample(sample.as_slice(), n, k, d, self.config.seed);
        let seeds = self.to_pruning_domain(&seeds, k, self.rotates());
        self.with_rotated(block, |rows| {
            let mut run = Run::new(self, k, n, self.config.iters, seeds);
            run.fit_rows(rows, squared_norms(rows, n, d));
            run
        })
    }

    /// Lloyd over rows already in the pruning domain, seeded with a Forgy
    /// sample of them. `norms` holds the rows' full-width squared norms.
    pub(crate) fn train_rotated(
        &self,
        rows: &[f32],
        norms: &[f32],
        k: usize,
        iters: u32,
    ) -> Run<'_> {
        let n = norms.len();
        let seeds = forgy_sample(rows, n, k, self.d, self.config.seed);
        let mut run = Run::new(self, k, n, iters, seeds);
        run.fit_rows(rows, norms.to_vec());
        run
    }

    /// Lloyd over a scan, one pass of `data` per iteration, seeded with a
    /// reservoir sample. `rotate` moves each batch into the pruning domain.
    pub(crate) fn train_stream<D: Dataset + ?Sized>(
        &self,
        data: &mut D,
        k: usize,
        iters: u32,
        rotate: bool,
    ) -> Result<Run<'_>, D::Error> {
        let (n, sample) = self.reservoir_sample(data, k)?;
        expect_enough_rows(n, k);
        let seeds = self.to_pruning_domain(&sample, k, rotate);
        let mut run = Run::new(self, k, n, iters, seeds);
        run.fit_stream(data, rotate)?;
        Ok(run)
    }

    /// One pass: count the rows and reservoir-sample `k` of them
    /// (Algorithm R, seeded from [`SuperKMeansConfig::seed`]).
    fn reservoir_sample<D: Dataset + ?Sized>(
        &self,
        data: &mut D,
        k: usize,
    ) -> Result<(usize, Vec<f32>), D::Error> {
        let d = self.d;
        let mut reservoir = vec![0.0_f32; k * d];
        let mut n = 0usize;
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        data.for_each_batch(&mut |matrix| {
            self.expect_d(matrix.d());
            for row in matrix.rows() {
                if n < k {
                    reservoir[n * d..(n + 1) * d].copy_from_slice(row);
                } else {
                    let j = rng.gen_range(0..=n);
                    if j < k {
                        reservoir[j * d..(j + 1) * d].copy_from_slice(row);
                    }
                }
                n += 1;
            }
        })?;
        Ok((n, reservoir))
    }

    /// Whether training rotates its input into the pruning domain.
    pub(crate) fn rotates(&self) -> bool {
        !self.config.data_already_rotated
    }

    /// Whether a run with `n_clusters` clusters can leave the exact GEMM path.
    ///
    /// Pruning needs a wide vector, more clusters than the GEMM cutoff, and
    /// the BLAS-only switch off.
    pub(crate) fn can_prune(&self, n_clusters: usize) -> bool {
        self.d >= DIMENSION_THRESHOLD_FOR_PRUNING
            && !self.config.use_blas_only
            && n_clusters > N_CLUSTERS_THRESHOLD_FOR_PRUNING
    }

    pub(crate) fn expect_d(&self, d: usize) {
        assert_eq!(
            d, self.d,
            "matrix dimensionality {d} does not match the model ({})",
            self.d
        );
    }

    /// A copy of the `n` rows, rotated into the pruning domain when `rotate`
    /// is set.
    fn to_pruning_domain(&self, rows: &[f32], n: usize, rotate: bool) -> Vec<f32> {
        if !rotate {
            return rows.to_vec();
        }
        let mut rotated = vec![0.0_f32; rows.len()];
        self.pruner.rotate(rows, &mut rotated, n);
        rotated
    }

    /// Run `body` over the block's rows in the pruning domain.
    ///
    /// A borrowed block is copied. An owned block is rotated in place and
    /// keeps whatever `body` leaves in it.
    pub(crate) fn with_rotated<R>(
        &self,
        block: Contiguous<'_>,
        body: impl FnOnce(&mut [f32]) -> R,
    ) -> R {
        match block {
            Contiguous::Borrowed(matrix) => {
                body(&mut self.to_pruning_domain(matrix.as_slice(), matrix.n(), self.rotates()))
            }
            Contiguous::Owned(vectors) => {
                if self.rotates() {
                    let n = vectors.n();
                    self.pruner.rotate_in_place(&mut vectors.data, n);
                }
                body(&mut vectors.data)
            }
        }
    }

    /// Pruning-domain centroids as training returns them: unrotated when
    /// [`SuperKMeansConfig::unrotate_centroids`] is on.
    pub(crate) fn output_centroids(&self, rotated: Vec<f32>) -> Vec<f32> {
        if !self.config.unrotate_centroids {
            return rotated;
        }
        let mut out = vec![0.0_f32; rotated.len()];
        self.pruner
            .unrotate(&rotated, &mut out, rotated.len() / self.d);
        out
    }

    /// The starting `d'`: half the vertically-stored dimensions, floored at
    /// [`MIN_PARTIAL_D`] and capped at what is actually stored vertically.
    /// [`Run::adjust_partial_d`] widens or narrows it from here.
    fn initial_partial_d(&self) -> u32 {
        MIN_PARTIAL_D
            .max(self.vertical_d as u32 / 2)
            .min(self.vertical_d as u32)
    }

    /// Brute-force assignment: returns the index of the nearest centroid for
    /// each vector (data and centroids assumed to share the same domain).
    ///
    /// # Panics
    ///
    /// Panics if either matrix's dimensionality differs from this model.
    pub fn assign(&self, vectors: Matrix<'_>, centroids: Matrix<'_>) -> Vec<u32> {
        assert_eq!(
            vectors.d(),
            self.d,
            "vectors dimensionality {} does not match the model ({})",
            vectors.d(),
            self.d
        );
        assert_eq!(
            centroids.d(),
            self.d,
            "centroids dimensionality {} does not match the model ({})",
            centroids.d(),
            self.d
        );
        assign(vectors, centroids)
    }

    /// Brute-force assignment to the nearest centroid and a SOAR secondary, as
    /// [`assign_soar`]: one index pair per vector, the secondary `u32::MAX` with a
    /// single centroid.
    ///
    /// # Panics
    ///
    /// Panics if either matrix's dimensionality differs from this model.
    pub fn assign_soar(
        &self,
        vectors: Matrix<'_>,
        centroids: Matrix<'_>,
        lambda: f32,
    ) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(vectors.d(), self.d, "vectors dimensionality does not match");
        assert_eq!(
            centroids.d(),
            self.d,
            "centroids dimensionality does not match"
        );
        assign_soar(vectors, centroids, lambda)
    }

    /// Re-assign the training set to its nearest centroid using the pruning path.
    ///
    /// Each point's pruning threshold is seeded with the centroid it landed on
    /// during training, which is what makes pruning pay: starting from that
    /// tight bound rather than a full GEMM over the first centroid batch is
    /// worth roughly 1.8x in the paper's ablation.
    ///
    /// A *different* set of vectors has no such seed available, so it needs a
    /// cold-start strategy and is deliberately not handled here — use
    /// [`Self::assign`], which is exact but unpruned.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is not the set `trained` was trained on, or its
    /// dimensionality differs from this model.
    pub fn assign_training_points(
        &self,
        vectors: Matrix<'_>,
        trained: &SuperKMeansResult,
    ) -> Vec<u32> {
        let d = self.d;
        assert_eq!(vectors.d(), d, "vectors dimensionality does not match");
        let n_vectors = vectors.n();
        assert_eq!(
            n_vectors,
            trained.assignments.len(),
            "assign_training_points re-assigns the training set; pass the {} rows given to \
             train, or use assign() for a different set",
            trained.assignments.len()
        );
        let n_centroids = trained.cluster_sizes.len();

        if !self.can_prune(n_centroids) {
            if !self.config.suppress_warnings {
                eprintln!(
                    "WARNING: AssignTrainingPoints cannot use pruning, falling back to brute force"
                );
            }
            return assign(vectors, Matrix::new(&trained.centroids, n_centroids, d));
        }

        let partial_d = self.initial_partial_d() as usize;
        let rotated;
        let data = if self.rotates() {
            rotated = self.to_pruning_domain(vectors.as_slice(), n_vectors, true);
            &rotated
        } else {
            vectors.as_slice()
        };
        let centroids = trained.rotated_centroids();
        let centroid_partial_norms = squared_norms_partial(centroids, n_centroids, d, partial_d);
        let data_partial_norms = squared_norms_partial(data, n_vectors, d, partial_d);

        // Training row i is input row i, so the trained assignments seed directly.
        let mut result_assignments = trained.assignments.clone();
        let mut result_distances = vec![0.0_f32; n_vectors];
        batch::find_nearest_neighbor_with_pruning(
            data,
            centroids,
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
            partial_d,
            &mut vec![0_usize; n_vectors],
            &mut Vec::new(),
        );

        result_assignments
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

impl<'a> Run<'a> {
    /// A run of `iters` passes over `n` rows, starting from the `k` centroids
    /// in `seeds`.
    fn new(skm: &'a SuperKMeans, k: usize, n: usize, iters: u32, seeds: Vec<f32>) -> Self {
        Self {
            skm,
            k,
            n,
            iters,
            prev_centroids: seeds.clone(),
            horizontal_centroids: seeds,
            centroid_norms: Vec::new(),
            cluster_sizes: vec![0_u32; k],
            assignments: vec![0_u32; n],
            partial_d: skm.initial_partial_d(),
            iteration: 0,
            stats: Vec::new(),
            gemm_buf: Vec::new(),
        }
    }

    /// The Lloyd driver over rows held in memory: alternate assignment and
    /// centroid update for `iters` passes, widening `d'` as the pruning bound
    /// tightens and stopping early on convergence.
    ///
    /// `norms` holds the rows' full-width squared norms.
    fn fit_rows(&mut self, data: &[f32], norms: Vec<f32>) {
        let can_prune = self.skm.can_prune(self.k);
        let mut scratch = RowScratch {
            norms,
            distances: vec![0.0_f32; self.n],
            not_pruned_counts: vec![0_usize; self.n],
        };
        let mut partial_norms_computed = false;

        for iter_idx in 0..self.iters {
            self.iteration = iter_idx;
            let gemm_only = iter_idx == 0 || !can_prune;
            if !gemm_only && !partial_norms_computed {
                scratch.norms =
                    squared_norms_partial(data, self.n, self.skm.d, self.partial_d as usize);
                partial_norms_computed = true;
            }
            self.pass_over_rows(data, gemm_only, &mut scratch);

            if self.should_stop_early() {
                break;
            }
        }
    }

    /// [`Self::fit_rows`] for a scan: one pass over `data` per iteration.
    fn fit_stream<D: Dataset + ?Sized>(
        &mut self,
        data: &mut D,
        rotate: bool,
    ) -> Result<(), D::Error> {
        let can_prune = self.skm.can_prune(self.k);
        let mut scratch = StreamScratch::default();

        for iter_idx in 0..self.iters {
            self.iteration = iter_idx;
            let gemm_only = iter_idx == 0 || !can_prune;
            self.pass_over_stream(data, gemm_only, rotate, &mut scratch)?;
            if self.should_stop_early() {
                break;
            }
        }
        Ok(())
    }

    fn pass_over_rows(&mut self, data: &[f32], gemm_only: bool, scratch: &mut RowScratch) {
        let skm = self.skm;
        let (d, n_clusters, n_samples) = (skm.d, self.k, self.n);

        if self.iteration > 0 {
            std::mem::swap(&mut self.horizontal_centroids, &mut self.prev_centroids);
        }

        if gemm_only {
            self.centroid_norms = squared_norms(&self.prev_centroids, n_clusters, d);
            batch::find_nearest_neighbor(
                data,
                &self.prev_centroids,
                n_samples,
                n_clusters,
                d,
                &scratch.norms,
                &self.centroid_norms,
                &mut self.assignments,
                &mut scratch.distances,
                &mut self.gemm_buf,
            );
        } else {
            // Partial norms for pruning.
            self.centroid_norms =
                squared_norms_partial(&self.prev_centroids, n_clusters, d, self.partial_d as usize);
            scratch.not_pruned_counts.fill(0);
            batch::find_nearest_neighbor_with_pruning(
                data,
                &self.prev_centroids,
                n_samples,
                n_clusters,
                d,
                skm.vertical_d,
                skm.horizontal_d,
                &scratch.norms,
                &self.centroid_norms,
                &mut self.assignments,
                &mut scratch.distances,
                &skm.pruner,
                self.partial_d as usize,
                &mut scratch.not_pruned_counts,
                &mut self.gemm_buf,
            );
        }

        // M-step: total each cluster, which `consolidate_centroids` then averages.
        sum_rows_by_assignment(
            data,
            &self.assignments,
            &mut self.horizontal_centroids,
            &mut self.cluster_sizes,
            d,
        );

        let pruning = (!gemm_only).then(|| {
            let partial_d = self.partial_d;
            let not_pruned: f64 = scratch.not_pruned_counts.iter().map(|&v| v as f64).sum();
            let not_pruned_pct =
                self.adjust_partial_d((not_pruned / (n_samples as f64 * n_clusters as f64)) as f32);
            if self.partial_d != partial_d {
                scratch.norms = squared_norms_partial(data, n_samples, d, self.partial_d as usize);
            }
            PruningStats {
                partial_d,
                not_pruned_pct,
            }
        });

        let (empty_revived, small_revived) = self.consolidate_centroids();
        let sse: f64 = scratch.distances.iter().map(|&dist| dist as f64).sum();
        let shift = centroid_shift(
            &self.horizontal_centroids,
            &self.prev_centroids,
            n_clusters,
            d,
        );

        self.push_iteration_stats(sse, shift, empty_revived, small_revived, pruning);
    }

    /// [`Self::pass_over_rows`], a work batch at a time.
    fn pass_over_stream<D: Dataset + ?Sized>(
        &mut self,
        data: &mut D,
        gemm_only: bool,
        rotate: bool,
        scratch: &mut StreamScratch,
    ) -> Result<(), D::Error> {
        let skm = self.skm;
        let (d, n_clusters, n_samples) = (skm.d, self.k, self.n);
        let partial_d = self.partial_d as usize;

        if self.iteration > 0 {
            std::mem::swap(&mut self.horizontal_centroids, &mut self.prev_centroids);
        }
        self.horizontal_centroids.fill(0.0);
        self.cluster_sizes.fill(0);
        self.centroid_norms = if gemm_only {
            squared_norms(&self.prev_centroids, n_clusters, d)
        } else {
            squared_norms_partial(&self.prev_centroids, n_clusters, d, partial_d)
        };

        let mut offset = 0usize;
        let mut sse = 0.0_f64;
        let mut not_pruned = 0.0_f64;
        {
            let pruner = &skm.pruner;
            let (vertical_d, horizontal_d) = (skm.vertical_d, skm.horizontal_d);
            let prev_centroids = &self.prev_centroids;
            let centroid_norms = &self.centroid_norms;
            let assignments = &mut self.assignments;
            let gemm_buf = &mut self.gemm_buf;
            let sums = &mut self.horizontal_centroids;
            let cluster_sizes = &mut self.cluster_sizes;
            let StreamScratch {
                coalesce,
                rotated,
                distances,
                not_pruned_counts,
            } = scratch;

            dataset::visit_work_batches(data, d, X_BATCH_SIZE, coalesce, |batch| {
                let n = batch.n();
                assert!(
                    offset + n <= n_samples,
                    "dataset yielded more rows than on its first pass ({n_samples})"
                );
                let x: &[f32] = if rotate {
                    rotated.resize(n * d, 0.0);
                    pruner.rotate(batch.as_slice(), rotated, n);
                    &rotated[..]
                } else {
                    batch.as_slice()
                };
                let knn = &mut assignments[offset..offset + n];
                distances.resize(n, 0.0);

                if gemm_only {
                    let norms = squared_norms(x, n, d);
                    batch::find_nearest_neighbor(
                        x,
                        prev_centroids,
                        n,
                        n_clusters,
                        d,
                        &norms,
                        centroid_norms,
                        knn,
                        distances,
                        gemm_buf,
                    );
                } else {
                    let norms = squared_norms_partial(x, n, d, partial_d);
                    not_pruned_counts.clear();
                    not_pruned_counts.resize(n, 0);
                    batch::find_nearest_neighbor_with_pruning(
                        x,
                        prev_centroids,
                        n,
                        n_clusters,
                        d,
                        vertical_d,
                        horizontal_d,
                        &norms,
                        centroid_norms,
                        knn,
                        distances,
                        pruner,
                        partial_d,
                        not_pruned_counts,
                        gemm_buf,
                    );
                    not_pruned += not_pruned_counts.iter().map(|&v| v as f64).sum::<f64>();
                }

                sse += distances.iter().map(|&dist| dist as f64).sum::<f64>();
                accumulate_rows_by_assignment(x, knn, sums, cluster_sizes, d);
                offset += n;
            })?;
        }
        assert_eq!(
            offset, n_samples,
            "dataset yielded {offset} rows, but {n_samples} on its first pass"
        );

        let pruning = (!gemm_only).then(|| {
            let partial_d = self.partial_d;
            let not_pruned_pct =
                self.adjust_partial_d((not_pruned / (n_samples as f64 * n_clusters as f64)) as f32);
            PruningStats {
                partial_d,
                not_pruned_pct,
            }
        });

        let (empty_revived, small_revived) = self.consolidate_centroids();
        let shift = centroid_shift(
            &self.horizontal_centroids,
            &self.prev_centroids,
            n_clusters,
            d,
        );
        self.push_iteration_stats(sse, shift, empty_revived, small_revived, pruning);
        Ok(())
    }

    fn push_iteration_stats(
        &mut self,
        sse: f64,
        shift: f32,
        empty_revived: usize,
        small_revived: usize,
        pruning: Option<PruningStats>,
    ) {
        let stats = SuperKMeansIterationStats {
            iteration: (self.iteration + 1) as usize,
            n_samples: self.n,
            n_clusters: self.k,
            sse,
            mse: sse / self.n.max(1) as f64,
            shift,
            empty_revived,
            small_revived,
            pruning,
        };
        if self.skm.config.verbose {
            let improvement = self
                .stats
                .last()
                .map_or(0.0, |prev| 1.0 - (sse / prev.sse.max(f64::EPSILON)));
            print!(
                "Iteration {}/{} | MSE: {:.4} (SSE: {:.2}) | Improvement: {:.4} | Shift: {:.4} | Revived: {} ({}+{})",
                stats.iteration,
                self.iters,
                stats.mse,
                stats.sse,
                improvement,
                stats.shift,
                stats.revived(),
                stats.empty_revived,
                stats.small_revived,
            );
            match pruning {
                None => println!(" [BLAS-only]"),
                Some(p) => println!(
                    " | Not Pruned %: {:.4} | d': {} -> {}",
                    p.not_pruned_pct * 100.0,
                    p.partial_d,
                    self.partial_d
                ),
            }
        }
        self.stats.push(stats);
    }

    /// Turn the per-cluster sums an M-step leaves into usable centroids:
    /// take each cluster's mean, revive empty/small ones, and — for angular
    /// distance — put the result back on the unit sphere.
    ///
    /// Normalizing last matters: [`Self::split_clusters`] perturbs rows, so
    /// those have to land on the sphere too.
    fn consolidate_centroids(&mut self) -> (usize, usize) {
        let d = self.skm.d;
        mean_rows_by_count(&mut self.horizontal_centroids, &self.cluster_sizes, d);

        let revived = self.split_clusters();

        if self.skm.config.angular {
            normalize_rows_l2(&mut self.horizontal_centroids, self.k, d);
        }
        revived
    }

    fn split_clusters(&mut self) -> (usize, usize) {
        let config = &self.skm.config;
        let (d, n_clusters, n_samples) = (self.skm.d, self.k, self.n);
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // On the final iteration no E-step follows to separate a newly split
        // pair or update `assignments`, so only strictly empty clusters are
        // reseeded; non-empty small clusters keep their actual member mean.
        let is_last_iter = self.iteration + 1 >= self.iters;
        let revive_small = config.use_aggressive_split && !is_last_iter;
        let threshold = if revive_small {
            const UNUSED_THRESHOLD: f32 = 0.25;
            let average_size = n_samples as f32 / n_clusters.max(1) as f32;
            (average_size * UNUSED_THRESHOLD) as u32
        } else {
            0
        };

        let mut targets = Vec::new();
        let mut donors = Vec::new();
        for ci in 0..n_clusters {
            let size = self.cluster_sizes[ci];
            if size == 0 || (revive_small && size < threshold) {
                targets.push((ci, size == 0));
            } else if size > 1 {
                donors.push(ci);
            }
        }

        let n_split = targets.len().min(donors.len());
        if n_split == 0 {
            return (0, 0);
        }

        // Efraimidis-Spirakis / Gumbel-max sampling without replacement in
        // proportion to donor cluster size (`log(size) - log(-log(U))`).
        let uniform = Uniform::new(f64::MIN_POSITIVE, 1.0_f64);
        let mut keyed_donors: Vec<(f64, usize)> = donors
            .into_iter()
            .map(|ci| {
                let u = uniform.sample(&mut rng);
                let key = (self.cluster_sizes[ci] as f64).ln() - (-u.ln()).ln();
                (key, ci)
            })
            .collect();
        keyed_donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let mut empty_revived = 0usize;
        let mut small_revived = 0usize;
        for (&(target, was_empty), &(_, source)) in
            targets[..n_split].iter().zip(&keyed_donors[..n_split])
        {
            let (child, donor) =
                split_two_rows_mut(&mut self.horizontal_centroids, target, source, d);
            child.copy_from_slice(donor);
            for j in 0..d {
                if j % 2 == 0 {
                    donor[j] *= 1.0 - CENTROID_PERTURBATION_EPS;
                    child[j] *= 1.0 + CENTROID_PERTURBATION_EPS;
                } else {
                    donor[j] *= 1.0 + CENTROID_PERTURBATION_EPS;
                    child[j] *= 1.0 - CENTROID_PERTURBATION_EPS;
                }
            }
            if donor == child && d > 0 {
                let next = f32::from_bits(child[0].to_bits().wrapping_add(1));
                child[0] = if next.is_finite() {
                    next
                } else {
                    f32::from_bits(child[0].to_bits().wrapping_sub(1))
                };
            }
            let half = self.cluster_sizes[source] / 2;
            self.cluster_sizes[target] = half;
            self.cluster_sizes[source] -= half;
            if was_empty {
                empty_revived += 1;
            } else {
                small_revived += 1;
            }
        }
        (empty_revived, small_revived)
    }

    /// Widen or narrow `d'` given the average fraction of centroids that
    /// survived pruning, and return that fraction.
    fn adjust_partial_d(&mut self, avg: f32) -> f32 {
        let config = &self.skm.config;
        let vertical_d = self.skm.vertical_d as u32;
        if avg > config.max_not_pruned_pct {
            let increase =
                ((self.partial_d as f32) * config.adjustment_factor_for_partial_d * 2.0) as u32;
            self.partial_d = (self.partial_d + increase.max(1)).min(vertical_d);
        } else if avg < config.min_not_pruned_pct {
            let decrease =
                ((self.partial_d as f32) * config.adjustment_factor_for_partial_d) as u32;
            self.partial_d = (self.partial_d.saturating_sub(decrease.max(1))).max(MIN_PARTIAL_D);
        }
        avg
    }

    fn should_stop_early(&self) -> bool {
        let config = &self.skm.config;
        config.early_termination && should_stop_early(&self.stats, config.tol, config.verbose)
    }
}

/// Whether the last pass in `stats` converged.
fn should_stop_early(stats: &[SuperKMeansIterationStats], tol: f32, verbose: bool) -> bool {
    let Some(curr) = stats.last() else {
        return false;
    };
    // Revived centroids need one more E-step before centroids and assignments agree.
    if curr.small_revived > 0 {
        return false;
    }
    if curr.shift < tol {
        if verbose {
            println!(
                "Converged at iteration {} (shift {} < tol {})",
                curr.iteration, curr.shift, tol
            );
        }
        return true;
    }
    if let Some(prev) = stats.len().checked_sub(2).and_then(|i| stats.get(i)) {
        let sse_ratio = curr.sse / prev.sse.max(f64::EPSILON);
        if sse_ratio > 1.0 - tol as f64 {
            if verbose {
                println!(
                    "Converged at iteration {} (SSE improved by only {})",
                    curr.iteration,
                    1.0 - sse_ratio
                );
            }
            return true;
        }
    }
    false
}

fn expect_enough_rows(n: usize, k: usize) {
    assert!(n > 0, "n must be positive");
    assert!(n >= k, "n must be >= n_clusters ({n} < {k})");
}

/// `k` distinct rows of the `n × d` `data`, drawn by a seeded shuffle (Forgy).
fn forgy_sample(data: &[f32], n: usize, k: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    let mut sample = Vec::with_capacity(k * d);
    for &row in &indices[..k] {
        sample.extend_from_slice(&data[row * d..(row + 1) * d]);
    }
    sample
}

/// Per-row buffers of an in-memory run.
struct RowScratch {
    /// Squared norms of the rows, full-width or `d'`-wide to match the pass.
    norms: Vec<f32>,
    distances: Vec<f32>,
    not_pruned_counts: Vec<usize>,
}

/// Buffers reused across the work batches of every streaming pass; each is
/// sized for one work batch, not for the dataset.
#[derive(Default)]
struct StreamScratch {
    /// Small batches packed up to a full work batch.
    coalesce: Vec<f32>,
    /// The current work batch, rotated into the pruning domain.
    rotated: Vec<f32>,
    distances: Vec<f32>,
    not_pruned_counts: Vec<usize>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(sse: f64, shift: f32, small_revived: usize) -> SuperKMeansIterationStats {
        SuperKMeansIterationStats {
            sse,
            shift,
            small_revived,
            ..Default::default()
        }
    }

    #[test]
    fn pruning_engages_only_above_both_cutoffs() {
        let narrow = SuperKMeans::new(32);
        assert!(!narrow.can_prune(1_000));

        let wide = SuperKMeans::new(128);
        assert!(!wide.can_prune(N_CLUSTERS_THRESHOLD_FOR_PRUNING));
        assert!(wide.can_prune(N_CLUSTERS_THRESHOLD_FOR_PRUNING + 1));

        let blas = SuperKMeans::with_config(
            128,
            SuperKMeansConfig {
                use_blas_only: true,
                ..Default::default()
            },
        );
        assert!(!blas.can_prune(1_000));
    }

    fn stops_early(passes: Vec<SuperKMeansIterationStats>) -> bool {
        should_stop_early(&passes, SuperKMeansConfig::default().tol, false)
    }

    /// A revived small cluster holds a copy of its donor's centroid while its
    /// rows are still the old members, so neither convergence test may end
    /// the run on that pass.
    #[test]
    fn early_stop_waits_out_a_small_cluster_revival() {
        // SSE went up, which is what the E-step after a revival looks like.
        assert!(stops_early(vec![pass(100.0, 1.0, 0), pass(101.0, 1.0, 0)]));
        assert!(!stops_early(vec![pass(100.0, 1.0, 1), pass(101.0, 1.0, 1)]));

        // Centroids barely moved.
        assert!(stops_early(vec![pass(100.0, 0.0, 0)]));
        assert!(!stops_early(vec![pass(100.0, 0.0, 1)]));

        // Only the current pass matters: the E-step after a revival has
        // already moved the rows.
        assert!(stops_early(vec![pass(100.0, 1.0, 1), pass(101.0, 1.0, 0)]));
    }
}
