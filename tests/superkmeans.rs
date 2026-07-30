//! Ported from SuperKMeans/tests/test_superkmeans.cpp.

use std::collections::HashSet;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

#[allow(unused_imports)]
use rand_distr::Distribution as _;

use superkmeans::adsampling::ADSamplingPruner;
use superkmeans::common::PRUNER_INITIAL_THRESHOLD;
use superkmeans::{SuperKMeans, SuperKMeansConfig, make_blobs};

fn default_blobs(n: usize, d: usize, n_clusters: usize) -> Vec<f32> {
    make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42)
}

#[test]
fn basic_training_small_dataset() {
    let n = 1000;
    let d = 32;
    let n_clusters = 10;
    let data = default_blobs(n, d, n_clusters);

    let cfg = SuperKMeansConfig {
        iters: 10,
        verbose: false,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    assert!(!kmeans.trained);

    let centroids = kmeans.train(&data, n);
    assert!(kmeans.trained);
    assert_eq!(centroids.len(), n_clusters * d);
    assert_eq!(kmeans.n_clusters, n_clusters);
}

#[test]
fn all_clusters_used() {
    let n = 10_000;
    let d = 128;
    let n_clusters = 50;
    let data = default_blobs(n, d, n_clusters);

    let cfg = SuperKMeansConfig {
        iters: 25,
        verbose: false,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);

    let assignments = kmeans.assign(&data, &centroids, n);
    let used: HashSet<u32> = assignments.iter().copied().collect();
    assert_eq!(
        used.len(),
        n_clusters,
        "expected all {n_clusters} clusters to be used, only {} were",
        used.len()
    );
}

#[test]
fn perform_assignments_populates_assignments() {
    let n = 1000;
    let d = 32;
    let n_clusters = 10;
    let data = default_blobs(n, d, n_clusters);

    let cfg = SuperKMeansConfig {
        iters: 10,
        verbose: false,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);
    let assignments = kmeans.assign(&data, &centroids, n);

    assert_eq!(assignments.len(), n);
    for (i, &a) in assignments.iter().enumerate() {
        assert!(
            (a as usize) < n_clusters,
            "assignment {i} has invalid cluster index: {a}"
        );
    }
}

// ----- InvalidInputs: split into one #[should_panic] per case -----

#[test]
#[should_panic(expected = "n must be >= n_clusters")]
fn invalid_n_less_than_n_clusters() {
    let n = 1000;
    let d = 32;
    let data = default_blobs(n, d, 10);
    let mut kmeans = SuperKMeans::new(n + 10, d);
    kmeans.train(&data, n);
}

#[test]
#[should_panic(expected = "Not enough samples")]
fn invalid_sampling_fraction_too_low() {
    let n = 10_000;
    let d = 32;
    let data = default_blobs(n, d, 10);
    let cfg = SuperKMeansConfig {
        sampling_fraction: 0.0001,
        max_points_per_cluster: 1,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(10, d, cfg);
    kmeans.train(&data, n);
}

#[test]
#[should_panic(expected = "n_clusters must be positive")]
fn invalid_zero_n_clusters() {
    let _ = SuperKMeans::new(0, 32);
}

#[test]
#[should_panic(expected = "dimensionality must be positive")]
fn invalid_zero_dimensionality() {
    let _ = SuperKMeans::new(10, 0);
}

#[test]
#[should_panic(expected = "iters must be positive")]
fn invalid_zero_iters() {
    let cfg = SuperKMeansConfig {
        iters: 0,
        ..Default::default()
    };
    let _ = SuperKMeans::with_config(10, 32, cfg);
}

#[test]
#[should_panic(expected = "sampling_fraction must be positive")]
fn invalid_zero_sampling_fraction() {
    let cfg = SuperKMeansConfig {
        sampling_fraction: 0.0,
        ..Default::default()
    };
    let _ = SuperKMeans::with_config(10, 32, cfg);
}

#[test]
#[should_panic(expected = "sampling_fraction must be positive")]
fn invalid_negative_sampling_fraction() {
    let cfg = SuperKMeansConfig {
        sampling_fraction: -0.5,
        ..Default::default()
    };
    let _ = SuperKMeans::with_config(10, 32, cfg);
}

#[test]
#[should_panic(expected = "sampling_fraction must be <= 1.0")]
fn invalid_sampling_fraction_above_one() {
    let cfg = SuperKMeansConfig {
        sampling_fraction: 1.5,
        ..Default::default()
    };
    let _ = SuperKMeans::with_config(10, 32, cfg);
}

#[test]
#[should_panic(expected = "already been trained")]
fn invalid_training_twice() {
    let n = 1000;
    let d = 32;
    let data = default_blobs(n, d, 10);
    let mut kmeans = SuperKMeans::new(10, d);
    let _ = kmeans.train(&data, n);
    let _ = kmeans.train(&data, n);
}

// ----- Early termination -----

fn make_well_separated_blobs(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise = Normal::new(0.0_f32, 0.5).unwrap();
    let mut centers = vec![0.0_f32; n_clusters * d];
    for c in 0..n_clusters {
        for j in 0..d {
            centers[c * d + j] = c as f32 * 20.0 + if j % 2 == 0 { 5.0 } else { -5.0 };
        }
    }
    let mut data = vec![0.0_f32; n * d];
    for i in 0..n {
        let cluster = i % n_clusters;
        for j in 0..d {
            data[i * d + j] = centers[cluster * d + j] + noise.sample(&mut rng);
        }
    }
    data
}

#[test]
fn early_termination_shift_below_tol_stops() {
    let n = 10_000;
    let d = 64;
    let n_clusters = 5;
    let max_iters = 100;
    let data = make_well_separated_blobs(n, d, n_clusters, 42);

    let cfg_early = SuperKMeansConfig {
        iters: max_iters,
        early_termination: true,
        tol: 1e-2,
        verbose: false,
        seed: 42,
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut k_early = SuperKMeans::with_config(n_clusters, d, cfg_early);
    k_early.train(&data, n);
    let iters_with_early = k_early.iteration_stats.len();

    let cfg_no_early = SuperKMeansConfig {
        iters: max_iters,
        early_termination: false,
        verbose: false,
        seed: 42,
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut k_no_early = SuperKMeans::with_config(n_clusters, d, cfg_no_early);
    k_no_early.train(&data, n);
    let iters_no_early = k_no_early.iteration_stats.len();

    assert!(
        iters_with_early < max_iters as usize,
        "early termination should stop before max_iters={max_iters}, got {iters_with_early}"
    );
    assert_eq!(
        iters_no_early, max_iters as usize,
        "without early termination should run all {max_iters} iterations"
    );
    assert!(
        iters_with_early < iters_no_early,
        "early termination ({iters_with_early}) should use fewer iterations than no early ({iters_no_early})"
    );
}

#[test]
fn early_termination_disabled_runs_all_iterations() {
    let n = 10_000;
    let d = 32;
    let n_clusters = 5;
    let max_iters = 50;
    let data = default_blobs(n, d, n_clusters);

    let cfg = SuperKMeansConfig {
        iters: max_iters,
        early_termination: false,
        verbose: false,
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    kmeans.train(&data, n);
    assert_eq!(
        kmeans.iteration_stats.len(),
        max_iters as usize,
        "with early_termination=false, should run exactly {max_iters} iterations"
    );
}

// ----- Angular mode -----

#[test]
fn angular_mode_normalizes() {
    let n = 5000;
    let d = 64;
    let n_clusters = 50;
    let data = default_blobs(n, d, n_clusters);

    let cfg = SuperKMeansConfig {
        iters: 10,
        angular: true,
        verbose: false,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);
    assert_eq!(centroids.len(), n_clusters * d);

    for c in 0..n_clusters {
        let row = &centroids[c * d..(c + 1) * d];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "centroid {c} should be unit-norm in angular mode (got {norm})"
        );
    }
}

// ----- Pre-rotated data equivalence -----

#[test]
fn pre_rotated_data_produces_identical_results() {
    let n = 10_000;
    let d = 256;
    let k = 100;
    let seed = 42_u64;
    let data = make_blobs(n, d, k, false, 1.0, 10.0, seed);

    // Case 1: SuperKMeans rotates internally, returns rotated centroids.
    let cfg1 = SuperKMeansConfig {
        iters: 10,
        seed,
        verbose: false,
        data_already_rotated: false,
        unrotate_centroids: false,
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut k1 = SuperKMeans::with_config(k, d, cfg1);
    let centroids1 = k1.train(&data, n);

    // Case 2: rotate externally with the same pruner config, then run with
    // data_already_rotated=true (which forces unrotate_centroids=false).
    let pruner = ADSamplingPruner::new(d, PRUNER_INITIAL_THRESHOLD, seed);
    let mut rotated_data = vec![0.0_f32; n * d];
    pruner.rotate(&data, &mut rotated_data, n);

    let cfg2 = SuperKMeansConfig {
        iters: 10,
        seed,
        verbose: false,
        data_already_rotated: true,
        unrotate_centroids: true, // should be forced to false by the constructor
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut k2 = SuperKMeans::with_config(k, d, cfg2);
    let centroids2 = k2.train(&rotated_data, n);

    assert_eq!(centroids1.len(), centroids2.len());
    assert_eq!(centroids1.len(), k * d);

    let mut mismatches = 0;
    let mut max_abs_error = 0.0_f32;
    let mut sum_abs_error = 0.0_f32;
    for i in 0..k * d {
        let e = (centroids1[i] - centroids2[i]).abs();
        if e > max_abs_error {
            max_abs_error = e;
        }
        sum_abs_error += e;
        if e > 1e-4 {
            mismatches += 1;
        }
    }
    let avg_abs_error = sum_abs_error / (k * d) as f32;

    assert_eq!(
        mismatches,
        0,
        "centroids should match within numerical precision. mismatches={mismatches}/{}, max={max_abs_error}, avg={avg_abs_error}",
        k * d
    );
    assert!(
        max_abs_error < 1e-3,
        "max abs error {max_abs_error} too large"
    );
    assert!(
        avg_abs_error < 1e-5,
        "avg abs error {avg_abs_error} too large"
    );
}

fn assert_centroids_match(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs_error = 0.0_f32;
    for i in 0..a.len() {
        let e = (a[i] - b[i]).abs();
        if e > max_abs_error {
            max_abs_error = e;
        }
    }
    assert!(
        max_abs_error < tol,
        "max abs error {max_abs_error} exceeds {tol}"
    );
}

#[test]
fn train_owned_matches_train() {
    let n = 5_000;
    let d = 64;
    let k = 50;
    let data = default_blobs(n, d, k);

    let cfg = SuperKMeansConfig {
        iters: 10,
        seed: 42,
        sampling_fraction: 1.0,
        ..Default::default()
    };
    let mut borrowed = SuperKMeans::with_config(k, d, cfg.clone());
    let centroids_borrowed = borrowed.train(&data, n);

    let mut owned = SuperKMeans::with_config(k, d, cfg);
    let centroids_owned = owned.train_owned(data.clone(), n);

    assert_centroids_match(&centroids_borrowed, &centroids_owned, 1e-4);
}

#[test]
fn train_owned_matches_train_with_sampling() {
    let n = 5_000;
    let d = 64;
    let k = 50;
    let data = default_blobs(n, d, k);

    let cfg = SuperKMeansConfig {
        iters: 10,
        seed: 42,
        sampling_fraction: 0.5,
        ..Default::default()
    };
    let mut borrowed = SuperKMeans::with_config(k, d, cfg.clone());
    let centroids_borrowed = borrowed.train(&data, n);

    let mut owned = SuperKMeans::with_config(k, d, cfg);
    let centroids_owned = owned.train_owned(data.clone(), n);

    assert_centroids_match(&centroids_borrowed, &centroids_owned, 1e-4);
}

#[test]
fn hierarchical_train_owned_matches_train() {
    use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig};

    let n = 10_000;
    let d = 64;
    let k = 200;
    let data = make_blobs(n, d, k, false, 1.0, 10.0, 42);

    let mut cfg = HierarchicalSuperKMeansConfig::default();
    cfg.base.seed = 42;
    cfg.base.suppress_warnings = true;

    let mut borrowed = HierarchicalSuperKMeans::with_config(k, d, cfg.clone());
    let centroids_borrowed = borrowed.train(&data, n);

    let mut owned = HierarchicalSuperKMeans::with_config(k, d, cfg);
    let centroids_owned = owned.train_owned(data.clone(), n);

    assert_centroids_match(&centroids_borrowed, &centroids_owned, 1e-4);
}
