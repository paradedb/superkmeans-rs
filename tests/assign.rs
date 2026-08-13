//! Ported from SuperKMeans/tests/test_assign.cpp.
//!
//! `#[ignore]`-tagged tests are heavyweight; run with
//! `cargo test --release -- --ignored`.

use std::collections::HashSet;

use superkmeans::{SuperKMeans, SuperKMeansConfig, make_blobs};

fn find_nearest_centroid_brute_force(
    point: &[f32],
    centroids: &[f32],
    n_clusters: usize,
    d: usize,
) -> u32 {
    let mut best_idx = 0_u32;
    let mut best_dist = f32::MAX;
    for c in 0..n_clusters {
        let row = &centroids[c * d..(c + 1) * d];
        let mut s = 0.0_f32;
        for k in 0..d {
            let diff = point[k] - row[k];
            s += diff * diff;
        }
        if s < best_dist {
            best_dist = s;
            best_idx = c as u32;
        }
    }
    best_idx
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[test]
fn each_point_assigned_to_nearest_centroid() {
    let n = 10_000;
    let d = 64;
    let n_clusters = 100;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42);

    let cfg = SuperKMeansConfig {
        iters: 10,
        verbose: false,
        seed: 42,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);
    let assignments = kmeans.assign(&data, &centroids, n);
    assert_eq!(assignments.len(), n);
    assert_eq!(centroids.len(), n_clusters * d);

    let mut incorrect = 0;
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let assigned = assignments[i] as usize;
        let nearest = find_nearest_centroid_brute_force(point, &centroids, n_clusters, d) as usize;
        if assigned != nearest {
            let a_dist = l2_sq(point, &centroids[assigned * d..(assigned + 1) * d]);
            let n_dist = l2_sq(point, &centroids[nearest * d..(nearest + 1) * d]);
            // Ties or near-ties don't count.
            if (a_dist - n_dist).abs() > 1e-4 * n_dist {
                incorrect += 1;
            }
        }
    }
    assert_eq!(
        incorrect, 0,
        "{incorrect} points not assigned to their nearest centroid"
    );
}

#[test]
#[ignore = "slow: d=512 Householder QR is expensive — run with --release --ignored"]
fn each_point_assigned_to_nearest_centroid_high_dim() {
    let n = 5_000;
    let d = 512;
    let n_clusters = 50;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 123);

    let cfg = SuperKMeansConfig {
        iters: 10,
        verbose: false,
        seed: 123,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);
    let assignments = kmeans.assign(&data, &centroids, n);
    assert_eq!(assignments.len(), n);

    let mut incorrect = 0;
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let assigned = assignments[i] as usize;
        let nearest = find_nearest_centroid_brute_force(point, &centroids, n_clusters, d) as usize;
        if assigned != nearest {
            let a_dist = l2_sq(point, &centroids[assigned * d..(assigned + 1) * d]);
            let n_dist = l2_sq(point, &centroids[nearest * d..(nearest + 1) * d]);
            if (a_dist - n_dist).abs() > 1e-4 * n_dist {
                incorrect += 1;
            }
        }
    }
    assert_eq!(
        incorrect, 0,
        "{incorrect} points not assigned to their nearest centroid (high-dim)"
    );
}

#[test]
fn use_train_state_matches_brute_force() {
    // k > N_CLUSTERS_THRESHOLD_FOR_PRUNING (256) so the pruning path is exercised.
    let n = 5_000;
    let d = 128;
    let n_clusters = 300;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42);

    let cfg = SuperKMeansConfig {
        iters: 5,
        verbose: false,
        seed: 42,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);

    let fast = kmeans.assign_training_points(&data, &centroids, n);
    let brute = kmeans.assign(&data, &centroids, n);
    assert_eq!(fast.len(), n);
    assert_eq!(brute.len(), n);

    let matches = fast.iter().zip(&brute).filter(|(a, b)| a == b).count();
    let pct = 100.0 * matches as f64 / n as f64;
    assert!(
        pct >= 98.0,
        "use_train_state should match brute force >=98%; got {pct:.2}% ({matches}/{n})"
    );
}

/// Sampling is the caller's job, so training on a subset is just training on a
/// smaller slice. The pruned assign path has to hold up there too.
#[test]
fn use_train_state_matches_brute_force_on_a_caller_side_sample() {
    let n = 5_000;
    let d = 128;
    let n_clusters = 300;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42);

    let cfg = SuperKMeansConfig {
        iters: 5,
        verbose: false,
        seed: 42,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);

    let sampled = n / 2;
    let sample = &data[..sampled * d];
    let centroids = kmeans.train(sample, sampled);

    let fast = kmeans.assign_training_points(sample, &centroids, sampled);
    let brute = kmeans.assign(sample, &centroids, sampled);
    assert_eq!(fast.len(), sampled);
    assert_eq!(brute.len(), sampled);

    let matches = fast.iter().zip(&brute).filter(|(a, b)| a == b).count();
    let pct = 100.0 * matches as f64 / sampled as f64;
    assert!(
        pct >= 98.0,
        "use_train_state should match brute force >=98%; got {pct:.2}% ({matches}/{sampled})"
    );
}

#[test]
#[ignore = "heavy: original C++ scale (n=50k, k=500). Run with --release --ignored."]
fn use_train_state_matches_brute_force_full() {
    let n = 50_000;
    let d = 128;
    let n_clusters = 500;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42);

    let cfg = SuperKMeansConfig {
        iters: 15,
        verbose: false,
        seed: 42,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);

    let fast = kmeans.assign_training_points(&data, &centroids, n);
    let brute = kmeans.assign(&data, &centroids, n);

    let matches = fast.iter().zip(&brute).filter(|(a, b)| a == b).count();
    let pct = 100.0 * matches as f64 / n as f64;
    assert!(
        pct >= 98.0,
        "use_train_state should match brute force >=98%; got {pct:.2}%"
    );
}

#[test]
fn all_clusters_non_empty() {
    let n = 10_000;
    let d = 128;
    let n_clusters = 100;
    let data = make_blobs(n, d, n_clusters, false, 1.0, 10.0, 42);

    let cfg = SuperKMeansConfig {
        iters: 15,
        verbose: false,
        seed: 42,
        unrotate_centroids: true,
        ..Default::default()
    };
    let mut kmeans = SuperKMeans::with_config(n_clusters, d, cfg);
    let centroids = kmeans.train(&data, n);
    let assignments = kmeans.assign(&data, &centroids, n);
    assert_eq!(assignments.len(), n);

    let mut counts = vec![0_usize; n_clusters];
    for &a in &assignments {
        assert!((a as usize) < n_clusters, "invalid cluster index {a}");
        counts[a as usize] += 1;
    }
    let empties: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == 0)
        .map(|(i, _)| i)
        .collect();
    assert!(
        empties.is_empty(),
        "found {} empty clusters out of {n_clusters}",
        empties.len()
    );

    let used: HashSet<u32> = assignments.iter().copied().collect();
    assert_eq!(used.len(), n_clusters);
}
