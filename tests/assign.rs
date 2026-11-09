//! Ported from SuperKMeans/tests/test_assign.cpp.
//!
//! `#[ignore]`-tagged tests are heavyweight; run with
//! `cargo test --release -- --ignored`.

use std::collections::HashSet;

use superkmeans::batch::{
    find_nearest_neighbor, find_soar_assignments, find_soar_replica_assignments,
};
use superkmeans::{
    Matrix, SuperKMeans, SuperKMeansConfig, assign, assign_soar, assign_soar_replicas,
    generate_random_vectors, make_blobs, squared_norms,
};

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

#[test]
fn standalone_assignment_matches_brute_force() {
    for d in [3, 64, 1024] {
        let mut data = generate_random_vectors(1031, d, -1.0, 1.0, 42);
        let mut centroids = generate_random_vectors(19, d, -1.0, 1.0, 17);
        centroids.extend_from_within(..d);
        data[..d].copy_from_slice(&centroids[..d]);
        for n in [0, 1, 1031] {
            let assigned = assign(
                Matrix::from_slice(&data[..n * d], d),
                Matrix::from_slice(&centroids, d),
            );
            assert_eq!(assigned.len(), n, "d={d}, n={n}");
            let mut incorrect = 0;
            for (i, &cluster) in assigned.iter().enumerate() {
                let point = &data[i * d..(i + 1) * d];
                let nearest = find_nearest_centroid_brute_force(point, &centroids, 20, d) as usize;
                let got = cluster as usize;
                if got == nearest {
                    continue;
                }
                // Centroid 19 is a copy of centroid 0, and the GEMM expansion of
                // ||x - c||^2 can break that tie the other way by a few ulps.
                let got_dist = l2_sq(point, &centroids[got * d..(got + 1) * d]);
                let nearest_dist = l2_sq(point, &centroids[nearest * d..(nearest + 1) * d]);
                if (got_dist - nearest_dist).abs() > 1e-4 * nearest_dist {
                    incorrect += 1;
                }
            }
            assert_eq!(
                incorrect, 0,
                "{incorrect} points not assigned to their nearest centroid (d={d}, n={n})"
            );
        }
    }
}

#[test]
#[should_panic(expected = "vectors and centroids must share d")]
fn assign_rejects_mismatched_dimensionality() {
    let vectors = Matrix::from_slice(&[0.0, 0.0], 2);
    let centroids = Matrix::from_slice(&[0.0], 1);
    let _ = assign(vectors, centroids);
}

// --- SOAR secondaries ---

/// The SOAR pair by brute force: the nearest centroid, then the argmin of
/// `‖x − c‖² + lambda · ⟨x − c, r̂1⟩²` over the other centroids, `r̂1` the unit residual to
/// the nearest (zero on the centroid itself). `u32::MAX` with one centroid.
fn soar_brute_force(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    d: usize,
    lambda: f32,
) -> (u32, u32) {
    let primary = find_nearest_centroid_brute_force(point, centroids, k, d);
    if k == 1 {
        return (primary, u32::MAX);
    }
    let nearest = &centroids[primary as usize * d..(primary as usize + 1) * d];
    let mut residual: Vec<f32> = point.iter().zip(nearest).map(|(a, b)| a - b).collect();
    let norm = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut residual {
            *v /= norm;
        }
    } else {
        residual.fill(0.0);
    }
    let along: f32 = point.iter().zip(&residual).map(|(a, b)| a * b).sum();
    let mut best = (f32::MAX, u32::MAX);
    for c in 0..k {
        if c as u32 == primary {
            continue;
        }
        let row = &centroids[c * d..(c + 1) * d];
        let projection: f32 = residual.iter().zip(row).map(|(a, b)| a * b).sum();
        let gap = along - projection;
        let cost = l2_sq(point, row) + lambda * gap * gap;
        if cost < best.0 {
            best = (cost, c as u32);
        }
    }
    (primary, best.1)
}

/// The SOAR objective of `c` for `point` whose nearest centroid is `primary`.
fn soar_cost(point: &[f32], centroids: &[f32], d: usize, primary: u32, c: u32, lambda: f32) -> f32 {
    let nearest = &centroids[primary as usize * d..(primary as usize + 1) * d];
    let mut residual: Vec<f32> = point.iter().zip(nearest).map(|(a, b)| a - b).collect();
    let norm = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut residual {
            *v /= norm;
        }
    } else {
        residual.fill(0.0);
    }
    let row = &centroids[c as usize * d..(c as usize + 1) * d];
    let gap: f32 = residual
        .iter()
        .zip(point.iter().zip(row))
        .map(|(r, (a, b))| r * (a - b))
        .sum();
    l2_sq(point, row) + lambda * gap * gap
}

/// Every secondary is off its primary and within rounding of the brute-force optimum, which may
/// differ in index on near ties between the GEMM and the direct distance.
#[allow(clippy::too_many_arguments)]
fn assert_soar_optimal(
    data: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
    lambda: f32,
    primary: &[u32],
    secondary: &[u32],
) {
    assert_eq!(primary.len(), n);
    assert_eq!(secondary.len(), n);
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let (expected_primary, expected_secondary) =
            soar_brute_force(point, centroids, k, d, lambda);
        assert_eq!(primary[i], expected_primary, "row {i}: primary");
        assert_ne!(
            secondary[i], primary[i],
            "row {i}: secondary equals primary"
        );
        assert!(
            (secondary[i] as usize) < k,
            "row {i}: secondary {} out of range",
            secondary[i]
        );
        let got = soar_cost(point, centroids, d, primary[i], secondary[i], lambda);
        let best = soar_cost(point, centroids, d, primary[i], expected_secondary, lambda);
        assert!(
            got <= best * (1.0 + 1e-4) + 1e-4,
            "row {i}: secondary {} costs {got}, brute force {} costs {best}",
            secondary[i],
            expected_secondary
        );
    }
}

#[test]
fn soar_primaries_equal_find_nearest_neighbor() {
    let (n, d, k) = (5_000, 32, 300);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 7);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 11);
    let norms_x = squared_norms(&data, n, d);
    let norms_y = squared_norms(&centroids, k, d);
    let mut nearest = vec![0_u32; n];
    let mut distances = vec![0.0_f32; n];
    let mut buf = Vec::new();
    find_nearest_neighbor(
        &data,
        &centroids,
        n,
        k,
        d,
        &norms_x,
        &norms_y,
        &mut nearest,
        &mut distances,
        &mut buf,
    );
    let mut ids = Vec::new();
    for candidates in [0, 1, 8, 64, k] {
        let mut primary = vec![0_u32; n];
        let mut secondary = vec![0_u32; n];
        find_soar_assignments(
            &data,
            &centroids,
            n,
            k,
            d,
            &norms_x,
            &norms_y,
            2.0,
            candidates,
            &mut primary,
            &mut secondary,
            &mut buf,
            &mut ids,
        );
        assert_eq!(primary, nearest, "candidates {candidates}");
    }
}

#[test]
fn soar_secondaries_minimize_the_objective() {
    let (n, d, k) = (3_000, 24, 200);
    let data = make_blobs(n, d, k, false, 1.0, 10.0, 5);
    let centroids = generate_random_vectors(k, d, -10.0, 10.0, 9);
    let kmeans = SuperKMeans::new(k, d);
    for lambda in [0.0_f32, 2.0] {
        let (primary, secondary) = kmeans.assign_soar(&data, &centroids, n, lambda, k);
        assert_soar_optimal(&data, &centroids, n, k, d, lambda, &primary, &secondary);
        let (free_primary, free_secondary) = assign_soar(
            Matrix::new(&data, n, d),
            Matrix::from_slice(&centroids, d),
            lambda,
            k,
        );
        assert_eq!(free_primary, primary);
        assert_eq!(free_secondary, secondary);
    }
}

#[test]
fn soar_with_lambda_zero_picks_the_second_nearest() {
    let (n, d, k) = (2_000, 16, 150);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 21);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 23);
    // Without the alignment term the second nearest wins, so two candidates are enough.
    let (primary, secondary) = assign_soar(
        Matrix::new(&data, n, d),
        Matrix::from_slice(&centroids, d),
        0.0,
        2,
    );
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let mut distances: Vec<(f32, u32)> = (0..k)
            .map(|c| (l2_sq(point, &centroids[c * d..(c + 1) * d]), c as u32))
            .collect();
        distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(primary[i], distances[0].1, "row {i}: primary");
        assert_ne!(secondary[i], primary[i], "row {i}");
        let got = l2_sq(
            point,
            &centroids[secondary[i] as usize * d..(secondary[i] as usize + 1) * d],
        );
        assert!(
            got <= distances[1].0 * (1.0 + 1e-5) + 1e-5,
            "row {i}: secondary {} at {got}, second nearest {} at {}",
            secondary[i],
            distances[1].1,
            distances[1].0
        );
    }
}

#[test]
fn soar_with_one_centroid_has_no_secondary() {
    let (n, d) = (500, 12);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 31);
    let centroid = generate_random_vectors(1, d, -1.0, 1.0, 37);
    let (primary, secondary) = assign_soar(
        Matrix::new(&data, n, d),
        Matrix::from_slice(&centroid, d),
        2.0,
        64,
    );
    assert!(primary.iter().all(|&c| c == 0));
    assert!(secondary.iter().all(|&c| c == u32::MAX));
}

#[test]
fn soar_crosses_tile_boundaries() {
    // 2,100 centroids exceed one y-batch on every target.
    let (n, d, k) = (2_000, 8, 2_100);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 41);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 43);
    let norms_x = squared_norms(&data, n, d);
    let norms_y = squared_norms(&centroids, k, d);
    let mut nearest = vec![0_u32; n];
    let mut distances = vec![0.0_f32; n];
    let mut buf = Vec::new();
    find_nearest_neighbor(
        &data,
        &centroids,
        n,
        k,
        d,
        &norms_x,
        &norms_y,
        &mut nearest,
        &mut distances,
        &mut buf,
    );
    let mut ids = Vec::new();
    let mut primary = vec![0_u32; n];
    let mut secondary = vec![0_u32; n];
    find_soar_assignments(
        &data,
        &centroids,
        n,
        k,
        d,
        &norms_x,
        &norms_y,
        2.0,
        k,
        &mut primary,
        &mut secondary,
        &mut buf,
        &mut ids,
    );
    assert_eq!(primary, nearest);
    assert_soar_optimal(&data, &centroids, n, k, d, 2.0, &primary, &secondary);

    // More vectors than one x-batch: a shortlist's choice is per row, so splitting the rows at
    // the x-batch boundary changes nothing.
    let n = superkmeans::common::X_BATCH_SIZE + 40;
    let data = generate_random_vectors(n, d, -1.0, 1.0, 47);
    let norms_x = squared_norms(&data, n, d);
    let soar = |rows: std::ops::Range<usize>| {
        let mut out = (vec![0_u32; rows.len()], vec![0_u32; rows.len()]);
        find_soar_assignments(
            &data[rows.start * d..rows.end * d],
            &centroids,
            rows.len(),
            k,
            d,
            &norms_x[rows.clone()],
            &norms_y,
            2.0,
            64,
            &mut out.0,
            &mut out.1,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        out
    };
    let whole = soar(0..n);
    let split = superkmeans::common::X_BATCH_SIZE / 2;
    let (head, tail) = (soar(0..split), soar(split..n));
    assert_eq!(whole.0, [head.0, tail.0].concat());
    assert_eq!(whole.1, [head.1, tail.1].concat());
}

#[test]
fn soar_large_centroid_set_is_deterministic_across_threads() {
    let (n, d, k) = (521, 8, 39_801);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 59);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 61);
    let vectors = Matrix::new(&data, n, d);
    let centers = Matrix::new(&centroids, k, d);
    let mut expected = None;
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let pair = pool.install(|| assign_soar(vectors, centers, 2.0, 64));
        if let Some(expected) = &expected {
            assert_eq!(&pair, expected);
        } else {
            assert_eq!(pair.0, assign(vectors, centers));
            assert!(
                pair.0
                    .iter()
                    .zip(&pair.1)
                    .all(|(a, b)| a != b && (*b as usize) < k)
            );
            expected = Some(pair);
        }
    }
}

#[test]
fn soar_ties_across_centroid_tiles_keep_the_first_centroid() {
    let k = superkmeans::common::Y_BATCH_SIZE + 3;
    let centroids = vec![0.0; k * 4];
    let data = vec![1.0; 257 * 4];
    for candidates in [2, 64, k] {
        let (primary, secondary) = assign_soar(
            Matrix::from_slice(&data, 4),
            Matrix::from_slice(&centroids, 4),
            2.0,
            candidates,
        );
        assert!(primary.iter().all(|&c| c == 0));
        assert!(secondary.iter().all(|&c| c == 1), "candidates {candidates}");
    }
}

#[test]
fn soar_four_replicas_minimize_cumulative_residual_loss() {
    let (n, d, k) = (257, 16, 79);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 71);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 73);
    let vectors = Matrix::new(&data, n, d);
    let centers = Matrix::new(&centroids, k, d);
    for lambda in [0.0, 2.0, 50.0] {
        let pair = assign_soar(vectors, centers, lambda, k);
        let (primary, secondary) = assign_soar_replicas(vectors, centers, lambda, 4, k);
        assert_eq!(primary, pair.0);
        assert_eq!(secondary.len(), n * 3);
        for row in 0..n {
            let point = &data[row * d..(row + 1) * d];
            let mut chosen = vec![primary[row]];
            assert_eq!(secondary[row * 3], pair.1[row]);
            for &actual in &secondary[row * 3..(row + 1) * 3] {
                assert!(!chosen.contains(&actual));
                let cost = |candidate: u32| {
                    let distance = l2_sq(point, &centroids[candidate as usize * d..][..d]);
                    distance
                        + chosen
                            .iter()
                            .map(|&previous| {
                                soar_cost(point, &centroids, d, previous, candidate, lambda)
                                    - distance
                            })
                            .sum::<f32>()
                };
                let best = (0..k as u32)
                    .filter(|c| !chosen.contains(c))
                    .map(cost)
                    .fold(f32::MAX, f32::min);
                assert!(cost(actual) <= best * (1.0 + 1e-4) + 1e-4);
                chosen.push(actual);
            }
        }
    }
}

#[test]
fn soar_replica_counts_handle_missing_centroids_and_zero_residuals() {
    // Every centroid ties at distance zero, so the shortlist is the lowest `candidates` indices
    // (at least one, at most k) and the secondaries fill it in index order.
    for k in 1..=5 {
        let centroids = vec![0.0; k * 4];
        let data = vec![0.0; 7 * 4];
        for candidates in [0, 1, 3, k, 64] {
            let shortlist = candidates.clamp(1, k);
            for replicas in [1, 2, 4, 8] {
                let (primary, secondary) = assign_soar_replicas(
                    Matrix::from_slice(&data, 4),
                    Matrix::from_slice(&centroids, 4),
                    2.0,
                    replicas,
                    candidates,
                );
                assert_eq!(primary, vec![0; 7]);
                assert_eq!(secondary.len(), 7 * (replicas - 1));
                if replicas > 1 {
                    for row in secondary.chunks(replicas - 1) {
                        let expected: Vec<_> = (1..replicas)
                            .map(|c| if c < shortlist { c as u32 } else { u32::MAX })
                            .collect();
                        assert_eq!(row, expected, "k {k}, candidates {candidates}");
                    }
                }
            }
        }
    }
}

#[test]
fn soar_four_replicas_are_deterministic_across_tiles_and_threads() {
    let (n, d, k) = (521, 8, 39_801);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 79);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 83);
    let mut expected = None;
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let assignments = pool.install(|| {
            assign_soar_replicas(
                Matrix::new(&data, n, d),
                Matrix::new(&centroids, k, d),
                2.0,
                4,
                64,
            )
        });
        if let Some(expected) = &expected {
            assert_eq!(&assignments, expected);
        } else {
            expected = Some(assignments);
        }
    }
    let centroids = vec![0.0; k * d];
    let (primary, secondary) = assign_soar_replicas(
        Matrix::new(&data, n, d),
        Matrix::new(&centroids, k, d),
        2.0,
        4,
        64,
    );
    assert_eq!(primary, vec![0; n]);
    assert!(secondary.chunks(3).all(|row| row == [1, 2, 3]));
}

#[test]
fn shortlisted_secondaries_stay_among_the_nearest() {
    let (n, d, k, candidates) = (2_000, 16, 200, 8);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 89);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 97);
    let lambda = 50.0;
    let (primary, secondary) = assign_soar_replicas(
        Matrix::new(&data, n, d),
        Matrix::new(&centroids, k, d),
        lambda,
        4,
        candidates,
    );
    let (exhaustive, _) = assign_soar_replicas(
        Matrix::new(&data, n, d),
        Matrix::new(&centroids, k, d),
        lambda,
        4,
        k,
    );
    assert_eq!(primary, exhaustive);
    for row in 0..n {
        let point = &data[row * d..(row + 1) * d];
        let mut nearest: Vec<(f32, u32)> = (0..k as u32)
            .map(|c| (l2_sq(point, &centroids[c as usize * d..][..d]), c))
            .collect();
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let bound = nearest[candidates - 1].0;
        let chosen = &secondary[row * 3..(row + 1) * 3];
        let mut all = vec![primary[row]];
        all.extend_from_slice(chosen);
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            4,
            "row {row}: {chosen:?} repeats or includes the primary"
        );
        for &c in chosen {
            let distance = l2_sq(point, &centroids[c as usize * d..][..d]);
            assert!(
                distance <= bound * (1.0 + 1e-5) + 1e-5,
                "row {row}: secondary {c} at {distance} lies outside the {candidates} nearest"
            );
        }
        // The first secondary is the objective's argmin over the shortlist.
        let best = nearest[..candidates]
            .iter()
            .filter(|&&(_, c)| c != primary[row])
            .map(|&(_, c)| soar_cost(point, &centroids, d, primary[row], c, lambda))
            .fold(f32::MAX, f32::min);
        let got = soar_cost(point, &centroids, d, primary[row], chosen[0], lambda);
        assert!(
            got <= best * (1.0 + 1e-4) + 1e-4,
            "row {row}: {got} vs {best}"
        );
    }
}

#[test]
fn candidates_beyond_the_centroid_count_are_exhaustive() {
    let (n, d, k) = (1_000, 16, 120);
    let data = generate_random_vectors(n, d, -1.0, 1.0, 101);
    let centroids = generate_random_vectors(k, d, -1.0, 1.0, 103);
    let soar = |candidates| {
        assign_soar_replicas(
            Matrix::new(&data, n, d),
            Matrix::new(&centroids, k, d),
            2.0,
            4,
            candidates,
        )
    };
    assert_eq!(soar(k + 100), soar(k));
    assert_eq!(soar(usize::MAX), soar(k));
}

#[test]
fn soar_scratch_is_reused_across_shapes() {
    let d = 16;
    let shapes = [(1_000, 300, 64), (3_000, 50, 16), (500, 300, 300)];
    let (mut buf, mut ids) = (Vec::new(), Vec::new());
    for (i, &(n, k, candidates)) in shapes.iter().enumerate() {
        let data = generate_random_vectors(n, d, -1.0, 1.0, 107 + i as u64);
        let centroids = generate_random_vectors(k, d, -1.0, 1.0, 109 + i as u64);
        let norms_x = squared_norms(&data, n, d);
        let norms_y = squared_norms(&centroids, k, d);
        let run = |buf: &mut Vec<f32>, ids: &mut Vec<u32>| {
            let mut out = (vec![0_u32; n], vec![0_u32; n * 3]);
            find_soar_replica_assignments(
                &data, &centroids, n, k, d, &norms_x, &norms_y, 2.0, 4, candidates, &mut out.0,
                &mut out.1, buf, ids,
            );
            out
        };
        let reused = run(&mut buf, &mut ids);
        let fresh = run(&mut Vec::new(), &mut Vec::new());
        assert_eq!(reused, fresh, "shape {i}");
    }
}
