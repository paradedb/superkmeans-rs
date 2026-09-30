//! Streaming training: the data lives in a `Vec`, but training only ever sees
//! it through an iterator (or `Dataset`) of `Matrix` batches.

use std::convert::Infallible;

use superkmeans::{
    Dataset, FineClusters, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix,
    SuperKMeans, SuperKMeansConfig, make_blobs,
};

/// Mean squared distance from each row to the centroid it is assigned to.
fn distortion(data: &[f32], d: usize, centroids: &[f32], assignments: &[u32]) -> f64 {
    let total: f64 = data
        .chunks_exact(d)
        .zip(assignments)
        .map(|(x, &c)| {
            let c = &centroids[c as usize * d..(c as usize + 1) * d];
            x.iter()
                .zip(c)
                .map(|(a, b)| ((a - b) * (a - b)) as f64)
                .sum::<f64>()
        })
        .sum();
    total / assignments.len() as f64
}

fn leaf_config(max_leaf_size: usize) -> HierarchicalSuperKMeansConfig {
    HierarchicalSuperKMeansConfig {
        max_leaf_size,
        ..Default::default()
    }
}

fn nearest_distortion(kmeans: &SuperKMeans, data: &[f32], n: usize, centroids: &[f32]) -> f64 {
    let assignments = kmeans.assign(data, centroids, n);
    distortion(data, kmeans.d, centroids, &assignments)
}

#[test]
fn superkmeans_trains_from_an_iterator() {
    let (n, d, k) = (10_000, 32, 40);
    let data = make_blobs(n, d, k, true, 1.0, 10.0, 42);

    let mut streamed = SuperKMeans::new(k, d);
    let centroids = streamed.train_iter(Matrix::new(&data, n, d).chunks(1_000));
    assert!(streamed.trained);
    assert_eq!(centroids.len(), k * d);
    assert_eq!(streamed.n_samples, n);

    let mut in_memory = SuperKMeans::new(k, d);
    let reference = in_memory.train(&data, n);

    let streamed_distortion = nearest_distortion(&streamed, &data, n, &centroids);
    let reference_distortion = nearest_distortion(&in_memory, &data, n, &reference);
    assert!(
        streamed_distortion <= reference_distortion * 1.1,
        "streamed distortion {streamed_distortion} vs in-memory {reference_distortion}"
    );
}

/// Batch shape is an input-side detail: one vector per item and large chunks
/// are packed into the same work batches, so they train identically.
#[test]
fn batch_shape_does_not_change_the_result() {
    let (n, d, k) = (5_000, 16, 20);
    let data = make_blobs(n, d, k, true, 1.0, 10.0, 3);

    let mut by_vector = SuperKMeans::new(k, d);
    let one_at_a_time = by_vector.train_iter(data.chunks_exact(d).map(Matrix::vector));

    let mut by_chunk = SuperKMeans::new(k, d);
    let chunked = by_chunk.train_iter(Matrix::new(&data, n, d).chunks(777));

    assert_eq!(one_at_a_time, chunked);
}

/// d >= 128 and k > 256 turn on the pruning kernels, which seed each pass
/// with the previous pass's assignments.
#[test]
fn streaming_with_pruning_matches_in_memory_quality() {
    let (n, d, k) = (20_000, 128, 300);
    let data = make_blobs(n, d, 60, true, 1.0, 10.0, 11);
    let cfg = SuperKMeansConfig {
        iters: 8,
        early_termination: false,
        ..Default::default()
    };

    let mut streamed = SuperKMeans::with_config(k, d, cfg.clone());
    let centroids = streamed.train_iter(Matrix::new(&data, n, d).chunks(4_096));
    assert!(
        streamed.iteration_stats.iter().any(|s| !s.is_gemm_only),
        "pruning never engaged"
    );

    let mut in_memory = SuperKMeans::with_config(k, d, cfg);
    let reference = in_memory.train(&data, n);

    let streamed_distortion = nearest_distortion(&streamed, &data, n, &centroids);
    let reference_distortion = nearest_distortion(&in_memory, &data, n, &reference);
    assert!(
        streamed_distortion <= reference_distortion * 1.1,
        "streamed distortion {streamed_distortion} vs in-memory {reference_distortion}"
    );
}

/// The full hierarchical flow the way a caller that can't hold the data runs
/// it: stream the meso step, then load and split one meso cluster at a time.
#[test]
fn hierarchical_trains_from_an_iterator() {
    let (n, d, max_leaf_size) = (20_000usize, 32usize, 100usize);
    let data = make_blobs(n, d, 40, true, 1.0, 10.0, 7);
    let cfg = leaf_config(max_leaf_size);
    let kmeans = HierarchicalSuperKMeans::with_config(d, cfg.clone());

    let meso = kmeans.train_meso_iter(Matrix::new(&data, n, d).chunks(1_000), n);
    assert!(meso.n_meso() >= 2);

    let mut centroids = Vec::new();
    let mut assignments = vec![u32::MAX; n];
    let mut sizes = Vec::new();
    for m in 0..meso.n_meso() {
        let rows = meso.meso_rows(m);
        assert!(!rows.is_empty(), "meso cluster {m} is empty");
        assert!(rows.windows(2).all(|w| w[0] < w[1]), "rows not ascending");

        // What a caller does with random access: fetch this cluster's rows.
        let vectors: Vec<f32> = rows
            .iter()
            .flat_map(|&r| data[r as usize * d..(r as usize + 1) * d].iter().copied())
            .collect();
        let fine: FineClusters = kmeans.train_meso_cluster(&vectors, rows.len());

        let first = sizes.len() as u32;
        for c in 0..fine.n_fine() {
            let members = fine.members(c);
            sizes.push(members.len());
            for &pos in members {
                let row = rows[pos as usize] as usize;
                assert_eq!(assignments[row], u32::MAX, "row {row} assigned twice");
                assignments[row] = first + c as u32;
            }
        }
        centroids.extend_from_slice(&fine.centroids);
    }

    assert!(
        assignments.iter().all(|&a| a != u32::MAX),
        "rows left unassigned"
    );
    assert!(sizes.iter().all(|&s| s > 0 && s <= max_leaf_size));
    assert!(sizes.len() >= n.div_ceil(max_leaf_size));

    let mut in_memory = HierarchicalSuperKMeans::with_config(d, cfg);
    let reference = in_memory.train(&data, n);
    let reference_assignments = in_memory.assign(&data, &reference, n);

    let streamed_distortion = distortion(&data, d, &centroids, &assignments);
    let reference_distortion = distortion(&data, d, &reference, &reference_assignments);
    assert!(
        streamed_distortion <= reference_distortion * 1.1,
        "streamed distortion {streamed_distortion} vs in-memory {reference_distortion}"
    );
}

/// A scan wrapper that counts how many passes training makes over it.
struct Counting<'a> {
    data: Matrix<'a>,
    passes: usize,
}

impl Dataset for Counting<'_> {
    type Error = Infallible;

    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) -> Result<(), Infallible> {
        self.passes += 1;
        for chunk in self.data.chunks(512) {
            f(chunk);
        }
        Ok(())
    }
}

#[test]
fn meso_step_makes_one_pass_per_iteration_plus_one() {
    let (n, d) = (8_000usize, 16usize);
    let data = make_blobs(n, d, 10, true, 1.0, 10.0, 5);
    let mut cfg = HierarchicalSuperKMeansConfig {
        iters_meso: 4,
        ..leaf_config(50)
    };
    cfg.base.early_termination = false;
    let kmeans = HierarchicalSuperKMeans::with_config(d, cfg);

    let mut scan = Counting {
        data: Matrix::new(&data, n, d),
        passes: 0,
    };
    let meso = kmeans.train_meso(&mut scan, n).unwrap();
    assert_eq!(scan.passes, 1 + 4);
    assert_eq!(meso.stats.len(), 4);

    let mut seen = vec![false; n];
    for &row in &meso.rows {
        assert!(!std::mem::replace(&mut seen[row as usize], true));
    }
    assert!(seen.iter().all(|&s| s), "partition dropped rows");
}

#[test]
fn a_sample_that_fits_one_leaf_needs_no_meso_pass() {
    let (n, d) = (64usize, 8usize);
    let data = make_blobs(n, d, 4, true, 1.0, 10.0, 2);
    let kmeans = HierarchicalSuperKMeans::with_config(d, leaf_config(64));

    let mut scan = Counting {
        data: Matrix::new(&data, n, d),
        passes: 0,
    };
    let meso = kmeans.train_meso(&mut scan, n).unwrap();
    assert_eq!(scan.passes, 0);
    assert_eq!(meso.n_meso(), 1);

    let fine = kmeans.train_meso_cluster(&data, n);
    assert_eq!(fine.n_fine(), 1);
    assert_eq!(fine.members(0).len(), n);
}

/// A scan that fails partway through its second pass.
struct FailsOnSecondPass<'a> {
    data: Matrix<'a>,
    passes: usize,
}

impl Dataset for FailsOnSecondPass<'_> {
    type Error = String;

    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) -> Result<(), String> {
        self.passes += 1;
        for chunk in self.data.chunks(256) {
            if self.passes == 2 {
                return Err("read failed".to_string());
            }
            f(chunk);
        }
        Ok(())
    }
}

#[test]
fn scan_errors_come_back_out_of_training() {
    let (n, d) = (4_000usize, 16usize);
    let data = make_blobs(n, d, 10, true, 1.0, 10.0, 9);

    let mut scan = FailsOnSecondPass {
        data: Matrix::new(&data, n, d),
        passes: 0,
    };
    let mut kmeans = SuperKMeans::new(10, d);
    assert_eq!(
        kmeans.train_dataset(&mut scan),
        Err("read failed".to_string())
    );
    assert!(!kmeans.trained);

    let hierarchical = HierarchicalSuperKMeans::with_config(d, leaf_config(50));
    scan.passes = 0;
    assert_eq!(
        hierarchical.train_meso(&mut scan, n).unwrap_err(),
        "read failed"
    );
}

#[test]
#[should_panic(expected = "expected 100")]
fn train_meso_rejects_a_wrong_row_count() {
    let (n, d) = (4_000usize, 16usize);
    let data = make_blobs(n, d, 10, true, 1.0, 10.0, 9);
    let kmeans = HierarchicalSuperKMeans::with_config(d, leaf_config(10));
    let _ = kmeans.train_meso_iter(Matrix::new(&data, n, d).chunks(500), 100);
}
