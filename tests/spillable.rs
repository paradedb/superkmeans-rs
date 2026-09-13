use std::cell::Cell;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use superkmeans::{
    FileTempStorage, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix, NodeId,
    SpillOptions, SuperKMeans, SuperKMeansConfig, TempMatrix, TempStorage, TryDataset, make_blobs,
};

const OPTIONS: SpillOptions = SpillOptions {
    memory_budget: 128 * 1024,
};

#[test]
fn flat_spill_matches_streaming_and_retains_no_per_row_arrays() {
    let (n, d, k) = (1400, 16, 12);
    let data = make_blobs(n, d, k, true, 1.0, 8.0, 7);
    let cfg = SuperKMeansConfig {
        early_termination: false,
        iters: 4,
        data_already_rotated: true,
        ..Default::default()
    };
    let mut reference = SuperKMeans::with_config(k, d, cfg.clone());
    let expected = reference.train_iter(Matrix::new(&data, n, d).chunks(17));
    let mut spilled = SuperKMeans::with_config(k, d, cfg);
    let got = spilled
        .train_spillable(&mut Matrix::new(&data, n, d), &mut FileTempStorage, OPTIONS)
        .unwrap();
    assert!(expected.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-5));
    assert_eq!(reference.cluster_sizes, spilled.cluster_sizes);
    assert_eq!(spilled.n_samples, n);
    assert!(spilled.assignments.is_empty());
    assert!(spilled.distances.is_empty());
    assert!(spilled.data_norms.is_empty());
}

#[test]
fn spill_preserves_pruning_warm_starts() {
    let (n, d, k) = (800, 128, 257);
    let data = make_blobs(n, d, 30, true, 1.0, 10.0, 7);
    let cfg = SuperKMeansConfig {
        early_termination: false,
        iters: 3,
        data_already_rotated: true,
        ..Default::default()
    };
    let mut reference = SuperKMeans::with_config(k, d, cfg.clone());
    let expected = reference.train_iter(Matrix::new(&data, n, d).chunks(29));
    let mut spilled = SuperKMeans::with_config(k, d, cfg);
    let got = spilled
        .train_spillable(
            &mut Matrix::new(&data, n, d),
            &mut FileTempStorage,
            SpillOptions {
                memory_budget: 1024 * 1024,
            },
        )
        .unwrap();
    assert!(!spilled.iteration_stats[1].is_gemm_only);
    assert!(expected.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-5));
    assert_eq!(reference.cluster_sizes, spilled.cluster_sizes);
}

#[test]
fn hierarchy_covers_rows_and_obeys_leaf_cap() {
    for branching_factor in [None, Some(2)] {
        let (n, d) = (1200, 12);
        let data = make_blobs(n, d, 16, true, 1.0, 8.0, 9);
        let cfg = HierarchicalSuperKMeansConfig {
            max_leaf_size: 40,
            branching_factor,
            ..Default::default()
        };
        let mut model = HierarchicalSuperKMeans::with_config(d, cfg.clone());
        let centroids = model
            .train_spillable(&mut Matrix::new(&data, n, d), &mut FileTempStorage, OPTIONS)
            .unwrap();
        assert_eq!(centroids.len(), model.tree.n_leaves * d);
        assert_eq!(
            model.tree.leaves().map(|node| node.size()).sum::<usize>(),
            n
        );
        assert!(
            model
                .tree
                .leaves()
                .all(|node| node.size() > 0 && node.size() <= 40)
        );
        for (i, node) in model.tree.nodes.iter().enumerate() {
            if !node.is_leaf() {
                assert_eq!(
                    model
                        .tree
                        .children(NodeId(i))
                        .map(|child| model.tree.node(child).size())
                        .sum::<usize>(),
                    node.size()
                );
                let mut offset = node.data_offset();
                for child in model.tree.children(NodeId(i)) {
                    let child = model.tree.node(child);
                    assert_eq!(child.data_offset(), offset);
                    offset += child.size();
                }
            }
        }
        let mut replay = HierarchicalSuperKMeans::with_config(d, cfg);
        let mut chunks = superkmeans::IterDataset::new(Matrix::new(&data, n, d).chunks(7));
        let again = replay
            .train_spillable(&mut chunks, &mut FileTempStorage, OPTIONS)
            .unwrap();
        assert_eq!(centroids, again);
        let labels = model.assign(&data, &centroids, n);
        let distortion = data
            .chunks_exact(d)
            .zip(labels)
            .map(|(row, label)| {
                row.iter()
                    .zip(&centroids[label as usize * d..(label as usize + 1) * d])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / n as f32;
        assert!(distortion < 0.5, "uninformative centroids: {distortion}");
    }
}

#[test]
fn collapsed_splits_bisect_and_small_inputs_stay_leaves() {
    for n in [1, 16, 513] {
        let d = 8;
        let data = vec![0.0; n * d];
        let cfg = HierarchicalSuperKMeansConfig {
            max_leaf_size: 16,
            ..Default::default()
        };
        let mut model = HierarchicalSuperKMeans::with_config(d, cfg);
        let centroids = model
            .train_spillable(&mut Matrix::new(&data, n, d), &mut FileTempStorage, OPTIONS)
            .unwrap();
        assert!(centroids.iter().all(|&v| v == 0.0));
        assert!(model.tree.leaves().all(|node| node.size() <= 16));
        assert_eq!(
            model.tree.leaves().map(|node| node.size()).sum::<usize>(),
            n
        );
        if n <= 16 {
            assert_eq!(model.tree.n_leaves, 1);
        }
    }
}

#[test]
fn temporary_matrix_replays_replaced_rows() {
    let mut matrix = TempMatrix::new(FileTempStorage.create().unwrap(), 2, 2).unwrap();
    matrix
        .append(Matrix::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2))
        .unwrap();
    matrix.replace_row(1, &[7.0, 8.0]).unwrap();
    for _ in 0..2 {
        let mut values = Vec::new();
        matrix
            .try_for_each_batch(&mut |batch| {
                values.extend_from_slice(batch.as_slice());
                Ok(())
            })
            .unwrap();
        assert_eq!(values, [1.0, 2.0, 7.0, 8.0, 5.0, 6.0]);
    }
}

struct TrackingStorage {
    live: Rc<Cell<usize>>,
    peak: Rc<Cell<usize>>,
    fail_reads: bool,
}
struct TrackingFile {
    data: Cursor<Vec<u8>>,
    live: Rc<Cell<usize>>,
    fail_reads: bool,
}
impl Drop for TrackingFile {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
impl Read for TrackingFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.fail_reads {
            Err(io::Error::other("injected read failure"))
        } else {
            self.data.read(buf)
        }
    }
}
impl Write for TrackingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Seek for TrackingFile {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.data.seek(position)
    }
}
impl TempStorage for TrackingStorage {
    type File = TrackingFile;
    fn create(&mut self) -> io::Result<TrackingFile> {
        self.live.set(self.live.get() + 1);
        self.peak.set(self.peak.get().max(self.live.get()));
        Ok(TrackingFile {
            data: Cursor::new(Vec::new()),
            live: self.live.clone(),
            fail_reads: self.fail_reads,
        })
    }
}

#[test]
fn storage_is_thread_local_and_files_are_bounded_and_reclaimed() {
    for fail_reads in [false, true] {
        let mut storage = TrackingStorage {
            live: Rc::new(Cell::new(0)),
            peak: Rc::new(Cell::new(0)),
            fail_reads,
        };
        let data = vec![0.0; 500 * 8];
        let cfg = HierarchicalSuperKMeansConfig {
            max_leaf_size: 10,
            ..Default::default()
        };
        let mut model = HierarchicalSuperKMeans::with_config(8, cfg);
        let result = model.train_spillable(&mut Matrix::new(&data, 500, 8), &mut storage, OPTIONS);
        assert_eq!(result.is_err(), fail_reads);
        assert_eq!(storage.live.get(), 0);
        assert!(storage.peak.get() <= 3);
    }
}

#[test]
fn insufficient_budget_and_invalid_inputs_return_errors() {
    let mut flat = SuperKMeans::new(4, 8);
    let data = vec![0.0; 8 * 8];
    assert!(
        flat.train_spillable(
            &mut Matrix::new(&data, 8, 8),
            &mut FileTempStorage,
            SpillOptions { memory_budget: 1 }
        )
        .is_err()
    );
    assert!(
        flat.train_spillable(
            &mut Matrix::new(&data[..16], 2, 8),
            &mut FileTempStorage,
            OPTIONS
        )
        .is_err()
    );
    assert!(
        flat.train_spillable(&mut Matrix::new(&[], 0, 8), &mut FileTempStorage, OPTIONS)
            .is_err()
    );
    assert!(
        flat.train_spillable(
            &mut Matrix::new(&data, 16, 4),
            &mut FileTempStorage,
            OPTIONS
        )
        .is_err()
    );
}
