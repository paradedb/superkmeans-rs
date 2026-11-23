//! `SuperKMeansConfig::n_threads` bounds the cores a model keeps busy.
//!
//! The measure is process CPU time over wall time, so this file holds one
//! test: a second test running beside it would be counted too.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    target_pointer_width = "64"
))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use superkmeans::{
    HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, IterDataset, Matrix, SuperKMeans,
    SuperKMeansConfig, make_blobs,
};

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

unsafe extern "C" {
    fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
}

const CLOCK_PROCESS_CPUTIME_ID: i32 = if cfg!(target_os = "macos") { 12 } else { 2 };

/// CPU seconds this process has used, summed over all of its threads.
fn process_cpu_secs() -> f64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime failed");
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// Average number of cores the process kept busy while `work` ran.
fn cores_busy(work: impl FnOnce()) -> f64 {
    let (cpu, wall) = (process_cpu_secs(), Instant::now());
    work();
    (process_cpu_secs() - cpu) / wall.elapsed().as_secs_f64()
}

#[test]
fn one_thread_model_keeps_one_core_busy() {
    // d >= 128 and k > 256 put training on the pruning kernels as well.
    let (n, d, k) = (8_000usize, 128usize, 300usize);
    let data = make_blobs(n, d, 60, true, 1.0, 10.0, 42);
    let matrix = || Matrix::new(&data, n, d);
    let stream = || IterDataset::new(Matrix::new(&data, n, d).chunks(4_096));

    let flat = SuperKMeans::with_config(
        d,
        SuperKMeansConfig {
            n_threads: 1,
            iters: 3,
            early_termination: false,
            suppress_warnings: true,
            ..Default::default()
        },
    );
    let mut hier_cfg = HierarchicalSuperKMeansConfig {
        target_leaf_size: 50,
        ..Default::default()
    };
    hier_cfg.base.n_threads = 1;
    let hier = HierarchicalSuperKMeans::with_config(d, hier_cfg);

    let trained = flat.train(&mut matrix(), k).unwrap();
    let centroids = Matrix::new(&trained.centroids, k, d);
    let meso = hier.train_meso(&mut stream(), n).unwrap();
    let meso_blocks: Vec<Vec<f32>> = (0..meso.n_meso())
        .map(|m| {
            meso.meso_rows(m)
                .iter()
                .flat_map(|&r| data[r as usize * d..(r as usize + 1) * d].iter().copied())
                .collect()
        })
        .collect();

    let methods: [(&str, &dyn Fn()); 9] = [
        ("SuperKMeans::train on a matrix", &|| {
            flat.train(&mut matrix(), k).unwrap();
        }),
        ("SuperKMeans::train on a scan", &|| {
            flat.train(&mut stream(), k).unwrap();
        }),
        ("SuperKMeans::assign", &|| {
            flat.assign(matrix(), centroids);
        }),
        ("SuperKMeans::assign_soar", &|| {
            flat.assign_soar(matrix(), centroids, 1.0);
        }),
        ("SuperKMeans::assign_training_points", &|| {
            flat.assign_training_points(matrix(), &trained);
        }),
        ("HierarchicalSuperKMeans::train", &|| {
            hier.train(&mut matrix());
        }),
        ("HierarchicalSuperKMeans::train_meso", &|| {
            hier.train_meso(&mut stream(), n).unwrap();
        }),
        ("HierarchicalSuperKMeans::assign", &|| {
            hier.assign(matrix(), centroids);
        }),
        ("train_meso_cluster from four threads", &|| {
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| {
                        while let Some(block) =
                            meso_blocks.get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            hier.train_meso_cluster(&mut Matrix::new(block, block.len() / d, d));
                        }
                    });
                }
            });
        }),
    ];

    for (method, work) in methods {
        let busy = cores_busy(work);
        // One pool thread caps this at 1. The slack covers the callers'
        // own bookkeeping, and is far below the 2 a second thread would add.
        assert!(busy < 1.5, "{method} kept {busy:.2} cores busy");
    }
}
