//! Criterion benches for the shared numeric kernels in `superkmeans::utils`.
//!
//! One group per kernel, comparing a sequential pass against rayon's per-row
//! parallelism to find where the fork-join starts paying for itself. Training
//! calls these from a non-worker thread, so the parallel numbers include rayon's
//! full entry cost: pushing the job to the pool queue and blocking on a latch.
//!
//! Run:
//!   cargo bench --bench utils
//!   cargo bench --bench utils -- centroid_shift
//!   cargo bench --bench utils -- d768

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use superkmeans::utils::{
    centroid_shift_parallel, centroid_shift_sequential, make_blobs, mean_rows_by_count_parallel,
    mean_rows_by_count_sequential, normalize_rows_l2_parallel, normalize_rows_l2_sequential,
};

const DIMS: [usize; 3] = [128, 768, 1536];
const ROWS: [usize; 5] = [2, 8, 32, 128, 1024];

/// Shared timing settings: these kernels run in microseconds, so the window has
/// to be long enough to ride out scheduler noise.
fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
}

fn bench_centroid_shift(c: &mut Criterion) {
    let mut group = c.benchmark_group("centroid_shift");
    configure(&mut group);

    for d in DIMS {
        for k in ROWS {
            // The shift between the two is arbitrary; only its cost is measured.
            let new_centroids = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            let prev_centroids = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 2);
            let id = format!("d{d}_k{k}");

            group.throughput(Throughput::Elements((k * d) as u64));

            group.bench_function(BenchmarkId::new("sequential", &id), |b| {
                b.iter(|| {
                    black_box(centroid_shift_sequential(
                        black_box(&new_centroids),
                        black_box(&prev_centroids),
                        k,
                        d,
                    ))
                });
            });

            group.bench_function(BenchmarkId::new("parallel", &id), |b| {
                b.iter(|| {
                    black_box(centroid_shift_parallel(
                        black_box(&new_centroids),
                        black_box(&prev_centroids),
                        k,
                        d,
                    ))
                });
            });
        }
    }

    group.finish();
}

fn bench_mean_rows_by_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_rows_by_count");
    configure(&mut group);

    for d in DIMS {
        for k in ROWS {
            let mut rows = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            // Counts of 1 keep the scale idempotent, so the same buffer survives
            // the timing loop instead of decaying towards denormals. The work per
            // row — one reciprocal and `d` multiplies — is unchanged.
            let counts = vec![1_u32; k];
            let id = format!("d{d}_k{k}");

            group.throughput(Throughput::Elements((k * d) as u64));

            group.bench_function(BenchmarkId::new("sequential", &id), |b| {
                b.iter(|| {
                    mean_rows_by_count_sequential(black_box(&mut rows), black_box(&counts), d)
                });
            });

            group.bench_function(BenchmarkId::new("parallel", &id), |b| {
                b.iter(|| mean_rows_by_count_parallel(black_box(&mut rows), black_box(&counts), d));
            });
        }
    }

    group.finish();
}

fn bench_normalize_rows_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_rows_l2");
    configure(&mut group);

    for d in DIMS {
        for k in ROWS {
            // Normalization is idempotent, so repeated passes over one buffer stay
            // representative: after the first the rows are already unit-norm.
            let mut rows = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            let id = format!("d{d}_k{k}");

            group.throughput(Throughput::Elements((k * d) as u64));

            group.bench_function(BenchmarkId::new("sequential", &id), |b| {
                b.iter(|| normalize_rows_l2_sequential(black_box(&mut rows), k, d));
            });

            group.bench_function(BenchmarkId::new("parallel", &id), |b| {
                b.iter(|| normalize_rows_l2_parallel(black_box(&mut rows), k, d));
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_centroid_shift,
    bench_mean_rows_by_count,
    bench_normalize_rows_l2
);
criterion_main!(benches);
