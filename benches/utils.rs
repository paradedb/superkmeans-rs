//! Criterion benches for the shared numeric kernels in `superkmeans::utils`.
//!
//! One group per kernel, comparing a sequential pass against the rayon form to
//! find where the fork-join starts paying for itself. Training calls these from a
//! non-worker thread, so the parallel numbers include rayon's full entry cost:
//! pushing the job to the pool queue and blocking on a latch.
//!
//! Every case is named for its shape, `n` vectors by `d` dimensions by `k`
//! clusters, in that order and omitting whatever a kernel does not take — so
//! 100000 vectors of 768 dimensions over 1000 clusters reads `n100_000_d768_k1000`,
//! and a kernel that only walks centroids reads `d768_k1000`. `p` stands in for
//! `k` where a kernel reads a prefix of each vector rather than clustering it. A
//! filter therefore picks out one dimensionality or width across every group.
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
    squared_norms_partial_parallel, squared_norms_partial_sequential,
    sum_rows_by_assignment_parallel, sum_rows_by_assignment_sequential,
};

const DIMS: [usize; 3] = [128, 768, 1536];
const CLUSTERS: [usize; 5] = [2, 8, 32, 128, 1024];

/// Shared timing settings: these kernels run in microseconds, so the window has
/// to be long enough to ride out scheduler noise.
fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
}

/// Id for a shape that only touches centroids: `d768_k1000`.
fn centroid_id(d: usize, k: usize) -> String {
    format!("d{}_k{}", readable(d), readable(k))
}

/// Id for a shape that also reads a vector set: `n100_000_d768_k1000`.
fn vector_id(n: usize, d: usize, k: usize) -> String {
    format!("n{}_{}", readable(n), centroid_id(d, k))
}

/// Id for a shape that reads only a prefix of each vector: `n100_000_d768_p96`.
fn prefix_id(n: usize, d: usize, partial_d: usize) -> String {
    format!("n{}_d{}_p{}", readable(n), readable(d), readable(partial_d))
}

/// Group a magnitude into thousands once it gets long enough to misread, at the
/// same five-digit threshold as clippy's `unreadable_literal`.
fn readable(x: usize) -> String {
    let digits = x.to_string();
    if digits.len() < 5 {
        return digits;
    }
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.char_indices() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push('_');
        }
        grouped.push(digit);
    }
    grouped
}

fn bench_centroid_shift(c: &mut Criterion) {
    let mut group = c.benchmark_group("centroid_shift");
    configure(&mut group);

    for d in DIMS {
        for k in CLUSTERS {
            // The shift between the two is arbitrary; only its cost is measured.
            let new_centroids = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            let prev_centroids = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 2);
            let id = centroid_id(d, k);

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

/// `partial_d` is swept as a fraction of `d` as well as at `d` itself, since a
/// prefix norm reads a strided slice of the matrix: it sums fewer elements than it
/// pulls cache lines for, so it is worth checking whether the crossover follows
/// the elements summed or the footprint walked.
fn bench_squared_norms(c: &mut Criterion) {
    const VECTORS: [usize; 5] = [8, 64, 512, 4096, 32768];

    let mut group = c.benchmark_group("squared_norms");
    configure(&mut group);

    for d in DIMS {
        for n in VECTORS {
            let vectors = make_blobs(n, d, n.min(8), false, 1.0, 10.0, 1);

            for partial_d in [d, d / 8] {
                let id = prefix_id(n, d, partial_d);

                group.throughput(Throughput::Elements((n * partial_d) as u64));

                group.bench_function(BenchmarkId::new("sequential", &id), |b| {
                    b.iter(|| {
                        black_box(squared_norms_partial_sequential(
                            black_box(&vectors),
                            n,
                            d,
                            partial_d,
                        ))
                    });
                });

                group.bench_function(BenchmarkId::new("parallel", &id), |b| {
                    b.iter(|| {
                        black_box(squared_norms_partial_parallel(
                            black_box(&vectors),
                            n,
                            d,
                            partial_d,
                        ))
                    });
                });
            }
        }
    }

    group.finish();
}

/// This kernel splits its output over rayon while its cost tracks its input, so
/// the two are swept independently: `k` sets how many bands of clusters there are
/// to hand out, `n` how much work there is to cover the dispatch with. Both
/// crossovers land on `k * d` and `n * d` rather than on either factor, so the
/// shapes pair a small `d` with a large `k` and vice versa to check that.
fn bench_sum_rows_by_assignment(c: &mut Criterion) {
    const SHAPES: [(usize, usize); 4] = [(768, 16), (768, 64), (128, 96), (128, 384)];
    const SAMPLES: [usize; 4] = [256, 1024, 8192, 65536];

    let mut group = c.benchmark_group("sum_rows_by_assignment");
    configure(&mut group);

    for n in SAMPLES {
        for (d, k) in SHAPES {
            if k > n {
                continue;
            }
            let vectors = make_blobs(n, d, k, false, 1.0, 10.0, 1);
            // Round-robin, so every cluster draws the same share. A skewed split
            // would leave one band doing all the work while the rest idle.
            let assignments: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
            let mut centroids = vec![0.0_f32; k * d];
            let mut cluster_sizes = vec![0_u32; k];
            let id = vector_id(n, d, k);

            group.throughput(Throughput::Elements((n * d) as u64));

            group.bench_function(BenchmarkId::new("sequential", &id), |b| {
                b.iter(|| {
                    sum_rows_by_assignment_sequential(
                        black_box(&vectors),
                        black_box(&assignments),
                        black_box(&mut centroids),
                        black_box(&mut cluster_sizes),
                        d,
                    )
                });
            });

            group.bench_function(BenchmarkId::new("parallel", &id), |b| {
                b.iter(|| {
                    sum_rows_by_assignment_parallel(
                        black_box(&vectors),
                        black_box(&assignments),
                        black_box(&mut centroids),
                        black_box(&mut cluster_sizes),
                        d,
                    )
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
        for k in CLUSTERS {
            let mut rows = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            // Counts of 1 keep the scale idempotent, so the same buffer survives
            // the timing loop instead of decaying towards denormals. The work per
            // row — one reciprocal and `d` multiplies — is unchanged.
            let counts = vec![1_u32; k];
            let id = centroid_id(d, k);

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
        for k in CLUSTERS {
            // Normalization is idempotent, so repeated passes over one buffer stay
            // representative: after the first the rows are already unit-norm.
            let mut rows = make_blobs(k, d, k.min(8), false, 1.0, 10.0, 1);
            let id = centroid_id(d, k);

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
    bench_squared_norms,
    bench_sum_rows_by_assignment,
    bench_mean_rows_by_count,
    bench_normalize_rows_l2
);
criterion_main!(benches);
