//! Criterion benches for SuperKMeans / HierarchicalSuperKMeans on Cohere Embed-V3.
//!
//! Dataset: first 1M vectors of `Cohere/msmarco-v2.1-embed-english-v3` (d=1024),
//! matching the C++ cohere bench source at a RAM-friendly scale.
//! Download with: `uv run --script scripts/download_cohere.py`
//!
//! Run:
//!   cargo bench --bench cohere -- train
//!   cargo bench --bench cohere -- assign
//!   cargo bench --bench cohere

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::path::PathBuf;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use superkmeans::{HierarchicalSuperKMeans, SuperKMeans};

const N: usize = 1_000_000;
const D: usize = 1024;
const K: usize = 100_000;

fn data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/data_cohere_1m.bin")
}

fn load_cohere_1m() -> &'static [f32] {
    static DATA: OnceLock<Vec<f32>> = OnceLock::new();
    DATA.get_or_init(|| {
        let path = data_path();
        let expected_bytes = N * D * size_of::<f32>();
        let meta = std::fs::metadata(&path).unwrap_or_else(|e| {
            panic!(
                "missing Cohere 1M dataset at {} ({e}). Run: uv run --script scripts/download_cohere.py",
                path.display()
            );
        });
        assert_eq!(
            meta.len() as usize,
            expected_bytes,
            "unexpected size for {}: got {} bytes, want {} (n={N}, d={D})",
            path.display(),
            meta.len(),
            expected_bytes
        );

        let mut file = File::open(&path).expect("open cohere dataset");
        let mut floats = vec![0.0f32; N * D];
        // Raw little-endian f32 binary (same layout as SuperKMeans setup_data.py).
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(floats.as_mut_ptr().cast::<u8>(), expected_bytes)
        };
        file.read_exact(bytes).expect("read cohere dataset");
        floats
    })
    .as_slice()
}

fn bench_train(c: &mut Criterion) {
    let data = load_cohere_1m();

    let mut group = c.benchmark_group("train");
    // Criterion's minimum sample size is 10.
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function(
        BenchmarkId::new("SuperKMeans", format!("n{N}_k{K}_d{D}")),
        |b| {
            b.iter(|| {
                let mut kmeans = SuperKMeans::new(K, D);
                let centroids = kmeans.train(black_box(data), black_box(N));
                black_box(centroids);
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("HierarchicalSuperKMeans", format!("n{N}_k{K}_d{D}")),
        |b| {
            b.iter(|| {
                let mut kmeans = HierarchicalSuperKMeans::new(K, D);
                let centroids = kmeans.train(black_box(data), black_box(N));
                black_box(centroids);
            });
        },
    );

    group.finish();
}

fn bench_assign(c: &mut Criterion) {
    let data = load_cohere_1m();

    // Train once outside the timed loop; assign is what we measure.
    let flat_centroids = {
        let mut kmeans = SuperKMeans::new(K, D);
        kmeans.train(data, N)
    };
    let hier_centroids = {
        let mut kmeans = HierarchicalSuperKMeans::new(K, D);
        kmeans.train(data, N)
    };

    let mut group = c.benchmark_group("assign");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function(
        BenchmarkId::new("SuperKMeans", format!("n{N}_k{K}_d{D}")),
        |b| {
            b.iter(|| {
                let kmeans = SuperKMeans::new(K, D);
                let assignments =
                    kmeans.assign(black_box(data), black_box(&flat_centroids), black_box(N));
                black_box(assignments);
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("HierarchicalSuperKMeans", format!("n{N}_k{K}_d{D}")),
        |b| {
            b.iter(|| {
                let kmeans = HierarchicalSuperKMeans::new(K, D);
                let assignments =
                    kmeans.assign(black_box(data), black_box(&hier_centroids), black_box(N));
                black_box(assignments);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_train, bench_assign);
criterion_main!(benches);
