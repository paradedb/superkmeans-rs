//! Cluster distribution benchmark on the Cohere Embed-V3 1M dataset.
//!
//! 1. **Full-Dataset Baseline (`sample = 100%`, zero sampling noise):**
//!    Single baseline run at `1.0` sampling ratio (`n_sample = n`,
//!    `leaf_size = full_leaf_size`) to evaluate the split algorithm's
//!    intrinsic cluster size distribution without sampling variance.
//! 2. **Training Sample Sweep:**
//!    Sweeps `train_fraction` from `5%` to `100%` for a fixed `full_leaf_size`.
//!
//! Usage:
//!   cargo bench --bench cluster_distribution -- [n] [full_leaf_size]

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use superkmeans::{
    ClusterBalanceStats, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeans,
    SuperKMeansConfig, TicToc,
};

const COHERE_N: usize = 1_000_000;
const COHERE_D: usize = 1024;
const DEFAULT_FULL_LEAF_SIZE: usize = 200;

fn load_cohere(rows: usize) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/data_cohere_1m.bin");
    let expected_bytes = COHERE_N * COHERE_D * size_of::<f32>();
    let meta = std::fs::metadata(&path).unwrap_or_else(|e| {
        panic!(
            "missing Cohere 1M dataset at {} ({e}). Run: uv run --script scripts/download_cohere.py",
            path.display()
        );
    });
    assert_eq!(
        meta.len() as usize,
        expected_bytes,
        "unexpected size for {}: got {} bytes, want {}",
        path.display(),
        meta.len(),
        expected_bytes
    );
    assert!(rows <= COHERE_N, "dataset has only {COHERE_N} rows");

    let mut file = File::open(&path).expect("open cohere dataset");
    let mut floats = vec![0.0f32; rows * COHERE_D];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            floats.as_mut_ptr().cast::<u8>(),
            rows * COHERE_D * size_of::<f32>(),
        )
    };
    file.read_exact(bytes).expect("read cohere dataset");
    floats
}

fn strided_sample(base: &[f32], n: usize, d: usize, n_sample: usize) -> Vec<f32> {
    let n_sample = n_sample.clamp(1, n);
    let stride = (n / n_sample).max(1);
    let mut out = Vec::with_capacity(n_sample * d);
    for i in (0..n).step_by(stride).take(n_sample) {
        out.extend_from_slice(&base[i * d..(i + 1) * d]);
    }
    out
}

struct DistributionSummary {
    balance: ClusterBalanceStats,
    empty: usize,
    p01: usize,
    p10: usize,
    p25: usize,
    p50: usize,
    p75: usize,
    p90: usize,
    p99: usize,
}

fn summarize(sizes: &[u32]) -> DistributionSummary {
    let n_clusters = sizes.len();
    let total: usize = sizes.iter().map(|&s| s as usize).sum();
    let mean = total as f32 / n_clusters.max(1) as f32;

    let mut sorted: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
    sorted.sort_unstable();

    let empty = sorted.iter().filter(|&&s| s == 0).count();
    let mut log_sum = 0.0_f32;
    let mut non_zero = 0usize;
    let mut sq_sum = 0.0_f64;
    for &s in &sorted {
        if s > 0 {
            log_sum += (s as f32).ln();
            non_zero += 1;
        }
        sq_sum += (s as f64) * (s as f64);
    }
    let geometric_mean = if non_zero > 0 {
        (log_sum / non_zero as f32).exp()
    } else {
        0.0
    };
    let stdev = ((sq_sum / n_clusters.max(1) as f64) - (mean as f64) * (mean as f64))
        .max(0.0)
        .sqrt() as f32;
    let cv = if mean != 0.0 { stdev / mean } else { 0.0 };

    let pct = |q: f64| -> usize {
        if sorted.is_empty() {
            return 0;
        }
        let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[idx]
    };

    DistributionSummary {
        balance: ClusterBalanceStats {
            mean,
            geometric_mean,
            stdev,
            cv,
            min: *sorted.first().unwrap_or(&0),
            max: *sorted.last().unwrap_or(&0),
        },
        empty,
        p01: pct(0.01),
        p10: pct(0.10),
        p25: pct(0.25),
        p50: pct(0.50),
        p75: pct(0.75),
        p90: pct(0.90),
        p99: pct(0.99),
    }
}

fn summarize_assignments(assignments: &[u32], n_clusters: usize) -> DistributionSummary {
    let mut sizes = vec![0u32; n_clusters];
    for &c in assignments {
        sizes[c as usize] += 1;
    }
    summarize(&sizes)
}

fn bench_full_sample_baseline(base: &[f32], n: usize, full_leaf_size: usize) {
    let target_k = n.div_ceil(full_leaf_size);
    println!(
        "\n=== 1. Full-Sample Baseline (sample = 100%, n = {n}, leaf_size = {full_leaf_size}, target_k = {target_k}) ==="
    );
    println!(
        "{:<14} {:>9} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>8}",
        "stage",
        "leaf_size",
        "tgt_k",
        "act_k",
        "mean",
        "min",
        "p01",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p99",
        "max",
        "p90/p10",
        "cv",
        "empty",
        "train_ms"
    );
    println!("{}", "-".repeat(160));

    let print_baseline_row = |stage: &str, act_k: usize, s: &DistributionSummary, train_ms: f64| {
        let p90_p10 = if s.p10 > 0 {
            format!("{:.2}x", s.p90 as f64 / s.p10 as f64)
        } else {
            "inf".to_string()
        };
        println!(
            "{:<14} {:>9} {:>8} {:>8} {:>8.1} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6.3} {:>6} {:>8.1}",
            stage,
            full_leaf_size,
            target_k,
            act_k,
            s.balance.mean,
            s.balance.min,
            s.p01,
            s.p10,
            s.p25,
            s.p50,
            s.p75,
            s.p90,
            s.p99,
            s.balance.max,
            p90_p10,
            s.balance.cv,
            s.empty,
            train_ms,
        );
    };

    // 1. Flat SuperKMeans with use_aggressive_split = true
    {
        let mut timer = TicToc::new();
        let cfg = SuperKMeansConfig {
            angular: true,
            suppress_warnings: true,
            use_aggressive_split: true,
            ..Default::default()
        };
        let mut skm = SuperKMeans::with_config(target_k, COHERE_D, cfg);
        timer.tic();
        let centroids = skm.train(&base[..n * COHERE_D], n);
        timer.toc();
        let train_ms = timer.milliseconds();

        let train_summary = summarize(&skm.cluster_sizes);
        let assignments = skm.assign(&base[..n * COHERE_D], &centroids, n);
        let assign_summary = summarize_assignments(&assignments, target_k);

        print_baseline_row("flat-train", target_k, &train_summary, train_ms);
        print_baseline_row("flat-assigned", target_k, &assign_summary, train_ms);
        println!("{}", "-".repeat(160));
    }

    // 2. HierarchicalSuperKMeans
    {
        let mut timer = TicToc::new();
        let mut cfg = HierarchicalSuperKMeansConfig {
            max_leaf_size: full_leaf_size,
            ..Default::default()
        };
        cfg.base.angular = true;
        cfg.base.suppress_warnings = true;
        let mut hskm = HierarchicalSuperKMeans::with_config(COHERE_D, cfg);
        timer.tic();
        let centroids = hskm.train(&base[..n * COHERE_D], n);
        timer.toc();
        let train_ms = timer.milliseconds();

        let actual_k = hskm.base.n_clusters;
        let tree_summary = summarize(&hskm.base.cluster_sizes);
        let assignments = hskm.assign(&base[..n * COHERE_D], &centroids, n);
        let assign_summary = summarize_assignments(&assignments, actual_k);

        print_baseline_row("hier-train", actual_k, &tree_summary, train_ms);
        print_baseline_row("hier-assigned", actual_k, &assign_summary, train_ms);
        println!("{}", "-".repeat(160));
    }
}

fn bench_sample_sweep(base: &[f32], n: usize, full_leaf_size: usize) {
    let expected_k = n.div_ceil(full_leaf_size);
    println!(
        "\n=== 2. Training Sample Sweep (n = {n}, full_leaf_size = {full_leaf_size}, expected_k = {expected_k}) ==="
    );
    println!(
        "{:<10} {:>8} {:>9} {:>10} {:>8} {:>8} {:>9} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>8}",
        "stage",
        "sample%",
        "n_sample",
        "leaf_size",
        "exp_k",
        "act_k",
        "exp_size",
        "mean",
        "min",
        "p10",
        "p50",
        "p90",
        "max",
        "p90/p10",
        "cv",
        "empty",
        "train_ms"
    );
    println!("{}", "-".repeat(144));

    for train_fraction in [0.05_f64, 0.10, 0.25, 0.32, 0.50, 1.00] {
        let sample_leaf_size = ((full_leaf_size as f64) * train_fraction).round().max(1.0) as usize;
        let n_sample = (expected_k * sample_leaf_size).min(n);
        let actual_fraction = n_sample as f64 / n as f64;
        let sample = strided_sample(base, n, COHERE_D, n_sample);

        let mut timer = TicToc::new();
        let mut cfg = HierarchicalSuperKMeansConfig {
            max_leaf_size: sample_leaf_size,
            ..Default::default()
        };
        cfg.base.angular = true;
        cfg.base.suppress_warnings = true;
        let mut hskm = HierarchicalSuperKMeans::with_config(COHERE_D, cfg);
        timer.tic();
        let centroids = hskm.train_owned(sample, n_sample);
        timer.toc();

        let actual_k = hskm.base.n_clusters;
        let train_summary = summarize(&hskm.base.cluster_sizes);
        let assignments = hskm.assign(base, &centroids, n);
        let assign_summary = summarize_assignments(&assignments, actual_k);

        for (stage, exp_size, s) in [
            ("train", sample_leaf_size as f64, &train_summary),
            ("assigned", full_leaf_size as f64, &assign_summary),
        ] {
            let p90_p10 = if s.p10 > 0 {
                format!("{:.2}x", s.p90 as f64 / s.p10 as f64)
            } else {
                "inf".to_string()
            };
            println!(
                "{:<10} {:>7.1}% {:>9} {:>10} {:>8} {:>8} {:>9.1} {:>8.1} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6.3} {:>6} {:>8.1}",
                stage,
                actual_fraction * 100.0,
                n_sample,
                sample_leaf_size,
                expected_k,
                actual_k,
                exp_size,
                s.balance.mean,
                s.balance.min,
                s.p10,
                s.p50,
                s.p90,
                s.balance.max,
                p90_p10,
                s.balance.cv,
                s.empty,
                timer.milliseconds(),
            );
        }
        println!("{}", "-".repeat(144));
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).filter(|arg| arg != "--bench").collect();
    let mode = args.first().map(String::as_str).unwrap_or("baseline");
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(COHERE_N);
    let full_leaf_size: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FULL_LEAF_SIZE);

    println!("Loading Cohere 1M dataset (first {n} rows, d={COHERE_D})...");
    let base = load_cohere(n);

    match mode {
        "baseline" => bench_full_sample_baseline(&base, n, full_leaf_size),
        "sample" => bench_sample_sweep(&base, n, full_leaf_size),
        "all" => {
            bench_full_sample_baseline(&base, n, full_leaf_size);
            bench_sample_sweep(&base, n, full_leaf_size);
        }
        other => panic!("unknown mode '{other}'; expected 'baseline', 'sample', or 'all'"),
    }
}
