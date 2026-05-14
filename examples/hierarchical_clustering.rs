//! Rust port of `SuperKMeans/examples/hierarchical_clustering.cpp`.

use std::env;
use std::process::ExitCode;

use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, TicToc, make_blobs};

fn print_usage(program: &str) {
    println!(
        "Usage: {} [n] [d] [k]\n  n: Number of vectors (default: 1000000)\n  d: Dimensionality (default: 768)\n  k: Number of clusters (default: 10000)\n\nExample:\n  {} 500000 512 100",
        program, program
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    let mut n: usize = 1_000_000;
    let mut d: usize = 768;
    let mut k: usize = 10_000;

    if args.len() > 1 {
        if args[1] == "-h" || args[1] == "--help" {
            print_usage(&args[0]);
            return ExitCode::SUCCESS;
        }
        n = args[1].parse().unwrap_or(n);
    }
    if args.len() > 2 {
        d = args[2].parse().unwrap_or(d);
    }
    if args.len() > 3 {
        k = args[3].parse().unwrap_or(k);
    }

    println!("Parameters: n={}, d={}, k={}", n, d, k);
    println!("Generating {} vectors with d={}", n, d);
    let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);

    let mut cfg = HierarchicalSuperKMeansConfig::default();
    cfg.base.verbose = env::var("SUPERKMEANS_VERBOSE").is_ok();
    let mut kmeans = HierarchicalSuperKMeans::with_config(k, d, cfg);

    println!("Running HierarchicalSuperKMeans with {} clusters...", k);
    let mut timer = TicToc::new();
    timer.tic();
    let centroids = kmeans.train(&data, n);
    timer.toc();
    let construction_time_ms = timer.milliseconds();
    println!("Index built in: {} ms", construction_time_ms);

    let assignments = kmeans.assign(&data, &centroids, n);
    println!("Got {} assignments", assignments.len());

    ExitCode::SUCCESS
}
