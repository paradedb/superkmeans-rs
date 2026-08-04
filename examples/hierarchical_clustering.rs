//! Hierarchical SuperKMeans (HBC) example.
//!
//! By default the root uses a √K meso split and deeper nodes finish toward
//! `max_leaf_size`. Pass an optional branching factor to force fixed fan-out.

use std::env;
use std::process::ExitCode;

use superkmeans::{
    HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeans, TicToc, make_blobs,
};

fn print_usage(program: &str) {
    println!(
        "Usage: {} [n] [d] [max_leaf_size] [branching_factor]\n  \
         n: Number of vectors (default: 100000)\n  \
         d: Dimensionality (default: 128)\n  \
         max_leaf_size: Stop splitting below this size (default: 256)\n  \
         branching_factor: Optional fixed fan-out; omit for HBC √K meso default\n\n\
         Example:\n  {} 50000 64 128\n  {} 50000 64 128 16",
        program, program, program
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    let mut n: usize = 100_000;
    let mut d: usize = 128;
    let mut max_leaf_size: usize = 256;
    let mut branching_factor: Option<usize> = None;

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
        max_leaf_size = args[3].parse().unwrap_or(max_leaf_size);
    }
    if args.len() > 4 {
        branching_factor = args[4].parse().ok();
    }

    println!(
        "Parameters: n={n}, d={d}, max_leaf_size={max_leaf_size}, branching_factor={branching_factor:?}"
    );
    println!("Generating {n} vectors with d={d}");
    let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);

    let mut cfg = HierarchicalSuperKMeansConfig {
        max_leaf_size,
        branching_factor,
        ..Default::default()
    };
    cfg.base.verbose = env::var("SUPERKMEANS_VERBOSE").is_ok();
    let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);

    println!("Running HierarchicalSuperKMeans...");
    let mut timer = TicToc::new();
    timer.tic();
    let centroids = kmeans.train(&data, n);
    timer.toc();
    println!("Index built in: {} ms", timer.milliseconds());
    println!(
        "Tree: n_leaves={}, nodes={}, root_size={}",
        kmeans.tree.n_leaves,
        kmeans.tree.nodes.len(),
        kmeans.tree.node(kmeans.tree.root).size()
    );

    let assignments = kmeans.assign(&data, &centroids, n);
    println!("Got {} assignments", assignments.len());

    let stats = SuperKMeans::cluster_balance_stats(&assignments, n, kmeans.tree.n_leaves);
    println!(
        "Leaf balance: mean={:.1} min={} max={} cv={:.3}",
        stats.mean, stats.min, stats.max, stats.cv
    );

    ExitCode::SUCCESS
}
