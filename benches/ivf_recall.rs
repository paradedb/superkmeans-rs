//! IVF recall harness: trains on a sample, assigns the full base set to the
//! resulting leaves, and reports recall@100 against exact ground truth as a
//! function of how many lists a query probes.
//!
//! Usage:
//!   cargo bench --bench ivf_recall -- [n] [queries] [train_fraction]
//!
//! Falls back to synthetic blobs if the Cohere dataset is missing.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use rayon::prelude::*;
use superkmeans::distance::l2_squared;
use superkmeans::{
    HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeans, TicToc, make_blobs,
};

const COHERE_N: usize = 1_000_000;
const COHERE_D: usize = 1024;
const TOP_K: usize = 100;
const MAX_LEAF_SIZE: usize = 10;

/// Read the first `rows` vectors of the Cohere dump, or `None` if it is absent.
fn load_cohere(rows: usize) -> Option<(Vec<f32>, usize)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/data_cohere_1m.bin");
    let meta = std::fs::metadata(&path).ok()?;
    if meta.len() as usize != COHERE_N * COHERE_D * size_of::<f32>() {
        return None;
    }
    assert!(rows <= COHERE_N, "dataset has only {COHERE_N} rows");

    let mut file = File::open(&path).ok()?;
    let mut floats = vec![0.0f32; rows * COHERE_D];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            floats.as_mut_ptr().cast::<u8>(),
            rows * COHERE_D * size_of::<f32>(),
        )
    };
    file.read_exact(bytes).ok()?;
    Some((floats, COHERE_D))
}

/// Exact top-`TOP_K` neighbors of each query, by brute force.
fn ground_truth(
    base: &[f32],
    n: usize,
    queries: &[f32],
    n_queries: usize,
    d: usize,
) -> Vec<Vec<u32>> {
    (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let query = &queries[q * d..(q + 1) * d];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (l2_squared(query, &base[i * d..(i + 1) * d]), i as u32))
                .collect();
            scored.select_nth_unstable_by(TOP_K - 1, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(TOP_K);
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let n_queries: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let train_fraction: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let (all, d) = match load_cohere(n + n_queries) {
        Some(v) => {
            println!("dataset: Cohere 1M (first {} rows)", n + n_queries);
            v
        }
        None => {
            println!("dataset: synthetic blobs (Cohere dump not found)");
            let d = 768;
            (make_blobs(n + n_queries, d, 200, true, 1.0, 10.0, 42), d)
        }
    };
    let (base, queries) = all.split_at(n * d);
    println!(
        "n={n} d={d} queries={n_queries} train_fraction={train_fraction} \
         max_leaf_size={MAX_LEAF_SIZE} top_k={TOP_K}"
    );

    print!("computing exact ground truth... ");
    let mut timer = TicToc::new();
    timer.tic();
    let truth = ground_truth(base, n, queries, n_queries, d);
    timer.toc();
    println!("{:.1}s", timer.milliseconds() / 1000.0);

    // Caller-side sampling: take a strided sample so it spans the whole set.
    let stride = (1.0 / train_fraction).round().max(1.0) as usize;
    let mut train_set = Vec::with_capacity((n / stride + 1) * d);
    for i in (0..n).step_by(stride) {
        train_set.extend_from_slice(&base[i * d..(i + 1) * d]);
    }
    let n_train = train_set.len() / d;

    // max_leaf_size applies to the training sample, so scale it down to land on
    // the same number of leaves the full set would have produced.
    let leaf_size = (MAX_LEAF_SIZE / stride).max(1);
    let mut cfg = HierarchicalSuperKMeansConfig {
        max_leaf_size: leaf_size,
        ..Default::default()
    };
    cfg.base.suppress_warnings = true;

    println!("training on {n_train} rows (stride {stride}, leaf cap {leaf_size})...");
    let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);
    timer.tic();
    let centroids = kmeans.train_owned(train_set, n_train);
    timer.toc();
    let build_secs = timer.milliseconds() / 1000.0;
    let n_lists = kmeans.tree.n_leaves;
    println!("built {n_lists} lists in {build_secs:.2}s");

    // Assign the full base set, which is what actually fills the posting lists.
    timer.tic();
    let assignments = kmeans.assign(base, &centroids, n);
    timer.toc();
    println!(
        "assigned {n} vectors in {:.2}s",
        timer.milliseconds() / 1000.0
    );

    let balance = SuperKMeans::cluster_balance_stats(&assignments, n, n_lists);
    println!(
        "list sizes: mean={:.1} min={} max={} cv={:.3}",
        balance.mean, balance.min, balance.max, balance.cv
    );

    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_lists];
    for (i, &c) in assignments.iter().enumerate() {
        lists[c as usize].push(i as u32);
    }

    println!(
        "\n{:<8} {:<10} {:<12} recall@{TOP_K}",
        "nprobe", "candidates", "%_of_base"
    );
    for nprobe in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        if nprobe > n_lists {
            break;
        }
        let (recall_sum, cand_sum) = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let query = &queries[q * d..(q + 1) * d];

                // Rank the lists by centroid distance and take the closest few.
                let mut ranked: Vec<(f32, u32)> = (0..n_lists)
                    .map(|c| (l2_squared(query, &centroids[c * d..(c + 1) * d]), c as u32))
                    .collect();
                ranked.select_nth_unstable_by(nprobe - 1, |a, b| a.0.total_cmp(&b.0));
                ranked.truncate(nprobe);

                // Exact rescore of everything in those lists.
                let mut scored: Vec<(f32, u32)> = ranked
                    .iter()
                    .flat_map(|&(_, c)| lists[c as usize].iter())
                    .map(|&i| {
                        (
                            l2_squared(query, &base[i as usize * d..(i as usize + 1) * d]),
                            i,
                        )
                    })
                    .collect();
                let candidates = scored.len();

                let take = TOP_K.min(candidates);
                if take > 0 {
                    scored.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
                    scored.truncate(take);
                }
                let found = scored.iter().filter(|(_, i)| truth[q].contains(i)).count();
                (found as f64 / TOP_K as f64, candidates as f64)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        let mean_cand = cand_sum / n_queries as f64;
        println!(
            "{nprobe:<8} {mean_cand:<10.0} {:<12.2} {:.4}",
            100.0 * mean_cand / n as f64,
            recall_sum / n_queries as f64
        );
    }
}
