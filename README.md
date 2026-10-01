# superkmeans-rs

[![Crates.io](https://img.shields.io/crates/v/superkmeans-rs.svg)](https://crates.io/crates/superkmeans-rs)
[![codecov](https://codecov.io/gh/paradedb/superkmeans-rs/graph/badge.svg)](https://codecov.io/gh/paradedb/superkmeans-rs)
[![CI](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml)
[![Documentation](https://docs.rs/superkmeans-rs/badge.svg)](https://docs.rs/superkmeans-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Rust port of **SuperKMeans**: fast k-means clustering for high-dimensional
vector embeddings.

## Overview

`superkmeans-rs` re-implements the SuperKMeans C++ library in pure Rust (no
FFI required). It is designed for clustering large collections of
high-dimensional embeddings — the kind produced by text and image models — and
uses [ADSampling](https://github.com/gaoj0017/ADSampling)-style distance
pruning plus a cache-friendly matrix layout to keep assignment fast as
dimensionality grows.

- **Pure Rust by default** — the SGEMM inner loop runs on
  [`matrixmultiply`](https://crates.io/crates/matrixmultiply); no system BLAS
  is required to build or run.
- **Optional vendor BLAS** — route SGEMM through OpenBLAS or Apple Accelerate
  when you want the last bit of throughput (see [BLAS backends](#blas-backends)).
- **Parallel** — training and assignment are parallelized with
  [`rayon`](https://crates.io/crates/rayon).
- **Hierarchical clustering** — `HierarchicalSuperKMeans` splits into √K meso
  clusters, then re-clusters any group still over a size cap.

> **Note:** the crate is published as `superkmeans-rs` but the library target
> is named `superkmeans`, so you import it as `use superkmeans::...`.

## Installation

```toml
[dependencies]
superkmeans-rs = "0.3"
```

The minimum supported Rust version (MSRV) is **1.89** (required by the AVX512 intrinsics used in the pruning kernels).

## Quick Start

Vectors are passed as a single row-major `&[f32]` slice of length `n * d`
(`n` vectors of dimensionality `d`):

```rust
use superkmeans::{SuperKMeans, SuperKMeansConfig, make_blobs};

// 10,000 vectors of dimensionality 128, drawn from 100 blobs (seed = 42).
let n = 10_000;
let d = 128;
let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);

// Cluster into 100 centroids.
let k = 100;
let mut kmeans = SuperKMeans::with_config(k, d, SuperKMeansConfig::default());

// `train` returns the `k * d` centroids as a flat row-major slice.
let centroids = kmeans.train(&data, n);

// Assign every vector to its nearest centroid.
let assignments: Vec<u32> = kmeans.assign(&data, &centroids, n);
assert_eq!(assignments.len(), n);
```

### Tuning

`SuperKMeansConfig` exposes the knobs from the C++ original — number of
iterations, RNG seed, early-termination tolerances, ADSampling pruning bounds,
and more. Start from the defaults and override what you need:

```rust
use superkmeans::{SuperKMeans, SuperKMeansConfig};

let mut cfg = SuperKMeansConfig::default();
cfg.iters = 20;          // more refinement passes
cfg.seed = 7;            // deterministic runs
cfg.verbose = true;      // log per-iteration statistics

let mut kmeans = SuperKMeans::with_config(1000, 768, cfg);
```

Training runs on the ambient rayon thread pool, so the width comes from
`RAYON_NUM_THREADS` or from calling `train` inside your own
`ThreadPool::install`.

`train` clusters exactly the rows you hand it — there is no sampling knob.
Training on a subset is usually worth it (on 200k Cohere vectors, a 25% sample
halved build time and cost under a point of recall@100); just subsample before
calling `train` and pass the full set to `assign`.

### Streaming training

When the data doesn't fit in memory, train from a replayable stream of
`Matrix` batches instead of one slice. Each Lloyd iteration is one pass over
the stream, plus one pass up front to pick the starting centroids; the only
per-vector state kept between passes is one assignment (4 bytes).

```rust
use superkmeans::{Matrix, SuperKMeans, make_blobs};

let (n, d, k) = (10_000, 128, 100);
let data = make_blobs(n, d, k, true, 1.0, 10.0, 42);

// Any `Clone` iterator of batches works, as long as every clone yields the
// same vectors in the same order. Single vectors (`Matrix::vector`) are fine;
// they get packed into GEMM-sized batches.
let mut kmeans = SuperKMeans::new(k, d);
let centroids = kmeans.train_iter(Matrix::new(&data, n, d).chunks(1_024));
```

Sources that can't be cloned or can fail (a file or index re-read on every
pass) implement `Dataset` and call `train_dataset`, which returns the scan's
error if one occurs.

### Hierarchical clustering

`HierarchicalSuperKMeans` splits the sample into `ceil(sqrt(K))` meso clusters
with `K = ceil(n / max_leaf_size)` (3 iterations), then re-clusters every group
still larger than `max_leaf_size` into `ceil(n_i / max_leaf_size)` clusters
(5 iterations) and repeats that until every cluster fits. `train` returns the
fine centroids. The cluster count is emergent and lands near
`n / max_leaf_size`; read it from `kmeans.base.n_clusters`.

Clusters come out evenly sized without an explicit balance penalty: each split
rebalances its own undersized clusters.

Splits reorder the training set in place rather than copying each child out.
With `train_owned` peak memory is one copy of the training set; `train` must
duplicate the caller's slice before rotating it.

```rust
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, make_blobs};

let n = 100_000;
let d = 256;
let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);

let cfg = HierarchicalSuperKMeansConfig {
    max_leaf_size: 256,
    ..Default::default()
};

let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);
let centroids = kmeans.train(&data, n);
let assignments = kmeans.assign(&data, &centroids, n);
assert_eq!(centroids.len(), kmeans.base.n_clusters * d);
```

#### Streaming hierarchical clustering

Only the meso step needs every vector, and it streams: `train_meso` runs
Lloyd's over a `Dataset` (or `train_meso_iter` over an iterator) and returns
the scan positions grouped by meso cluster. Each meso cluster is about
`sqrt(n * max_leaf_size)` vectors, small enough to load on its own, so the
caller fetches one cluster at a time and hands it to `train_meso_cluster`,
which runs the in-memory splits. Peak memory is one meso cluster plus a few
bytes per vector, and the data is read `iters_meso + 1` times for the meso
step plus once more to load the clusters.

```rust
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix, make_blobs};

let (n, d) = (100_000, 256);
let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);
let kmeans = HierarchicalSuperKMeans::with_config(d, HierarchicalSuperKMeansConfig::default());

let meso = kmeans.train_meso_iter(Matrix::new(&data, n, d).chunks(4_096), n);
for m in 0..meso.n_meso() {
    let rows = meso.meso_rows(m); // ascending scan positions
    let vectors: Vec<f32> = rows
        .iter()
        .flat_map(|&r| data[r as usize * d..(r as usize + 1) * d].iter().copied())
        .collect();
    let fine = kmeans.train_meso_cluster(&vectors, rows.len());
    for c in 0..fine.n_fine() {
        let centroid = &fine.centroids[c * d..(c + 1) * d];
        let members = fine.members(c); // positions into `rows`
        // ... write out this fine cluster
    }
}
```

Fine clusters come out grouped by meso cluster, so writing them in the order
they are produced gives a cluster-ordered layout.

## BLAS backends

The default backend is pure Rust and requires nothing to be installed. Two
optional features route the SGEMM calls through a vendor BLAS via
`cblas_sgemm`:

| Feature       | Platform              | Requirement                                                   |
|---------------|-----------------------|--------------------------------------------------------------|
| _(default)_   | Any                   | None — uses `matrixmultiply`                                  |
| `openblas`    | Linux / Windows / macOS | A system OpenBLAS (`libopenblas-dev`, Homebrew `openblas`, …) |
| `accelerate`  | macOS                 | None — links the system Accelerate framework                 |

```toml
# Linux / Windows: link a system OpenBLAS
superkmeans-rs = { version = "0.3", features = ["openblas"] }

# macOS: use Apple Accelerate (AMX-backed on Apple Silicon)
superkmeans-rs = { version = "0.3", features = ["accelerate"] }
```

Enable **at most one** backend; `openblas` and `accelerate` are mutually
exclusive. When building with `openblas`, `build.rs` locates the library via
`pkg-config`, or via the `OPENBLAS_LIB_DIR` environment variable for custom
installs.

## Examples

Runnable examples live in [`examples/`](examples):

```bash
cargo run --release --example simple_clustering
cargo run --release --example hierarchical_clustering
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the checks CI
enforces.

## License

MIT License - see [LICENSE](LICENSE) for details.
