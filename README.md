# superkmeans-rs

[![Crates.io](https://img.shields.io/crates/v/superkmeans-rs.svg)](https://crates.io/crates/superkmeans-rs)
[![codecov](https://codecov.io/gh/paradedb/superkmeans-rs/graph/badge.svg)](https://codecov.io/gh/paradedb/superkmeans-rs)
[![CI](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml)
[![Documentation](https://docs.rs/superkmeans-rs/badge.svg)](https://docs.rs/superkmeans-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Rust port of SuperKMeans for fast k-means clustering of high-dimensional
vector embeddings.

## Overview

`superkmeans-rs` implements the SuperKMeans C++ library in pure Rust, with no
FFI required. It clusters large collections of high-dimensional embeddings,
such as those produced by text and image models. It uses
[ADSampling](https://github.com/gaoj0017/ADSampling)-style distance pruning and
a cache-friendly matrix layout to keep assignment fast as dimensionality grows.

- The SGEMM inner loop uses
  [`matrixmultiply`](https://crates.io/crates/matrixmultiply) by default, so no
  system BLAS is required to build or run.
- You can use OpenBLAS or Apple Accelerate for SGEMM calls. See
  [BLAS backends](#blas-backends) for setup.
- Training and assignment run in parallel with
  [`rayon`](https://crates.io/crates/rayon).
- `HierarchicalSuperKMeans` builds a balanced cluster tree (HBC) until every
  leaf is below a size cap. By default, the root splits into √K clusters.

## Installation

```toml
[dependencies]
superkmeans-rs = "0.1"
```

The minimum supported Rust version (MSRV) is 1.89, which is required by the
AVX512 intrinsics used in the pruning kernels.

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

`SuperKMeansConfig` provides the settings from the original C++ library,
including the number of iterations, RNG seed, early-termination tolerances,
and ADSampling pruning bounds. Start with the defaults and change what you need:

```rust
use superkmeans::{SuperKMeans, SuperKMeansConfig};

let mut cfg = SuperKMeansConfig::default();
cfg.iters = 20;          // more refinement passes
cfg.seed = 7;            // deterministic runs
cfg.verbose = true;      // log per-iteration statistics

let mut kmeans = SuperKMeans::with_config(1000, 768, cfg);
```

Training uses the current rayon thread pool. Set the thread count with
`RAYON_NUM_THREADS`, or call `train` inside `ThreadPool::install` to use your
own pool.

`train` clusters all the rows you pass to it. To train on a sample, select a
subset before calling `train`, then pass the full set to `assign`. On 200,000
Cohere vectors, training on a 25% sample halved build time with a loss of less
than one point of recall@100.

### Hierarchical clustering

`HierarchicalSuperKMeans` recursively splits the data until every leaf has at
most `max_leaf_size` points. `train` returns the leaf centroids, and the full
tree is available on `kmeans.tree`. The resulting cluster count is close to
`n / max_leaf_size`.

By default, the root splits into `ceil(sqrt(K))` children with
`K = ceil(n / max_leaf_size)`, and each deeper split uses
`k = ceil(n_i / max_leaf_size)` so an oversized subtree finishes in one pass.
Set `branching_factor = Some(b)` for a fixed fan-out, such as 2 for a balanced
k-means tree.

Each split rebalances its undersized clusters, producing evenly sized leaves
without an explicit balance penalty.

Splits reorder the training set in place rather than copying each child out.
With `train_owned`, peak memory is one copy of the training set. `train`
duplicates the caller's slice before rotating it.

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
assert_eq!(centroids.len(), kmeans.tree.n_leaves * d);
```

## BLAS backends

The default backend is pure Rust and needs no system libraries. Two optional
features route SGEMM calls through a vendor BLAS using
`cblas_sgemm`:

| Feature       | Platform              | Requirement                                                   |
|---------------|-----------------------|--------------------------------------------------------------|
| (default)     | Any                   | Uses `matrixmultiply`, with no system dependencies            |
| `openblas`    | Linux / Windows / macOS | A system OpenBLAS installation, such as `libopenblas-dev` or Homebrew `openblas` |
| `accelerate`  | macOS                 | Uses the system Accelerate framework                         |

```toml
# Linux / Windows: link a system OpenBLAS
superkmeans-rs = { version = "0.1", features = ["openblas"] }

# macOS: use Apple Accelerate (AMX-backed on Apple Silicon)
superkmeans-rs = { version = "0.1", features = ["accelerate"] }
```

Enable at most one backend. `openblas` and `accelerate` are mutually exclusive.
When building with `openblas`, `build.rs` finds the library through `pkg-config`.
For a custom installation, set the `OPENBLAS_LIB_DIR` environment variable.

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
