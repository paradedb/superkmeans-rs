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
- **Hierarchical clustering** — `HierarchicalSuperKMeans` builds a two-level
  clustering for very large numbers of clusters.

> **Note:** the crate is published as `superkmeans-rs` but the library target
> is named `superkmeans`, so you import it as `use superkmeans::...`.

## Installation

```toml
[dependencies]
superkmeans-rs = "0.1"
```

The minimum supported Rust version (MSRV) is **1.85**.

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
iterations, sampling fraction, thread count, RNG seed, early-termination
tolerances, ADSampling pruning bounds, and more. Start from the defaults and
override what you need:

```rust
use superkmeans::{SuperKMeans, SuperKMeansConfig};

let mut cfg = SuperKMeansConfig::default();
cfg.iters = 20;          // more refinement passes
cfg.n_threads = 8;       // 0 = use all available cores
cfg.seed = 7;            // deterministic runs
cfg.verbose = true;      // log per-iteration statistics

let mut kmeans = SuperKMeans::with_config(1000, 768, cfg);
```

### Hierarchical clustering

For very large `k`, `HierarchicalSuperKMeans` clusters in two levels:

```rust
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, make_blobs};

let n = 100_000;
let d = 256;
let data = make_blobs(n, d, 100, true, 1.0, 10.0, 42);

let mut kmeans =
    HierarchicalSuperKMeans::with_config(10_000, d, HierarchicalSuperKMeansConfig::default());
let centroids = kmeans.train(&data, n);
let assignments = kmeans.assign(&data, &centroids, n);
```

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
superkmeans-rs = { version = "0.1", features = ["openblas"] }

# macOS: use Apple Accelerate (AMX-backed on Apple Silicon)
superkmeans-rs = { version = "0.1", features = ["accelerate"] }
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
