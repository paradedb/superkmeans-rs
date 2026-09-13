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
- **Hierarchical clustering** — `HierarchicalSuperKMeans` iteratively builds a
  balanced cluster tree (HBC) until every leaf is below a size cap, using a
  √K meso root split by default.

> **Note:** the crate is published as `superkmeans-rs` but the library target
> is named `superkmeans`, so you import it as `use superkmeans::...`.

## Installation

```toml
[dependencies]
superkmeans-rs = "0.1"
```

The minimum supported Rust version (MSRV) is **1.89** (required by the AVX512 intrinsics used in the pruning kernels).

## Quick Start

Training data is always a contiguous row-major `n × d` buffer of `f32`s
(`n` vectors of dimensionality `d`). The in-memory path still takes that as a
flat `&[f32]`; streaming training yields the same layout as [`Matrix`]
items (`n = 1` is one vector, `n > 1` is a batch):

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

K-means only needs one pass over the points per Lloyd iteration (assign,
recompute centroids, measure how far they moved). When the full set cannot
stay in RAM, hand SuperKMeans an iterator of [`Matrix`] views instead of one
giant slice. Each item is still a contiguous `&[f32]` of length `n * d`:

```rust
use superkmeans::{Matrix, SuperKMeans, SuperKMeansConfig, make_blobs};

let n = 10_000;
let d = 128;
let k = 100;
let data = make_blobs(n, d, k, true, 1.0, 10.0, 42);
let view = Matrix::new(&data, n, d);

let mut kmeans = SuperKMeans::with_config(k, d, SuperKMeansConfig::default());

// Replayable iterator of batches. `clone()` must yield the same rows again —
// k-means walks the stream once to sample initial centroids and once per
// iteration. Prefer fat batches so GEMM / rayon stay saturated; single-vector
// items (`Matrix::vector`) are packed automatically.
let centroids = kmeans.train_iter(view.chunks(1_024));
let assignments = kmeans.assign(&data, &centroids, n);
assert_eq!(assignments.len(), n);
```

`train_iter` requires a [`Clone`] iterator. A source that cannot be cloned
(for example a file re-read each pass) implements [`Dataset`] and calls
`train_dataset`. These entry points retain assignments and distances for all rows.
Use `train_spillable` below to bound that state as well.

### Hierarchical clustering

`HierarchicalSuperKMeans` recursively splits the data until every leaf has at
most `max_leaf_size` points. `train` returns the leaf centroids; the full tree
is on `kmeans.tree`. The cluster count is emergent and lands near
`n / max_leaf_size`.

By default the root splits into `ceil(sqrt(K))` children with
`K = ceil(n / max_leaf_size)`, and each deeper split uses
`k = ceil(n_i / max_leaf_size)` so an oversized subtree finishes in one pass.
Set `branching_factor = Some(b)` for a fixed fan-out (e.g. 2 for a balanced
k-means tree).

Leaves come out evenly sized without an explicit balance penalty: each split
rebalances its own undersized clusters.

Splits reorder the training set in place rather than copying each child out.
With `train_owned` peak memory is one copy of the training set; `train` must
duplicate the caller's slice before rotating it. That permute is why
`HierarchicalSuperKMeans` has no `train_iter` / `train_dataset` today: child
splits are slices of the permuted buffer (`data_offset` / `size`), not
filtered replays of the original stream.

The separate `train_spillable` entry point uses temporary partitions instead of
permuting a resident dataset. The existing in-memory entry points remain available.

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

### Spillable training

Both `SuperKMeans` and `HierarchicalSuperKMeans` accept a batch source, a
temporary-storage provider, and a workspace budget:

```rust,no_run
use superkmeans::{FileTempStorage, HierarchicalSuperKMeans, Matrix, SpillOptions};

let mut storage = FileTempStorage;
let mut model = HierarchicalSuperKMeans::new(d);
let centroids = model.train_spillable(
    &mut Matrix::new(&data, n, d),
    &mut storage,
    SpillOptions { memory_budget: 64 * 1024 * 1024 },
)?;
```

The source is consumed once into temporary storage. For a source that is not
already in RAM, implement `TryDataset` to yield borrowed batches and propagate
I/O errors. Existing `Dataset` implementations also work. `TempMatrix` provides
a file-backed source with append and row replacement for reservoir sampling.

Vectors and assignment history live in temporary files; distances, norms,
centroid sums and GEMM scratch use bounded work buffers. Hierarchical training
processes one split at a time and uses at most three internal files, with
partitions represented by ranges in shared files. Files are reclaimed on drop,
including on errors. `TempStorage` only creates seekable files; a custom file's
`Drop` implements cleanup. It has no `Send` or `Sync` requirement, so a PostgreSQL
caller can supply `BufFile` wrappers while keeping I/O on its backend thread.

The workspace budget includes active local centroid buffers and the provider's
declared per-file buffering. Caller-owned input, the shared `O(d²)` rotation,
retained model/tree, and iteration statistics are additional memory. A fixed
`max_leaf_size` still produces more model centroids as the training set grows.
A budget that cannot hold one batch and the local centroids returns an error.
Row counts above `u32::MAX` are rejected because cluster counts use `u32`.

Spillable training does not retain per-row assignments on the model. Use
`assign` on bounded input batches when assignments are needed afterwards.
Initialization uses reservoir sampling, so results need not match the
in-memory path's shuffled initialization. Within a fixed configuration and
workspace budget, changing source batch boundaries preserves the work batches.
Tree `data_offset` values describe logical split order, not physical file offsets.

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
