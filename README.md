# superkmeans-rs

[![Crates.io](https://img.shields.io/crates/v/superkmeans-rs.svg)](https://crates.io/crates/superkmeans-rs)
[![codecov](https://codecov.io/gh/paradedb/superkmeans-rs/graph/badge.svg)](https://codecov.io/gh/paradedb/superkmeans-rs)
[![CI](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/paradedb/superkmeans-rs/actions/workflows/ci.yml)
[![Documentation](https://docs.rs/superkmeans-rs/badge.svg)](https://docs.rs/superkmeans-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Rust port of SuperKMeans for k-means clustering of high-dimensional vector
embeddings. Training and assignment run in parallel using Rayon, with
[ADSampling](https://github.com/gaoj0017/ADSampling)-style distance pruning.
The default backend is pure Rust and needs no system BLAS.

[API documentation](https://docs.rs/superkmeans-rs) | [Examples](examples)

## Usage

Add the crate to your `Cargo.toml` (requires Rust 1.89 or later):

```toml
[dependencies]
superkmeans-rs = "0.3"
```

The crate is published as `superkmeans-rs` and imported as `superkmeans`.
Vectors are stored in a flat, row-major slice of `n * d` floats:

```rust
use superkmeans::{SuperKMeans, make_blobs};

let (n, d, k) = (10_000, 128, 100);
let data = make_blobs(n, d, k, true, 1.0, 10.0, 42);

let mut kmeans = SuperKMeans::new(k, d);
let centroids = kmeans.train(&data, n);
let assignments = kmeans.assign(&data, &centroids, n);
assert_eq!(assignments.len(), n);
```

Use [`SuperKMeansConfig`](https://docs.rs/superkmeans-rs/latest/superkmeans/struct.SuperKMeansConfig.html)
to set iterations, seed, and pruning parameters. Set `RAYON_NUM_THREADS` to
control parallelism, or run training in your own Rayon thread pool.

For data that does not fit in memory, `train_iter` accepts replayable batches;
`train_dataset` supports fallible data sources. Both scan the data on each pass.
[`HierarchicalSuperKMeans`](https://docs.rs/superkmeans-rs/latest/superkmeans/struct.HierarchicalSuperKMeans.html)
splits clusters until they fit a size cap and also supports streaming the
initial partition. See the API docs and [examples](examples) for these workflows.

## BLAS backends

The default backend uses `matrixmultiply`. To use a system BLAS, enable one of
these mutually exclusive Cargo features:

| Feature | Platform | Requirement |
| --- | --- | --- |
| `openblas` | Linux, Windows, macOS | System OpenBLAS, found via `pkg-config` or `OPENBLAS_LIB_DIR` |
| `accelerate` | macOS | System Accelerate framework |

For example, on macOS:

```toml
superkmeans-rs = { version = "0.3", features = ["accelerate"] }
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.

## License

[MIT](LICENSE).
