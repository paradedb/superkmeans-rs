# Contributing to superkmeans-rs

Thanks for your interest in contributing to superkmeans-rs!

## Prerequisites

- **Rust 1.85+** (the minimum supported Rust version)
- Optional, for the BLAS backends:
  - `openblas` — a system OpenBLAS (`libopenblas-dev` on Debian/Ubuntu, `brew install openblas` on macOS)
  - `accelerate` — macOS only; uses the system Accelerate framework (no extra install)

## Development

```bash
cargo build                       # Default build (pure-Rust matrixmultiply backend)
cargo build --features openblas   # Route SGEMM through a system OpenBLAS
cargo build --features accelerate # Route SGEMM through Apple Accelerate (macOS)
cargo test                        # Run tests
cargo run --example simple_clustering
```

The `blas` feature is an internal marker — enable a concrete backend
(`openblas` or `accelerate`) rather than `blas` directly. Do not enable
`accelerate` and `openblas` at the same time.

Before submitting a PR, make sure CI checks pass locally:

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
```

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Add tests for any new functionality or bug fixes.
3. Run the checks above.
4. Open a pull request against `main`.

## Reporting Issues

Open an issue on [GitHub](https://github.com/paradedb/superkmeans-rs/issues) with a clear description of the problem or feature request.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
