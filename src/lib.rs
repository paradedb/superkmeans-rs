//! Rust port of SuperKMeans — a fast k-means clustering library for
//! high-dimensional vector embeddings using BLAS+ADSampling pruning.
//!
//! The C++ original lives in `SuperKMeans/`; this crate re-implements the
//! C++ public API (no Python surface) in pure Rust with no FFI dependencies.

pub mod adsampling;
pub mod batch;
pub mod common;
pub mod distance;
pub mod gemm;
pub mod hierarchical;
pub mod layout;
pub mod pdxearch;
pub mod superkmeans;
pub mod utils;

pub use common::{DistanceFunction, KnnCandidate};
pub use hierarchical::{
    HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, HierarchicalSuperKMeansIterationStats,
};
pub use superkmeans::{
    ClusterBalanceStats, SuperKMeans, SuperKMeansConfig, SuperKMeansIterationStats,
};
pub use utils::{
    TicToc, compute_l2_squared, compute_norms_row_major, find_nearest_neighbor_brute_force,
    generate_random_vectors, make_blobs,
};
