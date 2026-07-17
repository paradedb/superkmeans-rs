//! Squared-L2 distance with a SIMD-friendly reduction.
//!
//! A single-accumulator `acc += d*d` loop does NOT auto-vectorize: float
//! addition isn't reassociated without fast-math, so LLVM must keep the
//! sequential dependency and emits scalar code. Using independent per-lane
//! accumulators breaks that chain, letting LLVM emit a vector FMA loop (NEON /
//! AVX / AVX512 via `target-cpu=native`) — the portable equivalent of the C++'s
//! explicit SIMD distance kernels.

const LANES: usize = 8;

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (av, bv) in ca.by_ref().zip(cb.by_ref()) {
        // Independent accumulators — no cross-lane dependency, so this vectorizes.
        for l in 0..LANES {
            let d = av[l] - bv[l];
            acc[l] += d * d;
        }
    }
    let mut s = 0.0_f32;
    for l in 0..LANES {
        s += acc[l];
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Squared L2 over the first `len` elements of `a` and `b`.
#[inline]
pub fn l2_squared_range(a: &[f32], b: &[f32], len: usize) -> f32 {
    l2_squared(&a[..len], &b[..len])
}
