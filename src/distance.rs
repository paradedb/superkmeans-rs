//! Scalar squared-L2 distance — auto-vectorized by LLVM.

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = 0.0_f32;
    for i in 0..n {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

/// Squared L2 over a slice with `len` elements, given start offsets.
#[inline]
pub fn l2_squared_range(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..len {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}
