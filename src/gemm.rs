//! Thin SGEMM wrapper around `matrixmultiply` so callers don't need to think about strides.

/// Computes `c = a * b` (no transposes), all row-major.
///   `a`: m × k
///   `b`: k × n
///   `c`: m × n
pub fn sgemm_row_major(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// Computes `c = a * b^T`, all row-major in storage.
///   `a`: m × k
///   `b`: n × k  (interpreted as k × n via transpose access)
///   `c`: m × n
pub fn sgemm_row_major_b_transposed(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= n * k);
    debug_assert!(c.len() >= m * n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}
