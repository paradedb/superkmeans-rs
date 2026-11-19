//! Thin row-major matrix view used as the streaming training item.
//!
//! SuperKMeans never needs a general tensor type: every batch is a contiguous
//! `n × d` block of `f32`s (`n` vectors of dimensionality `d`). [`Matrix`] is
//! that view. `n = 1` is a single vector; `n > 1` is a batch.

/// A borrowed row-major `n × d` matrix over a contiguous `[f32]` buffer.
///
/// The buffer length is always `n * d`. This is the item type yielded by a
/// streaming training iterator: each item may be one vector (`n = 1`) or a
/// batch (`n > 1`). Prefer larger `n` so the GEMM / rayon kernels stay busy.
///
/// # Examples
///
/// ```
/// use superkmeans::Matrix;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let batch = Matrix::new(&data, 3, 2);
/// assert_eq!(batch.n(), 3);
/// assert_eq!(batch.d(), 2);
/// assert_eq!(batch.row(1), &[3.0, 4.0]);
///
/// let one = Matrix::vector(&[1.0, 2.0]);
/// assert_eq!(one.n(), 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    data: &'a [f32],
    n: usize,
    d: usize,
}

impl<'a> Matrix<'a> {
    /// View `data` as an `n × d` row-major matrix.
    ///
    /// # Panics
    ///
    /// Panics if `d == 0` or `data.len() != n * d`.
    pub fn new(data: &'a [f32], n: usize, d: usize) -> Self {
        assert!(d > 0, "dimensionality must be positive");
        assert_eq!(
            data.len(),
            n.checked_mul(d).expect("n * d overflows usize"),
            "data length {} is not n ({n}) * d ({d})",
            data.len()
        );
        Self { data, n, d }
    }

    /// A single vector (`n = 1`), with `d = data.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `data` is empty.
    pub fn vector(data: &'a [f32]) -> Self {
        assert!(
            !data.is_empty(),
            "a vector must have positive dimensionality"
        );
        Self {
            data,
            n: 1,
            d: data.len(),
        }
    }

    /// Number of vectors (rows).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Vector dimensionality (columns).
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Contiguous row-major storage, length `n * d`.
    #[inline]
    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    /// Row `i` as a length-`d` vector.
    ///
    /// # Panics
    ///
    /// Panics if `i >= n`.
    #[inline]
    pub fn row(&self, i: usize) -> &'a [f32] {
        let start = i * self.d;
        &self.data[start..start + self.d]
    }

    /// Iterator over the `n` rows.
    pub fn rows(&self) -> impl Iterator<Item = &'a [f32]> + Clone {
        self.data.chunks_exact(self.d)
    }

    /// Split into row-major chunks of at most `rows` vectors.
    ///
    /// The last chunk may be shorter. `rows` must be positive.
    ///
    /// The resulting iterator is [`Clone`], so it can be replayed across
    /// k-means passes inside an [`crate::IterDataset`].
    pub fn chunks(&self, rows: usize) -> MatrixChunks<'a> {
        assert!(rows > 0, "chunk row count must be positive");
        MatrixChunks {
            inner: self.data.chunks(rows * self.d),
            d: self.d,
        }
    }

    /// True when the view contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl AsRef<[f32]> for Matrix<'_> {
    fn as_ref(&self) -> &[f32] {
        self.data
    }
}

impl PartialEq for Matrix<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.d == other.d && self.data == other.data
    }
}

impl Eq for Matrix<'_> {}

/// Iterator of [`Matrix`] row-chunks produced by [`Matrix::chunks`].
///
/// Cloneable so an [`crate::IterDataset`] of chunks can be replayed each Lloyd
/// iteration.
#[derive(Clone, Debug)]
pub struct MatrixChunks<'a> {
    inner: std::slice::Chunks<'a, f32>,
    d: usize,
}

impl<'a> Iterator for MatrixChunks<'a> {
    type Item = Matrix<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.inner.next()?;
        // `chunks` is sized as a multiple of `d` except when the parent view
        // itself was malformed; `Matrix::new` already enforced `n * d`.
        let n = chunk.len() / self.d;
        Some(Matrix {
            data: chunk,
            n,
            d: self.d,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for MatrixChunks<'_> {}

/// An owned row-major `n × d` block of `f32`.
///
/// `d` is stored on the value. A bare [`Vec<f32>`] does not know it.
/// [`Vectors::as_matrix`] borrows the block as a [`Matrix`].
#[derive(Clone, Debug, PartialEq)]
pub struct Vectors {
    pub data: Vec<f32>,
    pub d: usize,
}

impl Vectors {
    /// # Panics
    ///
    /// Panics if `d` is zero or `data.len()` is not a multiple of `d`.
    pub fn new(data: Vec<f32>, d: usize) -> Self {
        assert!(d > 0, "dimensionality must be positive");
        assert_eq!(
            data.len() % d,
            0,
            "data length {} is not a multiple of d ({d})",
            data.len()
        );
        Self { data, d }
    }

    pub fn n(&self) -> usize {
        self.data.len() / self.d
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_matrix(&self) -> Matrix<'_> {
        Matrix::new(&self.data, self.n(), self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_follow_n_and_d() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let a = Matrix::new(&data, 2, 2);
        assert_eq!(a.row(0), &[1.0, 2.0]);
        assert_eq!(a.row(1), &[3.0, 4.0]);
    }

    #[test]
    fn vector_is_n_eq_1() {
        let v = Matrix::vector(&[9.0, 8.0, 7.0]);
        assert_eq!(v.n(), 1);
        assert_eq!(v.d(), 3);
        assert_eq!(v.as_slice(), &[9.0, 8.0, 7.0]);
    }

    #[test]
    fn chunks_split_rows_and_are_cloneable() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let m = Matrix::new(&data, 4, 3);
        let chunks: Vec<_> = m.chunks(2).collect();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].n(), 2);
        assert_eq!(chunks[1].row(0), &[6.0, 7.0, 8.0]);

        let replay: Vec<_> = m.chunks(2).collect();
        assert_eq!(chunks, replay);
    }

    #[test]
    fn empty_matrix_is_allowed() {
        let data: &[f32] = &[];
        let m = Matrix::new(data, 0, 4);
        assert!(m.is_empty());
        assert_eq!(m.chunks(8).count(), 0);
    }

    #[test]
    #[should_panic(expected = "dimensionality must be positive")]
    fn rejects_zero_d() {
        let _ = Matrix::new(&[1.0], 1, 0);
    }

    #[test]
    #[should_panic(expected = "is not n")]
    fn rejects_length_mismatch() {
        let _ = Matrix::new(&[1.0, 2.0, 3.0], 2, 2);
    }
}
