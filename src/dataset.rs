//! Restartable sources of [`Matrix`] batches for streaming k-means.

use crate::matrix::Matrix;

/// A source of row-major [`Matrix`] batches that can be replayed.
///
/// K-means walks the training set once to initialize centroids and once per
/// Lloyd iteration, so [`Dataset::for_each_batch`] must yield the **same
/// vectors in the same order** on every call. Implement this for sources that
/// are not a [`Clone`] iterator (for example a file that is re-read each pass).
///
/// Each yielded matrix is `n × d` (`n` may be 1). The training loop packs
/// small matrices into GEMM-sized work batches, so yielding one vector at a
/// time is correct but slower than handing over larger `n`.
pub trait Dataset {
    /// Invoke `f` once per batch. Called once per k-means pass.
    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>));
}

/// A [`Clone`] iterator of [`Matrix`] views, replayed by cloning each pass.
///
/// Built by [`crate::SuperKMeans::train_iter`]. The iterator must be
/// restartable: `clone()` has to produce the same sequence again.
pub struct IterDataset<I> {
    iter: I,
}

impl<I> IterDataset<I> {
    /// Wrap a cloneable iterator of matrices.
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<'a, I> Dataset for IterDataset<I>
where
    I: Iterator<Item = Matrix<'a>> + Clone,
{
    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) {
        for matrix in self.iter.clone() {
            if !matrix.is_empty() {
                f(matrix);
            }
        }
    }
}

impl Dataset for Matrix<'_> {
    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) {
        if !self.is_empty() {
            f(*self);
        }
    }
}

/// Walk `data`, coalescing small matrices and splitting large ones so each
/// call to `f` sees about `batch_rows` vectors (the GEMM / rayon width).
///
/// Large batches that already sit in a contiguous slice are split without
/// copying. Single-vector items are packed into `buf`.
pub(crate) fn visit_work_batches<D, F>(
    data: &mut D,
    d: usize,
    batch_rows: usize,
    buf: &mut Vec<f32>,
    mut f: F,
) where
    D: Dataset + ?Sized,
    F: FnMut(Matrix<'_>),
{
    assert!(d > 0, "dimensionality must be positive");
    assert!(batch_rows > 0, "batch_rows must be positive");
    buf.clear();

    data.for_each_batch(&mut |matrix| {
        assert_eq!(
            matrix.d(),
            d,
            "matrix dimensionality {} does not match the model ({d})",
            matrix.d()
        );
        if matrix.is_empty() {
            return;
        }

        if buf.is_empty() && matrix.n() >= batch_rows {
            for chunk in matrix.chunks(batch_rows) {
                f(chunk);
            }
            return;
        }

        for row in matrix.rows() {
            if buf.is_empty() {
                buf.reserve(batch_rows * d);
            }
            buf.extend_from_slice(row);
            if buf.len() == batch_rows * d {
                f(Matrix::new(buf, batch_rows, d));
                buf.clear();
            }
        }
    });

    if !buf.is_empty() {
        let n = buf.len() / d;
        f(Matrix::new(buf, n, d));
        buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_unit_matrices_and_splits_large_ones() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let d = 2;
        let matrices: Vec<Matrix<'_>> = data.chunks(d).map(Matrix::vector).collect();
        let mut dataset = IterDataset::new(matrices.into_iter());

        let mut buf = Vec::new();
        let mut seen = Vec::new();
        visit_work_batches(&mut dataset, d, 3, &mut buf, |m| {
            seen.push((m.n(), m.as_slice().to_vec()));
        });
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0].0, 3);
        assert_eq!(seen[3].0, 1);
        assert_eq!(seen[3].1, vec![18.0, 19.0]);

        // A single large matrix is split without changing values.
        let mut whole = Matrix::from_slice(&data, d);
        let mut sizes = Vec::new();
        visit_work_batches(&mut whole, d, 4, &mut buf, |m| {
            sizes.push(m.n());
        });
        assert_eq!(sizes, vec![4, 4, 2]);
    }

    #[test]
    fn iter_dataset_replays_in_the_same_order() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let a = Matrix::new(&data[..2], 1, 2);
        let b = Matrix::new(&data[2..], 1, 2);
        let mut dataset = IterDataset::new([a, b].into_iter());

        let mut first = Vec::new();
        dataset.for_each_batch(&mut |m| first.push(m.as_slice().to_vec()));
        let mut second = Vec::new();
        dataset.for_each_batch(&mut |m| second.push(m.as_slice().to_vec()));
        assert_eq!(first, second);
    }
}
