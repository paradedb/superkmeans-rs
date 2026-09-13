//! Temporary storage and bounded batches for opt-in spillable training.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::common::{X_BATCH_SIZE, Y_BATCH_SIZE};
use crate::{Dataset, Matrix};

/// A fallible batch source. Spillable training consumes it once into scratch storage.
pub trait TryDataset {
    fn try_for_each_batch(
        &mut self,
        f: &mut dyn FnMut(Matrix<'_>) -> io::Result<()>,
    ) -> io::Result<()>;
}

impl<D: Dataset + ?Sized> TryDataset for D {
    fn try_for_each_batch(
        &mut self,
        f: &mut dyn FnMut(Matrix<'_>) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut result = Ok(());
        self.for_each_batch(&mut |batch| {
            if result.is_ok() {
                result = f(batch);
            }
        });
        result
    }
}

/// Creates seekable scratch files. Dropping a file must reclaim its storage.
/// Files need not implement `Send` or `Sync`; all I/O runs on the calling thread.
pub trait TempStorage {
    type File: Read + Write + Seek;

    fn create(&mut self) -> io::Result<Self::File>;

    /// Resident buffering owned by each file, excluding the trainer's buffers.
    fn buffer_bytes_per_file(&self) -> usize {
        0
    }
}

/// Uses the operating system's temporary directory and deletes files on drop.
#[derive(Default)]
pub struct FileTempStorage;

impl TempStorage for FileTempStorage {
    type File = File;

    fn create(&mut self) -> io::Result<File> {
        tempfile::tempfile()
    }
}

/// Bounds active training buffers, including local centroid accumulators.
/// Caller-owned input, the shared rotation, the retained tree/model and its
/// iteration statistics are additional memory proportional to model size.
#[derive(Clone, Copy, Debug)]
pub struct SpillOptions {
    pub memory_budget: usize,
}

impl SpillOptions {
    pub(crate) fn batch_rows<S: TempStorage>(
        self,
        d: usize,
        k: usize,
        storage: &S,
    ) -> io::Result<usize> {
        if d == 0 {
            return Err(invalid("dimensionality must be positive"));
        }
        let tile = k.min(Y_BATCH_SIZE);
        let fixed = d
            .checked_mul(k)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_add(k.checked_mul(64)?))
            .and_then(|v| {
                v.checked_add(
                    rayon::current_num_threads()
                        .checked_mul(tile)?
                        .checked_mul(4)?,
                )
            })
            .and_then(|v| v.checked_add(storage.buffer_bytes_per_file().checked_mul(4)?))
            .ok_or_else(|| invalid("training workspace size overflow"))?;
        let per_row = d
            .checked_mul(2)
            .and_then(|v| v.checked_add(tile))
            .and_then(|v| v.checked_add(16))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| invalid("training row size overflow"))?;
        let rows = self.memory_budget.saturating_sub(fixed) / per_row;
        if rows == 0 {
            return Err(invalid(
                "training memory budget cannot fit one work batch and local centroids",
            ));
        }
        Ok(rows.min(X_BATCH_SIZE))
    }
}

/// A row-major temporary matrix, also usable as a disk-backed sampling reservoir.
pub struct TempMatrix<F> {
    pub(crate) file: F,
    pub(crate) d: usize,
    pub(crate) n: usize,
    batch_rows: usize,
}

impl<F: Read + Write + Seek> TempMatrix<F> {
    pub fn new(file: F, d: usize, batch_rows: usize) -> io::Result<Self> {
        if d == 0 || batch_rows == 0 || batch_rows.checked_mul(d).is_none() {
            return Err(invalid(
                "matrix dimensions and batch size must be positive and fit usize",
            ));
        }
        Ok(Self {
            file,
            d,
            n: 0,
            batch_rows,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn append(&mut self, batch: Matrix<'_>) -> io::Result<()> {
        if batch.d() != self.d {
            return Err(invalid("matrix dimensionality changed"));
        }
        let n = self
            .n
            .checked_add(batch.n())
            .ok_or_else(|| invalid("row count overflow"))?;
        self.file
            .seek(SeekFrom::Start(row_offset(self.n, self.d)?))?;
        self.file
            .write_all(bytemuck::cast_slice(batch.as_slice()))?;
        self.n = n;
        Ok(())
    }

    pub fn replace_row(&mut self, row: usize, values: &[f32]) -> io::Result<()> {
        if row >= self.n || values.len() != self.d {
            return Err(invalid(
                "replacement row is outside the matrix or has the wrong dimension",
            ));
        }
        self.file.seek(SeekFrom::Start(row_offset(row, self.d)?))?;
        self.file.write_all(bytemuck::cast_slice(values))
    }
}

impl<F: Read + Write + Seek> TryDataset for TempMatrix<F> {
    fn try_for_each_batch(
        &mut self,
        f: &mut dyn FnMut(Matrix<'_>) -> io::Result<()>,
    ) -> io::Result<()> {
        visit_partition(&mut self.file, 0, self.n, self.d, self.batch_rows, f)
    }
}

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(crate) fn row_offset(row: usize, d: usize) -> io::Result<u64> {
    (row as u64)
        .checked_mul(d as u64)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("temporary matrix offset overflow"))
}

pub(crate) fn visit_partition<F: Read + Seek>(
    file: &mut F,
    start: usize,
    n: usize,
    d: usize,
    batch_rows: usize,
    mut f: impl FnMut(Matrix<'_>) -> io::Result<()>,
) -> io::Result<()> {
    file.seek(SeekFrom::Start(row_offset(start, d)?))?;
    let mut rows = vec![0.0_f32; n.min(batch_rows) * d];
    let mut remaining = n;
    while remaining > 0 {
        let count = remaining.min(batch_rows);
        let values = &mut rows[..count * d];
        file.read_exact(bytemuck::cast_slice_mut(values))?;
        f(Matrix::new(values, count, d))?;
        remaining -= count;
    }
    Ok(())
}

pub(crate) fn spool<D: TryDataset + ?Sized, S: TempStorage>(
    data: &mut D,
    d: usize,
    storage: &mut S,
    options: SpillOptions,
    pruner: &crate::adsampling::ADSamplingPruner,
    rotate: bool,
) -> io::Result<TempMatrix<S::File>> {
    let batch_rows = options.batch_rows(d, 0, storage)?;
    let mut matrix = TempMatrix::new(storage.create()?, d, batch_rows)?;
    let mut rotated = Vec::new();
    let mut pending = Vec::with_capacity(batch_rows * d);
    let mut write_batch = |values: &[f32]| -> io::Result<()> {
        let n = values.len() / d;
        if rotate {
            rotated.resize(values.len(), 0.0);
            pruner.rotate(values, &mut rotated, n);
            matrix.append(Matrix::new(&rotated, n, d))
        } else {
            matrix.append(Matrix::new(values, n, d))
        }
    };
    data.try_for_each_batch(&mut |batch| {
        if batch.d() != d {
            return Err(invalid("dataset dimensionality does not match the model"));
        }
        for row in batch.rows() {
            pending.extend_from_slice(row);
            if pending.len() == batch_rows * d {
                write_batch(&pending)?;
                pending.clear();
            }
        }
        Ok(())
    })?;
    if !pending.is_empty() {
        write_batch(&pending)?;
    }
    if matrix.n == 0 || matrix.n > u32::MAX as usize {
        return Err(invalid("training row count must be between 1 and u32::MAX"));
    }
    Ok(matrix)
}
