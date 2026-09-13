use std::io::{self, Read, Seek, SeekFrom, Write};

use super::*;
use crate::spill::{
    SpillOptions, TempStorage, TryDataset, invalid, row_offset, spool, visit_partition,
};

impl HierarchicalSuperKMeans {
    /// Train through temporary partitions, keeping the existing in-memory path separate.
    /// `data_offset` remains a logical offset in stable split order; it is not
    /// an offset into the source or a retained temporary file.
    pub fn train_spillable<D: TryDataset + ?Sized, S: TempStorage>(
        &mut self,
        data: &mut D,
        storage: &mut S,
        options: SpillOptions,
    ) -> io::Result<Vec<f32>> {
        if self.base.trained {
            return Err(invalid("the clustering has already been trained"));
        }
        let d = self.base.d;
        let mut source = spool(
            data,
            d,
            storage,
            options,
            &self.pruner,
            !self.config.base.data_already_rotated,
        )?;
        let n = source.n;
        self.tree.clear();
        self.iteration_stats = HierarchicalSuperKMeansIterationStats::default();
        let root = self.push_leaf(0, n, &vec![0.0; d]);
        self.tree.root = root;

        if self.children_for_split(n, true) < 2 {
            let batch_rows = options.batch_rows(d, 1, storage)?;
            let centroid = &mut self.tree.centroids[..d];
            visit_partition(&mut source.file, 0, n, d, batch_rows, |batch| {
                for row in batch.rows() {
                    for (sum, value) in centroid.iter_mut().zip(row) {
                        *sum += value;
                    }
                }
                Ok(())
            })?;
            for value in centroid {
                *value *= 1.0 / n as f32;
            }
        } else {
            let mut pending = vec![(root, 0)];
            while !pending.is_empty() {
                let mut next_file = storage.create()?;
                let mut next_rows = 0;
                let mut next = Vec::new();
                for (id, start) in pending {
                    let size = self.tree.node(id).size();
                    let logical_start = self.tree.node(id).data_offset();
                    let centroid_offset = self.tree.node(id).centroid_offset();
                    let k = self.children_for_split(size, id == root);
                    let batch_rows = options.batch_rows(d, k, storage)?;
                    let mut local = self.new_superkmeans_from_config(
                        k,
                        LocalSkmMode::Split {
                            iters: self.config.iters_per_split,
                        },
                    );
                    let (mut labels, mut counts) = local.train_spillable_partition(
                        &mut source.file,
                        start,
                        size,
                        storage,
                        options,
                    )?;
                    self.iteration_stats
                        .local_runs
                        .append(&mut local.iteration_stats);
                    let mut centroids = std::mem::take(&mut local.horizontal_centroids);
                    drop(local);
                    let repaired = counts.iter().filter(|&&count| count > 0).count() < 2;
                    if repaired {
                        counts = vec![size / 2, size - size / 2];
                        centroids = vec![0.0; 2 * d];
                    }
                    let mut destinations = Vec::with_capacity(counts.len());
                    for &count in &counts {
                        destinations.push(if count > self.config.max_leaf_size {
                            let offset = next_rows;
                            next_rows += count;
                            Some(offset)
                        } else {
                            None
                        });
                    }
                    scatter_partition(
                        &mut source.file,
                        &mut next_file,
                        &mut labels,
                        start,
                        size,
                        d,
                        batch_rows,
                        &counts,
                        &destinations,
                        &mut centroids,
                        repaired,
                    )?;

                    let child_start = self.tree.nodes.len();
                    let mut logical_offset = logical_start;
                    for (cluster, &count) in counts.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let child = self.push_leaf(
                            logical_offset,
                            count,
                            &centroids[cluster * d..(cluster + 1) * d],
                        );
                        if let Some(offset) = destinations[cluster] {
                            next.push((child, offset));
                        }
                        logical_offset += count;
                    }
                    self.tree.nodes[id.0] = TreeNode::Internal {
                        centroid_offset,
                        data_offset: logical_start,
                        size,
                        children_offset: child_start,
                        children_size: self.tree.nodes.len() - child_start,
                    };
                }
                source.file = next_file;
                pending = next;
            }
        }
        self.count_leaves();
        Ok(self.finish_tree(n))
    }
}

fn scatter_partition<F: Read + Write + Seek>(
    source: &mut F,
    destination: &mut F,
    labels: &mut F,
    start: usize,
    n: usize,
    d: usize,
    batch_rows: usize,
    counts: &[usize],
    destinations: &[Option<usize>],
    centroids: &mut [f32],
    repaired: bool,
) -> io::Result<()> {
    if !repaired && destinations.iter().all(Option::is_none) {
        return Ok(());
    }
    let mut cursors = destinations.to_vec();
    let mut assignments = vec![0_u32; n.min(batch_rows)];
    let mut order = Vec::with_capacity(assignments.len());
    let mut grouped = Vec::with_capacity(assignments.len() * d);
    let mut offset = 0;
    labels.rewind()?;
    visit_partition(source, start, n, d, batch_rows, |batch| {
        let assign = &mut assignments[..batch.n()];
        if repaired {
            for (i, label) in assign.iter_mut().enumerate() {
                *label = u32::from(offset + i >= counts[0]);
            }
            for (row, &cluster) in batch.rows().zip(assign.iter()) {
                for (sum, value) in centroids[cluster as usize * d..(cluster as usize + 1) * d]
                    .iter_mut()
                    .zip(row)
                {
                    *sum += value;
                }
            }
        } else {
            labels.read_exact(bytemuck::cast_slice_mut(assign))?;
        }
        order.clear();
        order.extend(0..batch.n());
        order.sort_by_key(|&i| assign[i]);
        let mut first = 0;
        while first < order.len() {
            let cluster = assign[order[first]] as usize;
            let mut end = first + 1;
            while end < order.len() && assign[order[end]] as usize == cluster {
                end += 1;
            }
            if let Some(cursor) = &mut cursors[cluster] {
                grouped.clear();
                for &i in &order[first..end] {
                    grouped.extend_from_slice(&batch.as_slice()[i * d..(i + 1) * d]);
                }
                destination.seek(SeekFrom::Start(row_offset(*cursor, d)?))?;
                destination.write_all(bytemuck::cast_slice(&grouped))?;
                *cursor += end - first;
            }
            first = end;
        }
        offset += batch.n();
        Ok(())
    })?;
    if repaired {
        for (row, &count) in centroids.chunks_exact_mut(d).zip(counts) {
            for value in row {
                *value *= 1.0 / count as f32;
            }
        }
    }
    Ok(())
}
