//! Centroid layout for the pruning path.
//!
//! The original implementation maintains a hybrid column-major (vertical) /
//! row-major-block (horizontal) layout for cache locality during the
//! ADSampling+PDX prune phase. The dimension split is preserved here because
//! the prune algorithm walks dimensions in a specific order (front H_DIM_SIZE
//! blocks first, then the trailing slab), but the storage itself is the
//! plain row-major centroid matrix — sufficient for a correct port. A future
//! optimisation pass can reintroduce the block-transposed layout for cache
//! efficiency.

use crate::common::{H_DIM_SIZE, PROPORTION_HORIZONTAL_DIM};

#[derive(Copy, Clone, Debug)]
pub struct PdxDimensionSplit {
    /// Number of "horizontal" dimensions (processed in H_DIM_SIZE blocks).
    pub horizontal_d: usize,
    /// Number of "vertical" dimensions (processed as a single trailing slab).
    pub vertical_d: usize,
}

/// 25% vertical / 75% horizontal by default, with tweaks for small `d`.
pub fn get_dimension_split(d: usize) -> PdxDimensionSplit {
    let mut local_proportion = PROPORTION_HORIZONTAL_DIM as f64;
    if d <= 256 {
        local_proportion = 0.25;
    }
    let mut horizontal_d = (d as f64 * local_proportion) as usize;
    let mut vertical_d = d - horizontal_d;
    if horizontal_d % H_DIM_SIZE > 0 {
        horizontal_d = ((horizontal_d as f64 / H_DIM_SIZE as f64).round() as usize) * H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    if vertical_d == 0 {
        horizontal_d = H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    if d <= H_DIM_SIZE {
        horizontal_d = 0;
        vertical_d = d;
    }
    PdxDimensionSplit {
        horizontal_d,
        vertical_d,
    }
}
