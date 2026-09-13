use std::alloc::{GlobalAlloc, Layout, System};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

use superkmeans::{
    FileTempStorage, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix, SpillOptions,
    SuperKMeans, SuperKMeansConfig, TryDataset,
};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct GeneratedRows {
    n: usize,
}

impl TryDataset for GeneratedRows {
    fn try_for_each_batch(
        &mut self,
        f: &mut dyn FnMut(Matrix<'_>) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut data = vec![0.0; 512 * 16];
        for start in (0..self.n).step_by(512) {
            let rows = (self.n - start).min(512);
            for (i, row) in data[..rows * 16].chunks_exact_mut(16).enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = ((start + i) % 32) as f32 + j as f32 * 0.01;
                }
            }
            f(Matrix::new(&data[..rows * 16], rows, 16))?;
        }
        Ok(())
    }
}

fn measure(n: usize, hierarchical: bool) -> usize {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let options = SpillOptions {
        memory_budget: 128 * 1024,
    };
    let base = SuperKMeansConfig {
        iters: 2,
        early_termination: false,
        data_already_rotated: true,
        ..Default::default()
    };
    let mut data = GeneratedRows { n };
    if hierarchical {
        let cfg = HierarchicalSuperKMeansConfig {
            base,
            max_leaf_size: n / 32,
            iters_per_split: 2,
            ..Default::default()
        };
        let mut model = HierarchicalSuperKMeans::with_config(16, cfg);
        model
            .train_spillable(&mut data, &mut FileTempStorage, options)
            .unwrap();
    } else {
        let mut model = SuperKMeans::with_config(8, 16, base);
        model
            .train_spillable(&mut data, &mut FileTempStorage, options)
            .unwrap();
    }
    PEAK.load(Ordering::SeqCst).saturating_sub(baseline)
}

#[test]
fn training_memory_does_not_scale_with_row_count() {
    measure(8192, false);
    measure(8192, true);
    for hierarchical in [false, true] {
        let small = measure(8192, hierarchical);
        let large = measure(131072, hierarchical);
        eprintln!("hierarchical={hierarchical}: 8192 rows peak={small}, 131072 rows peak={large}");
        assert!(
            large <= small + 16 * 1024,
            "training allocations grew with N: {small} -> {large}"
        );
        assert!(
            large < 256 * 1024,
            "training allocations exceeded workspace + input/model allowance: {large}"
        );
    }
}
