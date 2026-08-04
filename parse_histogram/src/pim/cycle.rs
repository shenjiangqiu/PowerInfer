use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::pim::simulate::PimResult;

/// Derive the simulation JSON path from the input path:
/// - file → same dir, `.json` extension
/// - dir  → `<dir>/simulation.json`
/// If `remap` is true, use `simulation_remap.json`
pub fn derive_json_path(input_path: &Path) -> PathBuf {
    if input_path.is_dir() {
        input_path.join("simulation.json")
    } else {
        input_path.with_extension("json")
    }
}

pub fn derive_remap_json_path(input_path: &Path) -> PathBuf {
    if input_path.is_dir() {
        input_path.join("simulation_remap.json")
    } else {
        input_path.with_extension("remap.json")
    }
}

/// Cycle-count result converted from PIM simulation stats.
#[derive(Debug, Clone, Serialize)]
pub struct CycleResult {
    pub gpu_cycle: u64,
    pub gpu_cycle_sparse: u64,
    pub up_dense_row_open: u64,
    pub up_dense_compute: u64,
    pub up_total_naive_time_row_open: u64,
    pub up_total_naive_time_compute: u64,
    pub up_total_asnc_time_row_open: u64,
    pub up_total_asnc_time_compute: u64,
    pub down_dense_row_open: u64,
    pub down_dense_compute: u64,
    pub down_total_interproduct_time_single_row_open: u64,
    pub down_total_interproduct_time_single_compute: u64,
    pub down_total_interproduct_time_two_row_open: u64,
    pub down_total_interproduct_time_two_compute: u64,
    pub down_total_async_time_row_open: u64,
    pub down_total_async_time_compute: u64,
    // Balanced versions
    pub up_total_asnc_time_bal_row_open: u64,
    pub up_total_asnc_time_bal_compute: u64,
    pub down_total_async_time_bal_row_open: u64,
    pub down_total_async_time_bal_compute: u64,
    // Imbalance overheads
    pub up_async_imbalance_overhead_row_open: u64,
    pub up_async_imbalance_overhead_compute: u64,
    pub down_async_imbalance_overhead_row_open: u64,
    pub down_async_imbalance_overhead_compute: u64,
    // Batched-round metrics (batch_group consecutive same-layer records
    // unioned into one round; batch_group=1 matches the per-record numbers)
    pub batch_group: u64,
    pub up_total_batch_time_row_open: u64,
    pub up_total_batch_time_compute: u64,
    pub up_total_batch_time_bal_row_open: u64,
    pub up_total_batch_time_bal_compute: u64,
    pub up_batch_imbalance_overhead_row_open: u64,
    pub up_batch_imbalance_overhead_compute: u64,
    pub down_total_batch_time_row_open: u64,
    pub down_total_batch_time_compute: u64,
    pub down_total_batch_time_bal_row_open: u64,
    pub down_total_batch_time_bal_compute: u64,
    pub down_batch_imbalance_overhead_row_open: u64,
    pub down_batch_imbalance_overhead_compute: u64,
}

/// Convert PIM simulation stats to cycle counts.
pub fn compute_cycles(stat: &PimResult) -> CycleResult {
    let bandwidth: u64 = 128; // 1024/8 bytes/ns
    let data_width: u64 = 4;
    let row_open: u64 = 56; // ns
    let row_compute: u64 = 64; // 1024/16 ns
    let enable_gate = false;

    // GPU baseline
    let mut gpu_cycle = stat.total_neurons * 4096 * data_width / bandwidth;
    if enable_gate {
        gpu_cycle *= 3;
    } else {
        gpu_cycle *= 2;
    }
    let gpu_cycle_sparse = stat.total_selected_neurons * 4096 * data_width / bandwidth * 2;

    // UP dense
    let up_dense_row_open = stat.up_dense * row_open;
    let up_dense_compute = stat.up_dense * row_compute;

    // UP naive
    let up_total_naive_time_row_open = stat.up_total_naive_time * row_open;
    let up_total_naive_time_compute = stat.up_total_naive_time * row_compute;

    // UP async
    let up_total_asnc_time_row_open = stat.up_total_asnc_time * row_open;
    let up_total_asnc_time_compute = stat.up_total_asnc_time * row_compute;

    // DOWN dense
    let down_dense_row_open = stat.down_dense * row_open;
    let down_dense_compute = stat.down_dense * row_compute;

    // DOWN inner-product single
    let down_total_interproduct_time_single_row_open =
        stat.down_total_interproduct_time_single * row_open;
    let down_total_interproduct_time_single_compute =
        stat.down_total_interproduct_time_single * row_compute;

    // DOWN inner-product two
    let down_total_interproduct_time_two_row_open =
        stat.down_total_interproduct_time_two * row_open;
    let down_total_interproduct_time_two_compute =
        stat.down_total_interproduct_time_two * row_compute;

    // DOWN async (row-owning layout, normal per-bank adder — no bit-serial
    // accumulation): costed the same way as UP async, since it has the
    // same "read N rows sequentially in the busiest bank" shape.
    let down_total_async_time_row_open = stat.down_total_async_time * row_open;
    let down_total_async_time_compute = stat.down_total_async_time * row_compute;

    // Balanced metrics
    let up_total_asnc_time_bal_row_open = stat.up_total_asnc_time_bal * row_open;
    let up_total_asnc_time_bal_compute = stat.up_total_asnc_time_bal * row_compute;
    let down_total_async_time_bal_row_open = stat.down_total_async_time_bal * row_open;
    let down_total_async_time_bal_compute = stat.down_total_async_time_bal * row_compute;

    // Imbalance overheads
    let up_async_imbalance_overhead_row_open = stat.up_async_imbalance_overhead * row_open;
    let up_async_imbalance_overhead_compute = stat.up_async_imbalance_overhead * row_compute;
    let down_async_imbalance_overhead_row_open = stat.down_async_imbalance_overhead * row_open;
    let down_async_imbalance_overhead_compute = stat.down_async_imbalance_overhead * row_compute;

    // Batched-round metrics
    let up_total_batch_time_row_open = stat.up_total_batch_time * row_open;
    let up_total_batch_time_compute = stat.up_total_batch_time * row_compute;
    let up_total_batch_time_bal_row_open = stat.up_total_batch_time_bal * row_open;
    let up_total_batch_time_bal_compute = stat.up_total_batch_time_bal * row_compute;
    let up_batch_imbalance_overhead_row_open = stat.up_batch_imbalance_overhead * row_open;
    let up_batch_imbalance_overhead_compute = stat.up_batch_imbalance_overhead * row_compute;
    let down_total_batch_time_row_open = stat.down_total_batch_time * row_open;
    let down_total_batch_time_compute = stat.down_total_batch_time * row_compute;
    let down_total_batch_time_bal_row_open = stat.down_total_batch_time_bal * row_open;
    let down_total_batch_time_bal_compute = stat.down_total_batch_time_bal * row_compute;
    let down_batch_imbalance_overhead_row_open = stat.down_batch_imbalance_overhead * row_open;
    let down_batch_imbalance_overhead_compute = stat.down_batch_imbalance_overhead * row_compute;

    CycleResult {
        gpu_cycle,
        gpu_cycle_sparse,
        up_dense_row_open,
        up_dense_compute,
        up_total_naive_time_row_open,
        up_total_naive_time_compute,
        up_total_asnc_time_row_open,
        up_total_asnc_time_compute,
        down_dense_row_open,
        down_dense_compute,
        down_total_interproduct_time_single_row_open,
        down_total_interproduct_time_single_compute,
        down_total_interproduct_time_two_row_open,
        down_total_interproduct_time_two_compute,
        down_total_async_time_row_open,
        down_total_async_time_compute,
        up_total_asnc_time_bal_row_open,
        up_total_asnc_time_bal_compute,
        down_total_async_time_bal_row_open,
        down_total_async_time_bal_compute,
        up_async_imbalance_overhead_row_open,
        up_async_imbalance_overhead_compute,
        down_async_imbalance_overhead_row_open,
        down_async_imbalance_overhead_compute,
        batch_group: stat.batch_group,
        up_total_batch_time_row_open,
        up_total_batch_time_compute,
        up_total_batch_time_bal_row_open,
        up_total_batch_time_bal_compute,
        up_batch_imbalance_overhead_row_open,
        up_batch_imbalance_overhead_compute,
        down_total_batch_time_row_open,
        down_total_batch_time_compute,
        down_total_batch_time_bal_row_open,
        down_total_batch_time_bal_compute,
        down_batch_imbalance_overhead_row_open,
        down_batch_imbalance_overhead_compute,
    }
}
