mod calibration;
mod histogram;
mod pim;
mod record;
mod sparsity;

pub use calibration::{
    compute_activation_stats, compute_activation_stats_parallel, merge_activation_tables,
    LayerActivationStats, LayerActivationTable,
};
pub use histogram::{compute_histograms, print_first_records, print_histograms, LayerHistograms};
pub use pim::{
    build_balance_report, build_remap_table, cluster_slot_of, compute_cycles, derive_json_path,
    derive_remap_json_path, generate_mapping, mapping_quality, print_balance_report,
    run_layout_comparison, run_layout_comparison_parallel, run_naive_and_remap_parallel,
    run_simulation, run_simulation_batched, run_simulation_parallel, BalanceReportRow,
    CycleResult, LayoutCalibration, LayoutComparisonResult, LayoutTotals, MappingParams,
    MappingQuality, MappingStrategy, PimConfig, PimContext, PimResult, RemapTable, SpanLayout,
};
pub use record::{list_bin_files, open, open_one, ChainFileIter, FilterIter, Record, RecordIter};
pub use sparsity::{compute_sparsity, print_sparsity, LayerSparsity, SparsityStats};
