//! PIM (processing-in-memory) simulation: hardware config, static
//! calibration-based bank mapping, per-record timing simulation, cycle
//! conversion, and the naive-vs-remap-vs-ideal balance report.

mod config;
mod cycle;
mod layout;
mod mapping;
mod report;
mod simulate;

pub use config::PimConfig;
pub use cycle::{compute_cycles, derive_json_path, derive_remap_json_path, CycleResult};
pub use layout::{
    run_layout_comparison, run_layout_comparison_parallel, LayoutCalibration,
    LayoutComparisonResult, LayoutTotals, SpanLayout,
};
pub use mapping::{
    build_remap_table, cluster_slot_of, generate_mapping, mapping_quality, MappingParams,
    MappingQuality, MappingStrategy, RemapTable,
};
pub use report::{build_balance_report, print_balance_report, BalanceReportRow};
pub use simulate::{
    run_naive_and_remap_parallel, run_simulation, run_simulation_batched, run_simulation_parallel,
    PimContext, PimResult,
};
