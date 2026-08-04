use serde::Serialize;

use crate::pim::simulate::PimResult;

/// One row of the naive-vs-remap-vs-ideal comparison (task: quantify the
/// performance left on the table by bank imbalance before committing to a
/// mapping, and how close the chosen mapping gets to the ideal).
#[derive(Debug, Clone, Serialize)]
pub struct BalanceReportRow {
    pub name: &'static str,
    /// Rows of work in the busiest bank with no remap (i % banks).
    pub naive_rows: u64,
    /// Rows of work in the busiest bank with the calibrated remap applied.
    pub remap_rows: u64,
    /// Rows of work in the busiest bank if load were perfectly divisible
    /// (total active rows / n_banks) — the same for naive and remap since
    /// it only depends on which neurons were active, not where they live.
    pub ideal_rows: u64,
    /// naive_rows / remap_rows — how much faster the remap makes the
    /// busiest bank vs. doing nothing.
    pub speedup_vs_naive: f64,
    /// remap_rows / ideal_rows — how close the remap gets to the
    /// information-theoretic best case (1.0 = perfect).
    pub remap_vs_ideal: f64,
}

/// Build the naive/remap/ideal comparison for UP-async and DOWN-async.
/// `naive` and `remapped` must come from the *same* record stream, one
/// simulated with `remap: None` and the other with the calibrated table.
pub fn build_balance_report(naive: &PimResult, remapped: &PimResult) -> Vec<BalanceReportRow> {
    let row = |name: &'static str, naive_rows: u64, remap_rows: u64, ideal_rows: u64| {
        BalanceReportRow {
            name,
            naive_rows,
            remap_rows,
            ideal_rows,
            speedup_vs_naive: if remap_rows > 0 {
                naive_rows as f64 / remap_rows as f64
            } else {
                1.0
            },
            remap_vs_ideal: if ideal_rows > 0 {
                remap_rows as f64 / ideal_rows as f64
            } else {
                1.0
            },
        }
    };

    let mut rows = vec![
        row(
            "up_async",
            naive.up_total_asnc_time,
            remapped.up_total_asnc_time,
            remapped.up_total_asnc_time_bal,
        ),
        row(
            "down_async",
            naive.down_total_async_time,
            remapped.down_total_async_time,
            remapped.down_total_async_time_bal,
        ),
    ];

    // Only meaningful once batch_group > 1 (both results must have been
    // produced by run_simulation_batched with the same group size).
    if naive.batch_group > 1 || remapped.batch_group > 1 {
        rows.push(row(
            "up_batch",
            naive.up_total_batch_time,
            remapped.up_total_batch_time,
            remapped.up_total_batch_time_bal,
        ));
        rows.push(row(
            "down_batch",
            naive.down_total_batch_time,
            remapped.down_total_batch_time,
            remapped.down_total_batch_time_bal,
        ));
    }

    rows
}

/// Print the naive/remap/ideal comparison as a readable table.
pub fn print_balance_report(rows: &[BalanceReportRow]) {
    println!(
        "{:<12}{:>16}{:>16}{:>16}{:>16}{:>16}",
        "metric", "naive(rows)", "remap(rows)", "ideal(rows)", "speedup", "remap/ideal"
    );
    for r in rows {
        println!(
            "{:<12}{:>16}{:>16}{:>16}{:>15.3}x{:>15.3}x",
            r.name, r.naive_rows, r.remap_rows, r.ideal_rows, r.speedup_vs_naive, r.remap_vs_ideal
        );
    }
}
