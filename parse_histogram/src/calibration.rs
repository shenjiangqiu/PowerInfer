use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::record::{open_one, Record};

/// Per-layer activation statistics used to derive calibration probabilities:
/// how many records were observed for a layer, and how many times each
/// neuron position had a positive (active) score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationStats {
    pub total_records: u64,
    pub activated_counts: Vec<u64>,
}

impl LayerActivationStats {
    /// P(neuron i is active), estimated from the calibration set.
    pub fn probabilities(&self) -> Vec<f64> {
        if self.total_records == 0 {
            return vec![0.0; self.activated_counts.len()];
        }
        let total = self.total_records as f64;
        self.activated_counts
            .iter()
            .map(|&c| c as f64 / total)
            .collect()
    }
}

pub type LayerActivationTable = HashMap<i32, LayerActivationStats>;

/// Compute per-layer, per-neuron activation counts plus the number of
/// records seen per layer (the denominator for activation probability).
/// This is the input to the offline calibration-based bank mapping.
/// `threshold` must match the model's own `powerinfer.sparse_threshold`
/// (0.0 for plain ReLU models) — see [`Record`] docs.
pub fn compute_activation_stats<I>(records: I, threshold: f32) -> LayerActivationTable
where
    I: Iterator<Item = Result<Record>>,
{
    let mut table: LayerActivationTable = HashMap::new();

    for record in records {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };

        let n = record.scores.len();
        let entry = table.entry(record.layer).or_insert_with(|| LayerActivationStats {
            total_records: 0,
            activated_counts: Vec::new(),
        });
        if entry.activated_counts.len() < n {
            entry.activated_counts.resize(n, 0);
        }
        entry.total_records += 1;
        for (i, &score) in record.scores.iter().enumerate() {
            if score > threshold {
                entry.activated_counts[i] += 1;
            }
        }
    }

    table
}

/// Merge two activation tables (sum matching layers' counts). Used to
/// combine independent per-file results from [`compute_activation_stats_parallel`].
pub fn merge_activation_tables(mut a: LayerActivationTable, b: LayerActivationTable) -> LayerActivationTable {
    for (layer, stats_b) in b {
        let entry = a.entry(layer).or_insert_with(|| LayerActivationStats {
            total_records: 0,
            activated_counts: Vec::new(),
        });
        if entry.activated_counts.len() < stats_b.activated_counts.len() {
            entry.activated_counts.resize(stats_b.activated_counts.len(), 0);
        }
        entry.total_records += stats_b.total_records;
        for (i, c) in stats_b.activated_counts.into_iter().enumerate() {
            entry.activated_counts[i] += c;
        }
    }
    a
}

/// Same as [`compute_activation_stats`], but processes every `.bin` file in
/// `dir` (see [`crate::list_bin_files`]) on a separate rayon worker and
/// merges the results — each file is a fully independent record stream, so
/// this is exact, not an approximation, and scales with however many CPU
/// cores are available rather than being limited to one file's I/O
/// throughput at a time.
pub fn compute_activation_stats_parallel(dir: impl AsRef<Path>, threshold: f32) -> Result<LayerActivationTable> {
    let paths = crate::record::list_bin_files(dir)?;
    Ok(paths
        .par_iter()
        .map(|p| -> Result<LayerActivationTable> {
            let iter = open_one(p)?;
            Ok(compute_activation_stats(iter, threshold))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(merge_activation_tables)
        .unwrap_or_default())
}
