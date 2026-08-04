use std::collections::HashMap;

use anyhow::Result;

use crate::record::Record;

/// Per-layer sparsity statistics.
#[derive(Debug, Clone)]
pub struct LayerSparsity {
    pub total_neurons: u64,
    pub activated_neurons: u64,
}

impl LayerSparsity {
    pub fn sparsity(&self) -> f64 {
        if self.total_neurons == 0 {
            return 0.0;
        }
        1.0 - (self.activated_neurons as f64) / (self.total_neurons as f64)
    }
}

/// Sparsity statistics: overall + per-layer.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    pub overall: LayerSparsity,
    pub per_layer: HashMap<i32, LayerSparsity>,
}

/// Compute sparsity statistics from record iterator. `threshold` must
/// match the model's own `powerinfer.sparse_threshold` (0.0 for plain
/// ReLU models); passing the wrong threshold silently over- or
/// under-counts active neurons — see [`Record`] docs.
pub fn compute_sparsity<I>(records: I, threshold: f32) -> SparsityStats
where
    I: Iterator<Item = Result<Record>>,
{
    let mut overall_total: u64 = 0;
    let mut overall_activated: u64 = 0;
    let mut per_layer: HashMap<i32, LayerSparsity> = HashMap::new();

    for record in records {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };

        let total = record.scores.len() as u64;
        let activated = record.activated_count(threshold) as u64;

        overall_total += total;
        overall_activated += activated;

        let entry = per_layer.entry(record.layer).or_insert(LayerSparsity {
            total_neurons: 0,
            activated_neurons: 0,
        });
        entry.total_neurons += total;
        entry.activated_neurons += activated;
    }

    SparsityStats {
        overall: LayerSparsity {
            total_neurons: overall_total,
            activated_neurons: overall_activated,
        },
        per_layer,
    }
}

/// Print sparsity statistics in tab-separated format.
pub fn print_sparsity(stats: &SparsityStats) {
    println!(
        "Overall sparsity: {:.4}  (activated {}/{} total neurons)",
        stats.overall.sparsity(),
        stats.overall.activated_neurons,
        stats.overall.total_neurons,
    );

    println!();
    println!("layer\ttotal_neurons\tactivated\tsparsity");
    let mut layers: Vec<_> = stats.per_layer.keys().copied().collect();
    layers.sort_unstable();
    for layer in layers {
        let ls = &stats.per_layer[&layer];
        println!(
            "{}\t{}\t{}\t{:.4}",
            layer,
            ls.total_neurons,
            ls.activated_neurons,
            ls.sparsity()
        );
    }
}
