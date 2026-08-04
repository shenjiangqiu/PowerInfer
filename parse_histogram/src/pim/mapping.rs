use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::calibration::LayerActivationTable;

/// Pre-computed neuron-to-bank remapping for load balancing.
/// Loaded from a JSON file generated from histogram data.
///
/// Both `up_remap` and `down_remap` map layer_id -> Vec<bank_index>, where
/// `bank_index` is a *global* bank id in `0..banks` (not restricted to a
/// single channel). UP and DOWN weights live in physically separate bank
/// arrays, so the two tables are independent permutations even though they
/// are derived from the same per-neuron activation probabilities (a neuron
/// is selected or not by the shared predictor; UP and DOWN are free to
/// place that neuron's row/column in different banks of their own array).
/// Every value is a multiple of the placement `span` used at generation
/// time (1 unless the target stores wide, multi-bank-spanning rows).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemapTable {
    /// Maps layer_id -> Vec<bank_index> (0..banks) for the DOWN async method.
    pub down_remap: HashMap<i32, Vec<usize>>,
    /// Maps layer_id -> Vec<bank_index> (0..banks) for the UP async method.
    pub up_remap: HashMap<i32, Vec<usize>>,
}

impl RemapTable {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read remap file {}", path.as_ref().display()))?;
        serde_json::from_str(&s).context("failed to parse remap JSON")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("failed to serialize remap table")?;
        fs::write(path.as_ref(), json)
            .with_context(|| format!("failed to write remap file {}", path.as_ref().display()))?;
        Ok(())
    }
}

// ── Calibration-based bank mapping ──────────────────────────────
//
// Goal: choose a static (offline, one-time) neuron -> bank assignment that
// balances the *expected* per-bank load under the async per-bank PIM
// execution model, using per-neuron activation probabilities estimated
// from an offline calibration set (see `crate::calibration`).
//
// A "placement unit" occupies `span` consecutive banks starting at a bank
// index that is a multiple of `span` (bank_start % span == 0). span=1 is
// the common case (a neuron's row/column fits in one bank); span>1 models
// wide rows that must be striped across an aligned group of banks (e.g.
// wider hidden dimensions or higher-precision weights than the evaluated
// config needs). The mapping algorithm treats each span-aligned group of
// banks as one "slot" and assigns whole neurons to slots.

/// Parameters controlling how neurons are packed into banks.
#[derive(Debug, Clone, Copy)]
pub struct MappingParams {
    /// Total number of physical banks in the target array.
    pub n_banks: usize,
    /// Banks per placement unit (must divide n_banks). 1 = one bank/neuron.
    pub span: usize,
    /// Soft cap on neurons per slot, expressed as a multiple of the mean
    /// occupancy (n_neurons / n_slots). 1.0 = perfectly even count: no
    /// slot may hold more neurons than any other by more than rounding.
    /// >1.0 relaxes the storage-balance constraint to let the packer chase
    /// load balance more aggressively.
    pub capacity_slack: f64,
}

impl Default for MappingParams {
    fn default() -> Self {
        Self {
            n_banks: 1024,
            span: 1,
            capacity_slack: 1.2,
        }
    }
}

/// Strategy used to turn a probability vector into a bank assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingStrategy {
    /// Baseline: i % n_slots. Ignores the calibration data entirely.
    Modulo,
    /// Sort neurons by probability (descending), deal them round-robin
    /// across slots. Uses calibration data but not running load state.
    SortedRoundRobin,
    /// Greedy longest-processing-time-first: sort neurons by probability
    /// (descending), repeatedly assign the next neuron to the currently
    /// least-loaded slot that is still under its capacity cap. This is the
    /// classic LPT list-scheduling heuristic for makespan minimization
    /// (within 4/3 of optimal in the worst case, typically much closer).
    GreedyLpt,
}

/// Per-slot occupancy/load after a mapping has been generated — used both
/// to report storage/load balance quality and to feed the imbalance-loss
/// report (see [`mapping_quality`]).
#[derive(Debug, Clone, Serialize)]
pub struct MappingQuality {
    pub n_slots: usize,
    pub neuron_count_min: usize,
    pub neuron_count_max: usize,
    pub neuron_count_mean: f64,
    pub neuron_count_stddev: f64,
    pub load_min: f64,
    pub load_max: f64,
    pub load_mean: f64,
    pub load_stddev: f64,
    /// max_load / mean_load — 1.0 is perfectly balanced; the theoretical
    /// slowdown a naive scheme pays versus the ideal (see the report
    /// command for the *measured* slowdown, which also depends on how
    /// requests are batched at runtime).
    pub load_imbalance_ratio: f64,
}

/// Assign each of `probs.len()` neurons to a bank, aligned to `span`.
/// Returns a `Vec<usize>` of the same length, each entry a bank index in
/// `0..params.n_banks` and a multiple of `params.span`.
pub fn generate_mapping(
    probs: &[f64],
    params: &MappingParams,
    strategy: MappingStrategy,
) -> Vec<usize> {
    assert!(params.span >= 1 && params.n_banks % params.span == 0);
    let n_slots = params.n_banks / params.span;
    let n = probs.len();
    assert!(n_slots > 0);

    let bank_of_slot = |slot: usize| slot * params.span;

    match strategy {
        MappingStrategy::Modulo => (0..n).map(|i| bank_of_slot(i % n_slots)).collect(),

        MappingStrategy::SortedRoundRobin => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let mut bank_of = vec![0usize; n];
            for (rank, &neuron) in order.iter().enumerate() {
                bank_of[neuron] = bank_of_slot(rank % n_slots);
            }
            bank_of
        }

        MappingStrategy::GreedyLpt => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

            let avg_count = n as f64 / n_slots as f64;
            let cap = ((avg_count * params.capacity_slack).ceil() as usize).max(1);

            let mut load = vec![0.0f64; n_slots];
            let mut count = vec![0usize; n_slots];
            let mut bank_of = vec![0usize; n];

            for &neuron in &order {
                // Least-loaded slot that still has capacity. n_slots is at
                // most a few thousand and this runs once per model, so a
                // linear scan keeps the implementation simple and obviously
                // correct rather than reaching for a float-keyed heap.
                let mut best_slot = 0usize;
                let mut best_load = f64::INFINITY;
                for slot in 0..n_slots {
                    if count[slot] < cap && load[slot] < best_load {
                        best_load = load[slot];
                        best_slot = slot;
                    }
                }
                load[best_slot] += probs[neuron];
                count[best_slot] += 1;
                bank_of[neuron] = bank_of_slot(best_slot);
            }
            bank_of
        }
    }
}

/// Compute storage/load balance statistics for a generated mapping —
/// used to compare strategies and to report the expected imbalance before
/// picking a final mapping.
pub fn mapping_quality(probs: &[f64], bank_of: &[usize], params: &MappingParams) -> MappingQuality {
    let n_slots = params.n_banks / params.span;
    let mut count = vec![0usize; n_slots];
    let mut load = vec![0.0f64; n_slots];
    for (neuron, &bank) in bank_of.iter().enumerate() {
        let slot = bank / params.span;
        count[slot] += 1;
        load[slot] += probs[neuron];
    }

    let mean_count = count.iter().sum::<usize>() as f64 / n_slots as f64;
    let var_count = count
        .iter()
        .map(|&c| (c as f64 - mean_count).powi(2))
        .sum::<f64>()
        / n_slots as f64;

    let mean_load = load.iter().sum::<f64>() / n_slots as f64;
    let var_load = load.iter().map(|&l| (l - mean_load).powi(2)).sum::<f64>() / n_slots as f64;
    let max_load = load.iter().cloned().fold(0.0f64, f64::max);

    MappingQuality {
        n_slots,
        neuron_count_min: count.iter().copied().min().unwrap_or(0),
        neuron_count_max: count.iter().copied().max().unwrap_or(0),
        neuron_count_mean: mean_count,
        neuron_count_stddev: var_count.sqrt(),
        load_min: load.iter().cloned().fold(f64::INFINITY, f64::min),
        load_max: max_load,
        load_mean: mean_load,
        load_stddev: var_load.sqrt(),
        load_imbalance_ratio: if mean_load > 0.0 {
            max_load / mean_load
        } else {
            1.0
        },
    }
}

/// Build a full [`RemapTable`] from calibration data, generating an
/// independent mapping per layer for both the UP and DOWN targets. UP and
/// DOWN share the same per-neuron probabilities (one predictor decides
/// activation for both) but are packed independently since they occupy
/// separate physical bank arrays.
/// Order neurons by activation probability, descending, and return
/// `slot_of[neuron] = its rank` (0 = highest probability).
///
/// This is *not* a bank assignment — it is the permutation used to decide
/// which neurons land in the same DOWN row-group (see [`crate::pim::layout`]):
/// clustering the highest-probability neurons into the lowest-ranked slots
/// concentrates them into as few row-groups as possible, maximizing how
/// often the remaining, colder row-groups can be skipped entirely.
pub fn cluster_slot_of(probs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut slot_of = vec![0usize; probs.len()];
    for (slot, &neuron) in order.iter().enumerate() {
        slot_of[neuron] = slot;
    }
    slot_of
}

pub fn build_remap_table(
    stats: &LayerActivationTable,
    up_params: &MappingParams,
    down_params: &MappingParams,
    strategy: MappingStrategy,
) -> RemapTable {
    let mut up_remap = HashMap::new();
    let mut down_remap = HashMap::new();

    let mut layers: Vec<_> = stats.keys().copied().collect();
    layers.sort_unstable();

    for layer in layers {
        let probs = stats[&layer].probabilities();
        up_remap.insert(layer, generate_mapping(&probs, up_params, strategy));
        down_remap.insert(layer, generate_mapping(&probs, down_params, strategy));
    }

    RemapTable {
        down_remap,
        up_remap,
    }
}
