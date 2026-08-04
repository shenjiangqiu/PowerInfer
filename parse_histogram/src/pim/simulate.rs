use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pim::config::PimConfig;
use crate::pim::mapping::RemapTable;
use crate::record::{open_one, Record};

/// Aggregated simulation result (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimResult {
    pub total_records: u64,
    pub total_neurons: u64,
    pub total_selected_neurons: u64,
    pub up_dense: u64,
    pub down_dense: u64,
    pub up_total_naive_time: u64,
    pub up_total_asnc_time: u64,
    pub down_total_interproduct_time_single: u64,
    pub down_total_interproduct_time_two: u64,
    pub down_total_async_time: u64,
    // Balanced metrics (ideal, given the same active-neuron sets)
    pub up_total_asnc_time_bal: u64,
    pub down_total_async_time_bal: u64,
    // Imbalance overhead stats: sum of (actual - ideal) per record
    pub up_async_imbalance_overhead: u64,
    pub down_async_imbalance_overhead: u64,
    // Batched-round metrics: `batch_group` consecutive same-layer records
    // are unioned into one round before dispatch (see PimContext docs).
    // batch_group=1 degenerates to the plain per-record async numbers above.
    pub batch_group: u64,
    pub up_total_batch_time: u64,
    pub up_total_batch_time_bal: u64,
    pub up_batch_imbalance_overhead: u64,
    pub down_total_batch_time: u64,
    pub down_total_batch_time_bal: u64,
    pub down_batch_imbalance_overhead: u64,
}

impl PimResult {
    /// Merge two independent partial results (e.g. from processing
    /// different `.bin` files) into one, as if they'd been accumulated by
    /// the same [`PimContext`] in sequence. Every field here is a simple
    /// per-record accumulator (including the "balanced"/ideal and
    /// imbalance-overhead fields, which are already sums of per-record
    /// ideals), so summing is exact — except `batch_group`, which is a
    /// configuration parameter, not a count, and must match between the
    /// two results being merged.
    ///
    /// One caveat: `{up,down}_total_batch_time*` batch a few consecutive
    /// same-layer records together (see `PimContext` docs). Merging
    /// per-file results flushes any pending partial batch at each file's
    /// end rather than continuing it into the next file, unlike a single
    /// serial pass over the same records — a negligible boundary effect
    /// given there are only as many boundaries as there are files.
    pub fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.batch_group, other.batch_group,
            "cannot merge PimResults computed with different batch_group values"
        );
        Self {
            total_records: self.total_records + other.total_records,
            total_neurons: self.total_neurons + other.total_neurons,
            total_selected_neurons: self.total_selected_neurons + other.total_selected_neurons,
            up_dense: self.up_dense + other.up_dense,
            down_dense: self.down_dense + other.down_dense,
            up_total_naive_time: self.up_total_naive_time + other.up_total_naive_time,
            up_total_asnc_time: self.up_total_asnc_time + other.up_total_asnc_time,
            down_total_interproduct_time_single: self.down_total_interproduct_time_single
                + other.down_total_interproduct_time_single,
            down_total_interproduct_time_two: self.down_total_interproduct_time_two + other.down_total_interproduct_time_two,
            down_total_async_time: self.down_total_async_time + other.down_total_async_time,
            up_total_asnc_time_bal: self.up_total_asnc_time_bal + other.up_total_asnc_time_bal,
            down_total_async_time_bal: self.down_total_async_time_bal + other.down_total_async_time_bal,
            up_async_imbalance_overhead: self.up_async_imbalance_overhead + other.up_async_imbalance_overhead,
            down_async_imbalance_overhead: self.down_async_imbalance_overhead + other.down_async_imbalance_overhead,
            batch_group: self.batch_group,
            up_total_batch_time: self.up_total_batch_time + other.up_total_batch_time,
            up_total_batch_time_bal: self.up_total_batch_time_bal + other.up_total_batch_time_bal,
            up_batch_imbalance_overhead: self.up_batch_imbalance_overhead + other.up_batch_imbalance_overhead,
            down_total_batch_time: self.down_total_batch_time + other.down_total_batch_time,
            down_total_batch_time_bal: self.down_total_batch_time_bal + other.down_total_batch_time_bal,
            down_batch_imbalance_overhead: self.down_batch_imbalance_overhead + other.down_batch_imbalance_overhead,
        }
    }
}

/// Accumulates PIM timing across records.
///
/// UP and DOWN use opposite physical layouts (see module docs on
/// [`RemapTable`]): UP owns one full weight *row* per active neuron inside
/// a single bank (gather: every active neuron reads the same broadcast
/// input, so per-bank work scales with how many active neurons a bank
/// owns — this is what the async/remap model targets). DOWN's dense
/// layout instead stripes every neuron's weight *column* across all banks
/// by output-embedding slice (scatter/accumulate onto a shared output), so
/// every bank always touches the same set of active neurons in lockstep —
/// that layout has no per-bank load to balance. `down_total_async_time`
/// models the alternative DOWN layout that mirrors UP's (one neuron's full
/// column owned by one bank, accumulated locally with a normal per-bank
/// adder); it is the DOWN metric that a remap table can actually improve.
///
/// `batch_group` controls an additional pair of metrics
/// (`{up,down}_total_batch_time`) that model dispatching `batch_group`
/// consecutive same-layer records (e.g. several prefill positions, or
/// several in-flight requests) to the banks together: the busiest bank is
/// computed once for the *union* of active neurons across the whole
/// group, rather than once per record. A neuron requested by more than one
/// record in the group still costs one row-read that round (the resident
/// row is reused for every request that wants it), so grouping can only
/// ever reduce or match the sum of per-record busiest-bank costs — the
/// question this answers is how much of that reduction survives once you
/// also account for imbalance.
pub struct PimContext {
    config: PimConfig,
    remap: Option<RemapTable>,
    batch_group: usize,

    up_dense: u64,
    up_total_naive_time: u64,
    up_total_asnc_time: u64,

    down_dense: u64,
    down_total_interproduct_time_single: u64,
    down_total_interproduct_time_two: u64,
    down_total_async_time: u64,

    // Balanced versions (ideal, given the same active-neuron sets)
    up_total_asnc_time_bal: u64,
    down_total_async_time_bal: u64,

    // Imbalance overhead: sum of (actual - balanced_if_perfect) per record
    up_async_imbalance_overhead: u64,
    down_async_imbalance_overhead: u64,

    // Batched-round metrics (see struct docs)
    up_total_batch_time: u64,
    up_total_batch_time_bal: u64,
    up_batch_imbalance_overhead: u64,
    down_total_batch_time: u64,
    down_total_batch_time_bal: u64,
    down_batch_imbalance_overhead: u64,

    total_records: u64,
    total_neurons: u64,
    total_selected_neurons: u64,

    // Pending records waiting to be unioned into one batched round. Kept
    // per-layer: a batch never spans a layer boundary, since that would mix
    // two disjoint neuron-index spaces into one "round". `down_pending`
    // reuses the up-side pending records rather than duplicating storage —
    // both batches are built from the same per-record indices, just
    // dispatched through a different bank_of() function at flush time.
    pending: Vec<Vec<usize>>,
    pending_layer: Option<i32>,

    // Single-slot buffer used only by the (legacy, unrelated-to-remap)
    // transposed-layout two-batch inner-product metric.
    last_round_index_down: Option<Vec<usize>>,
}

/// Assign each index to a bank via `bank_of`, and return
/// `(busiest_bank_count, ideal_busiest_bank_count)` for this round — the
/// ideal being what a perfectly even split of the same total count across
/// `banks` banks would need.
fn round_stats(indices: impl Iterator<Item = usize>, banks: usize, bank_of: impl Fn(usize) -> usize) -> (usize, usize) {
    let mut per_bank = vec![0usize; banks];
    let mut total = 0usize;
    for i in indices {
        per_bank[bank_of(i)] += 1;
        total += 1;
    }
    let max_count = per_bank.iter().max().copied().unwrap_or(0);
    let ideal_count = total.div_ceil(banks);
    (max_count, ideal_count)
}

impl PimContext {
    pub fn new(config: PimConfig, remap: Option<RemapTable>) -> Self {
        Self::with_batch_group(config, remap, 1)
    }

    /// `batch_group` = number of consecutive same-layer records unioned
    /// into one dispatched round for the `{up,down}_total_batch_time`
    /// metrics. Must be >= 1; 1 means "no batching" (one record per round).
    pub fn with_batch_group(config: PimConfig, remap: Option<RemapTable>, batch_group: usize) -> Self {
        assert!(batch_group >= 1);
        Self {
            config,
            remap,
            batch_group,
            up_dense: 0,
            up_total_naive_time: 0,
            up_total_asnc_time: 0,
            down_dense: 0,
            down_total_interproduct_time_single: 0,
            down_total_interproduct_time_two: 0,
            down_total_async_time: 0,
            up_total_asnc_time_bal: 0,
            down_total_async_time_bal: 0,
            up_async_imbalance_overhead: 0,
            down_async_imbalance_overhead: 0,
            up_total_batch_time: 0,
            up_total_batch_time_bal: 0,
            up_batch_imbalance_overhead: 0,
            down_total_batch_time: 0,
            down_total_batch_time_bal: 0,
            down_batch_imbalance_overhead: 0,
            total_records: 0,
            total_neurons: 0,
            total_selected_neurons: 0,
            pending: Vec::new(),
            pending_layer: None,
            last_round_index_down: None,
        }
    }

    fn up_bank_of<'a>(&'a self, layer: i32) -> impl Fn(usize) -> usize + 'a {
        let bpc = self.config.banks_per_channel as usize;
        let channels = self.config.channels as usize;
        let up_remap = self.remap.as_ref().and_then(|r| r.up_remap.get(&layer));
        move |i: usize| match up_remap {
            Some(remap) if i < remap.len() => remap[i],
            _ => {
                let channel_id = (i / bpc) % channels;
                channel_id * bpc + i % bpc
            }
        }
    }

    fn down_bank_of<'a>(&'a self, layer: i32) -> impl Fn(usize) -> usize + 'a {
        let banks = self.config.banks as usize;
        let down_remap = self.remap.as_ref().and_then(|r| r.down_remap.get(&layer));
        move |i: usize| match down_remap {
            Some(remap) if i < remap.len() => remap[i],
            _ => i % banks,
        }
    }

    /// Feed one record's active neuron indices into the simulation.
    pub fn compute_time(&mut self, layer: i32, total: usize, indices: &[usize]) {
        let total = total as u64;
        let active = indices.len() as u64;
        self.total_records += 1;
        self.total_neurons += total;
        self.total_selected_neurons += active;

        let banks = self.config.banks as usize;
        let page_size = self.config.page_size as usize;
        let data_width = self.config.data_width as usize;
        let activation_size = self.config.activation_size as usize;
        // The intermediate size differs per model (e.g. LLaMA 11008 vs
        // Mistral-based Bamboo 14336) — always take it from the record
        // itself, never from the config, or dense baselines silently use
        // the wrong dimension.
        let neuron_size = total as usize;

        // ── UP projection ──────────────────────────────────────
        let naive_single_neuron_size = activation_size * data_width; // 16 KB
        let up_rows_per_bank = naive_single_neuron_size / page_size; // 16
        assert_eq!(naive_single_neuron_size % page_size, 0);

        // dense baseline
        self.up_dense += ((neuron_size + banks - 1) / banks * up_rows_per_bank) as u64;

        // 1. naive layout — all banks in a channel activate the same row.
        {
            let channels = self.config.channels as usize;
            let bpc = self.config.banks_per_channel as usize;
            let mut valid_rows: Vec<std::collections::HashSet<usize>> =
                (0..channels).map(|_| std::collections::HashSet::new()).collect();
            for &i in indices {
                let channel_id = (i / bpc) % channels;
                let row_id = i / bpc / channels;
                valid_rows[channel_id].insert(row_id);
            }
            let max_rows = valid_rows.iter().map(|s| s.len()).max().unwrap_or(0);
            self.up_total_naive_time += (max_rows * up_rows_per_bank) as u64;
        }

        // 2. async layout — each bank activates independently, one record
        // (== one round) at a time. A neuron's row is free to live in *any*
        // bank of the UP array via remap; the ideal/balanced reference
        // assumes the same freedom.
        {
            let (max_rows, ideal_rows) = round_stats(indices.iter().copied(), banks, self.up_bank_of(layer));
            self.up_total_asnc_time += (max_rows * up_rows_per_bank) as u64;
            self.up_total_asnc_time_bal += (ideal_rows * up_rows_per_bank) as u64;
            if max_rows > ideal_rows {
                self.up_async_imbalance_overhead += ((max_rows - ideal_rows) * up_rows_per_bank) as u64;
            }
        }

        // ── DOWN projection ────────────────────────────────────
        // dense baseline — transposed layout: every bank stores a slice of
        // *every* neuron's output-embedding range, so all banks always
        // touch the same set of active neurons (see struct docs). No
        // per-bank imbalance is possible here; sparsity just shrinks the
        // shared row count every bank pays.
        let down_rows = (neuron_size * data_width + page_size - 1) / page_size;
        self.down_dense += (down_rows * (activation_size / banks)) as u64;

        // 1. inner product — single batch (transposed layout, lockstep)
        {
            let mut row_index_count: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for &i in indices {
                row_index_count.insert(i * data_width / page_size);
            }
            self.down_total_interproduct_time_single +=
                (row_index_count.len() * (activation_size / banks)) as u64;
        }

        // 2. inner product — two batches interleaved (transposed layout).
        // Unrelated to remap (see struct docs); kept as a fixed pair-of-2.
        if let Some(ref last) = self.last_round_index_down {
            let mut all_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &i in last {
                all_rows.insert(i * data_width / page_size);
            }
            for &i in indices {
                all_rows.insert(i * data_width / page_size);
            }
            self.down_total_interproduct_time_two +=
                (all_rows.len() * (activation_size / banks)) as u64;
            self.last_round_index_down = None;
        } else {
            self.last_round_index_down = Some(indices.to_vec());
        }

        // 3. async layout — the row-owning alternative DOWN layout: each
        // active neuron's full output column lives in one bank (free to be
        // any bank in the DOWN array via remap) and is accumulated there
        // with a normal per-bank adder. Same shape as UP's async model, so
        // it suffers the same load-imbalance problem and benefits from the
        // same calibration-based remap.
        {
            let async_rows_per_bank = (activation_size * data_width) / page_size; // 16
            let (max_tasks, ideal_tasks) = round_stats(indices.iter().copied(), banks, self.down_bank_of(layer));
            self.down_total_async_time += (max_tasks * async_rows_per_bank) as u64;
            self.down_total_async_time_bal += (ideal_tasks * async_rows_per_bank) as u64;
            if max_tasks > ideal_tasks {
                self.down_async_imbalance_overhead += ((max_tasks - ideal_tasks) * async_rows_per_bank) as u64;
            }
        }

        // 4. batched round — union this record's active set with up to
        // `batch_group - 1` other same-layer records before computing the
        // busiest bank, for both UP and DOWN's row-owning layouts.
        if self.pending_layer != Some(layer) {
            self.flush_batch();
            self.pending_layer = Some(layer);
        }
        self.pending.push(indices.to_vec());
        if self.pending.len() >= self.batch_group {
            self.flush_batch();
        }
    }

    fn flush_batch(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let layer = self.pending_layer.expect("pending records imply a pending layer");

        let page_size = self.config.page_size as usize;
        let data_width = self.config.data_width as usize;
        let activation_size = self.config.activation_size as usize;
        let banks = self.config.banks as usize;
        let up_rows_per_bank = (activation_size * data_width) / page_size;
        let down_rows_per_bank = (activation_size * data_width) / page_size;

        let mut union: HashSet<usize> = HashSet::new();
        for v in &self.pending {
            union.extend(v.iter().copied());
        }

        let (up_max, up_ideal) = round_stats(union.iter().copied(), banks, self.up_bank_of(layer));
        self.up_total_batch_time += (up_max * up_rows_per_bank) as u64;
        self.up_total_batch_time_bal += (up_ideal * up_rows_per_bank) as u64;
        if up_max > up_ideal {
            self.up_batch_imbalance_overhead += ((up_max - up_ideal) * up_rows_per_bank) as u64;
        }

        let (down_max, down_ideal) = round_stats(union.iter().copied(), banks, self.down_bank_of(layer));
        self.down_total_batch_time += (down_max * down_rows_per_bank) as u64;
        self.down_total_batch_time_bal += (down_ideal * down_rows_per_bank) as u64;
        if down_max > down_ideal {
            self.down_batch_imbalance_overhead += ((down_max - down_ideal) * down_rows_per_bank) as u64;
        }

        self.pending.clear();
    }

    /// Flush any remaining partial batch / unpaired interleaved records.
    pub fn finish(&mut self) {
        self.flush_batch();

        // Flush down interleave (two-batch inner product)
        if let Some(ref last) = self.last_round_index_down {
            let banks = self.config.banks as usize;
            let mut row_index_count: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for &i in last {
                row_index_count
                    .insert(i * self.config.data_width as usize / self.config.page_size as usize);
            }
            self.down_total_interproduct_time_single +=
                (row_index_count.len() * (self.config.activation_size as usize / banks)) as u64;
            self.last_round_index_down = None;
        }
    }

    pub fn into_result(mut self) -> PimResult {
        self.finish();
        PimResult {
            total_records: self.total_records,
            total_neurons: self.total_neurons,
            total_selected_neurons: self.total_selected_neurons,
            up_dense: self.up_dense,
            down_dense: self.down_dense,
            up_total_naive_time: self.up_total_naive_time,
            up_total_asnc_time: self.up_total_asnc_time,
            down_total_interproduct_time_single: self.down_total_interproduct_time_single,
            down_total_interproduct_time_two: self.down_total_interproduct_time_two,
            down_total_async_time: self.down_total_async_time,
            up_total_asnc_time_bal: self.up_total_asnc_time_bal,
            down_total_async_time_bal: self.down_total_async_time_bal,
            up_async_imbalance_overhead: self.up_async_imbalance_overhead,
            down_async_imbalance_overhead: self.down_async_imbalance_overhead,
            batch_group: self.batch_group as u64,
            up_total_batch_time: self.up_total_batch_time,
            up_total_batch_time_bal: self.up_total_batch_time_bal,
            up_batch_imbalance_overhead: self.up_batch_imbalance_overhead,
            down_total_batch_time: self.down_total_batch_time,
            down_total_batch_time_bal: self.down_total_batch_time_bal,
            down_batch_imbalance_overhead: self.down_batch_imbalance_overhead,
        }
    }
}

/// Run PIM simulation over a record iterator with the given activation threshold.
pub fn run_simulation<I>(
    records: I,
    threshold: f32,
    config: PimConfig,
    remap: Option<RemapTable>,
) -> PimResult
where
    I: Iterator<Item = Result<Record>>,
{
    run_simulation_batched(records, threshold, config, remap, 1)
}

/// Same as [`run_simulation`], but also computes the `{up,down}_total_batch_time`
/// metrics by unioning `batch_group` consecutive same-layer records into one
/// dispatched round. `batch_group=1` is equivalent to [`run_simulation`].
pub fn run_simulation_batched<I>(
    records: I,
    threshold: f32,
    config: PimConfig,
    remap: Option<RemapTable>,
    batch_group: usize,
) -> PimResult
where
    I: Iterator<Item = Result<Record>>,
{
    let mut ctx = PimContext::with_batch_group(config, remap, batch_group);
    for record in records {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };
        let indices: Vec<usize> = record
            .scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > threshold)
            .map(|(i, _)| i)
            .collect();
        ctx.compute_time(record.layer, record.scores.len(), &indices);
    }
    ctx.into_result()
}

/// Same as [`run_simulation_batched`], but processes every `.bin` file in
/// `dir` (see [`crate::list_bin_files`]) on a separate rayon worker and
/// merges the results with [`PimResult::merge`]. `remap` is cloned once
/// per file (cheap relative to a file's I/O) since each worker needs its
/// own [`PimContext`].
pub fn run_simulation_parallel(
    dir: impl AsRef<Path>,
    threshold: f32,
    config: PimConfig,
    remap: Option<RemapTable>,
    batch_group: usize,
) -> Result<PimResult> {
    let paths = crate::record::list_bin_files(dir)?;
    paths
        .par_iter()
        .map(|p| -> Result<PimResult> {
            let iter = open_one(p)?;
            Ok(run_simulation_batched(iter, threshold, config.clone(), remap.clone(), batch_group))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(PimResult::merge)
        .context("no .bin files processed")
}

/// Compute the naive (`remap: None`) and calibrated-remap results together
/// in a single pass per file — two [`PimContext`] accumulators fed the same
/// record stream, avoiding a second read of every file — then run that
/// per-file pass on a separate rayon worker for each `.bin` file and merge
/// both result streams independently. This is what backs the `report`
/// command and `run_all`'s "naive+remap report" stage.
pub fn run_naive_and_remap_parallel(
    dir: impl AsRef<Path>,
    threshold: f32,
    config: PimConfig,
    remap: RemapTable,
    batch_group: usize,
) -> Result<(PimResult, PimResult)> {
    let paths = crate::record::list_bin_files(dir)?;
    let (naive, remapped): (Vec<PimResult>, Vec<PimResult>) = paths
        .par_iter()
        .map(|p| -> Result<(PimResult, PimResult)> {
            let iter = open_one(p)?;
            let mut ctx_naive = PimContext::with_batch_group(config.clone(), None, batch_group);
            let mut ctx_remap = PimContext::with_batch_group(config.clone(), Some(remap.clone()), batch_group);
            for record in iter {
                let record = match record {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Warning: {}", e);
                        continue;
                    }
                };
                let indices: Vec<usize> = record
                    .scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s > threshold)
                    .map(|(i, _)| i)
                    .collect();
                ctx_naive.compute_time(record.layer, record.scores.len(), &indices);
                ctx_remap.compute_time(record.layer, record.scores.len(), &indices);
            }
            Ok((ctx_naive.into_result(), ctx_remap.into_result()))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    let naive = naive.into_iter().reduce(PimResult::merge).context("no .bin files processed")?;
    let remapped = remapped.into_iter().reduce(PimResult::merge).context("no .bin files processed")?;
    Ok((naive, remapped))
}
