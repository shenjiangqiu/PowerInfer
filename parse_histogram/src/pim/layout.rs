//! Corrected UP/DOWN physical-layout model, the 4-way naive / non-
//! distributed-async / distributed-no-stagger / distributed-stagger
//! comparison, and time-domain costing (row-open + compute, ns).
//!
//! # UP (gather)
//!
//! Each active neuron is one independent GEMV term: `1x4096 @ 4096x1`. Its
//! full row (`activation_size * data_width` bytes = `up_rows_per_bank` rows
//! at `span=1`) can be *striped* across `span` physical banks instead of
//! stored whole in one. Striping does two independent things:
//!   - it turns `n_banks` physical banks into `n_banks/span` bigger
//!     *logical* banks, each now holding `span`x as many neurons — more
//!     neurons pooled per logical bank means better statistical averaging
//!     (the mechanism [`crate::pim::mapping`]'s remap already balances,
//!     just with less relative variance left over), and
//!   - it lets the `span` physical banks stream a neuron's `up_rows_per_bank`
//!     rows *in parallel*, one row each, so a neuron that used to cost
//!     `up_rows_per_bank` sequential row-reads now costs
//!     `up_rows_per_bank / span`.
//! Since neurons are mutually independent (no shared structure), there is
//! nothing to "stagger" for UP — distributing with or without staggering
//! is the same thing. The bank assignment within a logical unit is the
//! *same* calibrated remap table used everywhere else in this crate, just
//! generated with a matching `span`. A UP neuron's row is either fully
//! wanted or fully not — there is no sub-row structure to exploit, so its
//! time cost is a straight `rows * (row_open + row_compute)`.
//!
//! # DOWN (scatter-accumulate)
//!
//! Its natural layout is transposed: 4096 independent inner products
//! (`1x11008 @ 11008x1`, one per output embedding dim), each needing its
//! own full `11008`-neuron coefficient vector split into
//! `n_row_groups = ceil(neuron_size / row_group_size)` rows of
//! `row_group_size = page_size/data_width` neurons each. A row is
//! skippable *for the row-open* only if *every* neuron in it is inactive —
//! so, unlike UP, DOWN's sparsity lives *inside* a single tensor's own row
//! structure, not across tensors. Three independent levers follow:
//!   - **Clustering**: since a row's skip probability collapses the moment
//!     it contains *any* high-probability neuron, concentrating
//!     high-probability neurons into as few rows as possible (rather than
//!     spreading them) maximizes how many of the *other* rows end up
//!     holding only rare neurons and can be skipped.
//!   - **Distribution + staggering**: striping one tensor's row-groups
//!     across `span` banks is only safe if *which physical bank handles
//!     row-group r* is rotated per tensor (staggered). Every one of the
//!     `activation_size` output-dim tensors shares the *identical*
//!     row-active/inactive pattern (sparsity doesn't depend on the output
//!     dim), so without staggering, whichever bank position happens to
//!     land on a persistently "hot" row-group class never gets to skip —
//!     a severe, structural imbalance that doesn't exist at `span=1`
//!     (already perfectly balanced there). Staggering restores parity with
//!     the `span=1` baseline — damage control for a problem distribution
//!     itself introduces, not a net improvement over not distributing.
//!   - **Sub-row (chunk) compute skipping**: opening a row only requires
//!     that *some* neuron in it be active, but the row-open cost and the
//!     compute cost are two different things. Compute streams a row
//!     `BYTES_PER_CYCLE` bytes (`chunk_size` neurons) at a time; a chunk
//!     whose neurons are *all* inactive contributes nothing and its cycle
//!     can be skipped even though the row around it had to be opened
//!     anyway. This is orthogonal to clustering/distribution/staggering
//!     (which only affect which/how many rows get opened) and is reported
//!     as a separate pair of numbers — with vs. without this skip — so its
//!     standalone effect is visible directly.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::pim::config::PimConfig;
use crate::record::{open_one, Record};

/// DRAM timing constants shared with `pim::cycle` (kept independent here
/// since this module's costing is per-chunk, not just per-row).
///
/// Row-open and compute always ADD, never overlap: in this PIM design the
/// compute unit operates directly in the row buffer, so activating the
/// next row would clobber the data still being computed on. A
/// double-buffered ("pipe") variant was tried and removed for exactly
/// this reason — do not reintroduce it without a second per-bank buffer
/// in the hardware model.
const ROW_OPEN_NS: u64 = 56; // precharge + activate
const BYTES_PER_CYCLE: u64 = 16; // compute throughput
const NS_PER_CYCLE: u64 = 1;

/// One physical-layout configuration for a single projection (UP or DOWN).
#[derive(Debug, Clone, Copy)]
pub struct SpanLayout {
    /// Physical banks striped per tensor. 1 = not distributed.
    pub span: usize,
    /// Whether the row/bank assignment is rotated per tensor (DOWN only;
    /// meaningless for UP since neurons don't share row structure).
    pub stagger: bool,
}

impl SpanLayout {
    pub const fn non_distributed() -> Self {
        Self { span: 1, stagger: false }
    }
}

/// Accumulated busiest-unit totals for one named layout configuration —
/// both raw row/chunk counts (for transparency and debugging) and the
/// resulting time in ns.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LayoutTotals {
    pub name: String,
    /// UP: busiest logical bank's total page-reads (row_open+compute unit).
    pub up_rows: u64,
    /// UP: `up_rows` converted to ns (row_open + compute per row).
    pub up_time_ns: u64,
    /// DOWN: busiest position's row-open count (row-group granularity).
    pub down_rows: u64,
    /// DOWN: busiest position's active-chunk count *with* sub-row compute
    /// skipping (chunks whose neurons are all inactive cost nothing).
    pub down_chunk_compute: u64,
    /// DOWN: same busiest position's chunk count *without* sub-row skip
    /// (every chunk in an opened row counted, i.e. the model before this
    /// optimization) — lets the skip's standalone effect be read off
    /// directly as `down_time_ns_full_row / down_time_ns`.
    pub down_chunk_compute_full_row: u64,
    /// DOWN time with sub-row compute skipping: `down_rows * ROW_OPEN_NS +
    /// down_chunk_compute * CHUNK_COMPUTE_NS`.
    pub down_time_ns: u64,
    /// DOWN time without sub-row compute skipping (comparison baseline):
    /// `down_rows * ROW_OPEN_NS + down_chunk_compute_full_row * CHUNK_COMPUTE_NS`.
    pub down_time_ns_full_row: u64,
}

impl LayoutTotals {
    /// Sum two per-file busiest-unit totals for the same named configuration.
    pub fn merge(mut self, other: Self) -> Self {
        self.up_rows += other.up_rows;
        self.up_time_ns += other.up_time_ns;
        self.down_rows += other.down_rows;
        self.down_chunk_compute += other.down_chunk_compute;
        self.down_chunk_compute_full_row += other.down_chunk_compute_full_row;
        self.down_time_ns += other.down_time_ns;
        self.down_time_ns_full_row += other.down_time_ns_full_row;
        self
    }
}

/// Result of [`run_layout_comparison`]: the true hardware-forced "naive"
/// reference (UP: all banks in a channel lockstep to any active row in it;
/// DOWN: always read every row and compute every chunk, no skip of any
/// kind) computed once per record, plus every requested async
/// configuration's busiest-unit totals.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LayoutComparisonResult {
    pub records: u64,
    pub naive_up_rows: u64,
    pub naive_up_time_ns: u64,
    pub naive_down_rows: u64,
    pub naive_down_time_ns: u64,
    /// GPU reference (bandwidth-limited model, see [`PimConfig::gpu_bandwidth_gbps`]):
    /// total FFN weight bytes a dense GPU kernel must stream per record
    /// (both projections, every neuron), and the same for an ideal sparse
    /// kernel that reads only active neurons' rows/columns. GB/s == B/ns,
    /// so `time_ns = bytes / gpu_bandwidth_gbps`.
    pub gpu_dense_bytes: u64,
    pub gpu_dense_time_ns: u64,
    pub gpu_sparse_bytes: u64,
    pub gpu_sparse_time_ns: u64,
    /// Per-token oracle lower bounds (no static placement can beat these;
    /// both are re-optimized for every record). UP: this record's active
    /// neurons spread perfectly evenly over all banks. DOWN: each bank's
    /// tensors have this record's active coefficients perfectly packed
    /// into the fewest row-groups/chunks.
    pub oracle_up_rows: u64,
    pub oracle_up_time_ns: u64,
    pub oracle_down_rows: u64,
    pub oracle_down_chunks: u64,
    pub oracle_down_time_ns: u64,
    pub configs: Vec<LayoutTotals>,
    /// Per-(token,batch) FFN latency (this file's records only; one entry
    /// per token per layer-record group), one Vec per config, `up_time +
    /// down_time` in ns summed over the token's records. Used for
    /// percentile reporting; not serialized (too large).
    #[serde(skip)]
    pub per_token_latency: Vec<Vec<u64>>,
}

impl LayoutComparisonResult {
    /// Sum two independent per-file results (same `configs` order/names,
    /// since both came from the same `configs` argument). Used to combine
    /// per-file results from [`run_layout_comparison_parallel`].
    pub fn merge(mut self, other: Self) -> Self {
        self.records += other.records;
        self.naive_up_rows += other.naive_up_rows;
        self.naive_up_time_ns += other.naive_up_time_ns;
        self.naive_down_rows += other.naive_down_rows;
        self.naive_down_time_ns += other.naive_down_time_ns;
        self.gpu_dense_bytes += other.gpu_dense_bytes;
        self.gpu_dense_time_ns += other.gpu_dense_time_ns;
        self.gpu_sparse_bytes += other.gpu_sparse_bytes;
        self.gpu_sparse_time_ns += other.gpu_sparse_time_ns;
        self.oracle_up_rows += other.oracle_up_rows;
        self.oracle_up_time_ns += other.oracle_up_time_ns;
        self.oracle_down_rows += other.oracle_down_rows;
        self.oracle_down_chunks += other.oracle_down_chunks;
        self.oracle_down_time_ns += other.oracle_down_time_ns;
        for (a, b) in self.configs.iter_mut().zip(other.configs) {
            *a = std::mem::take(a).merge(b);
        }
        if self.per_token_latency.is_empty() {
            self.per_token_latency = other.per_token_latency;
        } else {
            for (a, b) in self.per_token_latency.iter_mut().zip(other.per_token_latency) {
                a.extend(b);
            }
        }
        self
    }
}

/// One projection's calibration inputs: a per-layer bank remap for UP
/// (same format/semantics as [`crate::pim::mapping::RemapTable::up_remap`],
/// but generated for a specific `span`), and a per-layer clustering
/// permutation for DOWN (see [`crate::pim::mapping::cluster_slot_of`]).
#[derive(Debug, Clone, Default)]
pub struct LayoutCalibration {
    pub up_remap: HashMap<i32, Vec<usize>>,
    pub down_slot_of: HashMap<i32, Vec<usize>>,
}

/// Count how many active neurons land in each of `banks/up_layout.span`
/// logical banks, and return the busiest one's raw neuron count (not yet
/// scaled by the per-neuron row cost).
fn up_busiest_count(indices: &[usize], layer: i32, banks: usize, layout: SpanLayout, remap: Option<&HashMap<i32, Vec<usize>>>) -> usize {
    let n_slots = banks / layout.span;
    let mut per_slot = vec![0usize; n_slots];
    let table = remap.and_then(|r| r.get(&layer));
    for &i in indices {
        let bank = match table {
            Some(t) if i < t.len() => t[i],
            _ => i % banks,
        };
        per_slot[(bank / layout.span) % n_slots] += 1;
    }
    per_slot.into_iter().max().unwrap_or(0)
}

/// For each of DOWN's `n_row_groups`, whether it's active (>=1 active
/// neuron) and how many of its `row_group_size/chunk_size` sub-chunks are
/// individually active — both computed from the same clustering
/// permutation (`None` = natural neuron order) in one pass.
fn down_activity(
    indices: &[usize],
    chunk_size: usize,
    row_group_size: usize,
    n_row_groups: usize,
    slot_of: Option<&[usize]>,
) -> (Vec<bool>, Vec<usize>) {
    let chunks_per_row_group = row_group_size / chunk_size;
    let n_chunks = n_row_groups * chunks_per_row_group;
    let mut chunk_active = vec![false; n_chunks];
    for &i in indices {
        let slot = match slot_of {
            Some(s) if i < s.len() => s[i],
            _ => i,
        };
        let chunk = slot / chunk_size;
        if chunk < n_chunks {
            chunk_active[chunk] = true;
        }
    }

    let mut row_active = vec![false; n_row_groups];
    let mut chunk_counts = vec![0usize; n_row_groups];
    for (rg, (active_slot, count_slot)) in row_active.iter_mut().zip(chunk_counts.iter_mut()).enumerate() {
        let start = rg * chunks_per_row_group;
        let end = (start + chunks_per_row_group).min(n_chunks);
        let count = chunk_active[start..end].iter().filter(|&&a| a).count();
        *count_slot = count;
        *active_slot = count > 0;
    }
    (row_active, chunk_counts)
}

/// Busiest bank-position totals for DOWN this record: row-open count, and
/// chunk-compute count both with and without sub-row skipping. "Busiest"
/// is measured by combined row-open + compute *time* (with skipping),
/// not by either count alone, since the two costs have different per-unit
/// weights. `tensors_per_group` = how many of the `activation_size`
/// output-dim tensors share one physical-bank group (`activation_size *
/// span / banks`). All groups are identical (every tensor sees the same
/// row-active pattern), so simulating one representative group is exact,
/// not a sample.
fn down_busiest_position(
    row_active: &[bool],
    chunk_counts: &[usize],
    chunks_per_row_group: usize,
    tensors_per_group: usize,
    layout: SpanLayout,
) -> (u64, u64, u64) {
    let span = layout.span;
    let mut opens = vec![0u64; span];
    let mut chunks = vec![0u64; span];
    let mut chunks_full = vec![0u64; span];

    for tensor_idx in 0..tensors_per_group {
        let offset = if layout.stagger { tensor_idx % span } else { 0 };
        for (r, &active) in row_active.iter().enumerate() {
            if active {
                let pos = (r + offset) % span;
                opens[pos] += 1;
                chunks[pos] += chunk_counts[r] as u64;
                chunks_full[pos] += chunks_per_row_group as u64;
            }
        }
    }

    let mut best = 0usize;
    let mut best_time = 0u64;
    for p in 0..span {
        let t = opens[p] * ROW_OPEN_NS + chunks[p] * chunk_compute_ns();
        if t >= best_time {
            best_time = t;
            best = p;
        }
    }
    (opens[best], chunks[best], chunks_full[best])
}

fn page_compute_ns(page_size: usize) -> u64 {
    (page_size as u64 * NS_PER_CYCLE).div_ceil(BYTES_PER_CYCLE)
}

fn chunk_compute_ns() -> u64 {
    // one BYTES_PER_CYCLE-wide chunk = exactly one compute cycle.
    NS_PER_CYCLE
}

/// Run every named (up_layout, down_layout, calibration) configuration over
/// the same record stream in one pass, accumulating busiest-unit row/chunk
/// counts and their ns-costed time for each, plus the true hardware-forced
/// naive reference (UP: channel lockstep; DOWN: dense, no skip of any
/// kind) computed once per record. `configs` is `(name, up_layout,
/// up_calibration, down_layout, down_cluster)` — pass `None` for a
/// calibration to fall back to the naive `i % banks` / natural neuron
/// order for that axis.
#[allow(clippy::too_many_arguments)]
pub fn run_layout_comparison<I>(
    records: I,
    threshold: f32,
    config: &PimConfig,
    configs: &[(&str, SpanLayout, Option<&LayoutCalibration>, SpanLayout, Option<&LayoutCalibration>)],
) -> Result<LayoutComparisonResult>
where
    I: Iterator<Item = Result<Record>>,
{
    let banks = config.banks as usize;
    let page_size = config.page_size as usize;
    let weight_bits = config.weight_bits as usize;
    let activation_size = config.activation_size as usize;
    let channels = config.channels as usize;
    let bpc = config.banks_per_channel as usize;

    // All width-dependent geometry derives from weight_bits so sub-byte
    // precisions (int6, int4) work: a 16-byte chunk holds floor(128/bits)
    // neurons (sub-byte widths pack with padding), a 1KB row is 64 such
    // chunks, and one neuron's 4096-coefficient row spans
    // ceil(4096*bits / 8*1024) physical rows.
    let up_rows_per_bank = (activation_size * weight_bits).div_ceil(8 * page_size);
    let up_row_time_ns = ROW_OPEN_NS + page_compute_ns(page_size);
    let chunk_size = ((BYTES_PER_CYCLE as usize * 8) / weight_bits).max(1);
    let chunks_per_row_group = page_size / BYTES_PER_CYCLE as usize;
    let row_group_size = chunk_size * chunks_per_row_group;

    let mut totals: Vec<LayoutTotals> = configs
        .iter()
        .map(|(name, ..)| LayoutTotals { name: name.to_string(), ..Default::default() })
        .collect();
    let mut result = LayoutComparisonResult { configs: totals.drain(..).collect(), ..Default::default() };
    result.per_token_latency = vec![Vec::new(); configs.len()];
    // Per-token latency accumulation. The dump is layer-major (all batch
    // positions of layer 0, then layer 1, ...), so one (token, batch)'s
    // records are NOT contiguous — accumulate in a map instead.
    let mut tok_lat: HashMap<(i32, i32), Vec<u64>> = HashMap::new();

    for record in records {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };
        result.records += 1;
        // The intermediate size differs per model (LLaMA 11008, Mistral-based
        // Bamboo 14336) — take it from the record, never from the config.
        let neuron_size = record.scores.len();
        let n_row_groups = neuron_size.div_ceil(row_group_size);
        let down_rows_dense = n_row_groups; // dense: every row-group read, every chunk computed
        let indices: Vec<usize> = record
            .scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > threshold)
            .map(|(i, _)| i)
            .collect();

        let token_acc = tok_lat
            .entry((record.token, record.batch))
            .or_insert_with(|| vec![0u64; configs.len()]);

        // Naive UP: every bank in a channel locksteps to any active row in
        // that channel (same logic as the existing up_total_naive_time).
        {
            let mut valid_rows: Vec<std::collections::HashSet<usize>> =
                (0..channels).map(|_| std::collections::HashSet::new()).collect();
            for &i in &indices {
                let channel_id = (i / bpc) % channels;
                let row_id = i / bpc / channels;
                valid_rows[channel_id].insert(row_id);
            }
            let max_rows = valid_rows.iter().map(|s| s.len()).max().unwrap_or(0);
            result.naive_up_rows += (max_rows * up_rows_per_bank) as u64;
        }
        // Naive DOWN: dense, every row-group always read and every chunk
        // always computed regardless of activity (no skip of any kind).
        let naive_down_rows_this = (down_rows_dense * (activation_size / banks)) as u64;
        result.naive_down_rows += naive_down_rows_this;

        // GPU reference: bandwidth-limited streaming of the FFN weights for
        // this record. Dense reads both full matrices (up n×d + down d×n);
        // an ideal sparse kernel reads only the active neurons' rows/columns
        // of each. One record = one (token, layer), same scope as the PIM
        // numbers above.
        result.gpu_dense_bytes += 2 * ((neuron_size * activation_size * weight_bits).div_ceil(8)) as u64;
        result.gpu_sparse_bytes += 2 * ((indices.len() * activation_size * weight_bits).div_ceil(8)) as u64;

        // Per-token oracles (lower bounds, re-optimized every record).
        // UP: active neurons spread perfectly evenly over all banks.
        let a = indices.len();
        let oracle_up_rows_this = (a.div_ceil(banks) * up_rows_per_bank) as u64;
        result.oracle_up_rows += oracle_up_rows_this;
        // DOWN: each bank's tensors have the active coefficients perfectly
        // packed into the fewest row-groups and chunks.
        let tensors_per_bank = activation_size / banks;
        let oracle_opens = (a.div_ceil(row_group_size).max(usize::from(a > 0)) * tensors_per_bank) as u64;
        let oracle_chunks = (a.div_ceil(chunk_size) * tensors_per_bank) as u64;
        result.oracle_down_rows += oracle_opens;
        result.oracle_down_chunks += oracle_chunks;

        for (i, (_, up_layout, up_calib, down_layout, down_calib)) in configs.iter().enumerate() {
            let up_rows_per_neuron = up_rows_per_bank.div_ceil(up_layout.span);
            let up_remap = up_calib.map(|c| &c.up_remap);
            let up_count = up_busiest_count(&indices, record.layer, banks, *up_layout, up_remap);
            let up_rows_this = (up_count * up_rows_per_neuron) as u64;
            result.configs[i].up_rows += up_rows_this;
            let up_time_this = up_rows_this * up_row_time_ns;
            result.configs[i].up_time_ns += up_time_this;

            let down_slot_of = down_calib.and_then(|c| c.down_slot_of.get(&record.layer));
            let (row_active, chunk_counts) =
                down_activity(&indices, chunk_size, row_group_size, n_row_groups, down_slot_of.map(|v| v.as_slice()));
            let tensors_per_group = activation_size * down_layout.span / banks;
            let (opens, chunks, chunks_full) =
                down_busiest_position(&row_active, &chunk_counts, chunks_per_row_group, tensors_per_group, *down_layout);
            result.configs[i].down_rows += opens;
            result.configs[i].down_chunk_compute += chunks;
            result.configs[i].down_chunk_compute_full_row += chunks_full;
            let down_time_this = opens * ROW_OPEN_NS + chunks * chunk_compute_ns();
            result.configs[i].down_time_ns += down_time_this;
            result.configs[i].down_time_ns_full_row += opens * ROW_OPEN_NS + chunks_full * chunk_compute_ns();
            token_acc[i] += up_time_this + down_time_this;
        }
    }
    for v in tok_lat.into_values() {
        for (c, x) in v.into_iter().enumerate() {
            result.per_token_latency[c].push(x);
        }
    }

    result.oracle_up_time_ns = result.oracle_up_rows * up_row_time_ns;
    result.oracle_down_time_ns =
        result.oracle_down_rows * ROW_OPEN_NS + result.oracle_down_chunks * chunk_compute_ns();
    result.naive_up_time_ns = result.naive_up_rows * up_row_time_ns;
    result.naive_down_time_ns =
        result.naive_down_rows * ROW_OPEN_NS + result.naive_down_rows * (chunks_per_row_group as u64) * chunk_compute_ns();
    // GB/s == bytes/ns, so time follows directly from the byte totals.
    result.gpu_dense_time_ns = (result.gpu_dense_bytes as f64 / config.gpu_bandwidth_gbps) as u64;
    result.gpu_sparse_time_ns = (result.gpu_sparse_bytes as f64 / config.gpu_bandwidth_gbps) as u64;

    Ok(result)
}

/// Same as [`run_layout_comparison`], but processes every `.bin` file in
/// `dir` on a separate rayon worker and merges the busiest-unit totals —
/// each file is an independent record stream, so this is exact, not an
/// approximation.
#[allow(clippy::too_many_arguments)]
pub fn run_layout_comparison_parallel(
    dir: impl AsRef<Path>,
    threshold: f32,
    config: &PimConfig,
    configs: &[(&str, SpanLayout, Option<&LayoutCalibration>, SpanLayout, Option<&LayoutCalibration>)],
) -> Result<LayoutComparisonResult> {
    let paths = crate::record::list_bin_files(dir)?;
    paths
        .par_iter()
        .map(|p| -> Result<LayoutComparisonResult> {
            let iter = open_one(p)?;
            run_layout_comparison(iter, threshold, config, configs)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(LayoutComparisonResult::merge)
        .context("no .bin files processed")
}
