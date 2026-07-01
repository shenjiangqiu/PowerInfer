use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Record {
    pub token: i32,
    pub layer: i32,
    pub batch: i32,
    pub scores: Vec<f32>,
}

impl Record {
    pub fn activated_count(&self) -> usize {
        self.scores.iter().filter(|&&s| s > 0.0).count()
    }
}

pub struct RecordIter<R: Read> {
    reader: R,
    index: u64,
    done: bool,
}

impl<R: Read> RecordIter<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: 0,
            done: false,
        }
    }

    pub fn into_inner(self) -> R {
        self.reader
    }

    pub fn index(&self) -> u64 {
        self.index
    }
}

impl<R: Read> Iterator for RecordIter<R> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut hdr_buf = [0u8; 16];
        match self.reader.read_exact(&mut hdr_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.done = true;
                return None;
            }
            Err(e) => return Some(Err(anyhow::Error::from(e).context("failed to read header"))),
        }

        self.index += 1;

        let token = i32::from_le_bytes(hdr_buf[0..4].try_into().unwrap());
        let layer = i32::from_le_bytes(hdr_buf[4..8].try_into().unwrap());
        let batch = i32::from_le_bytes(hdr_buf[8..12].try_into().unwrap());
        let n_neurons = i32::from_le_bytes(hdr_buf[12..16].try_into().unwrap());

        if n_neurons <= 0 {
            return Some(Err(anyhow::anyhow!(
                "record {} has non-positive n_neurons={}",
                self.index,
                n_neurons
            )));
        }

        let n = n_neurons as usize;
        let data_bytes = n * 4;
        let mut buf = vec![0u8; data_bytes];
        if let Err(e) = self.reader.read_exact(&mut buf) {
            return Some(Err(anyhow::Error::from(e).context(format!(
                "failed to read data for record {}",
                self.index
            ))));
        }

        let scores: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Some(Ok(Record {
            token,
            layer,
            batch,
            scores,
        }))
    }
}

/// Lazily chains multiple .bin files into a single record iterator.
/// Only one file is open at a time.
pub struct ChainFileIter {
    paths: std::vec::IntoIter<PathBuf>,
    current: Option<RecordIter<BufReader<File>>>,
}

impl Iterator for ChainFileIter {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut iter) = self.current {
                match iter.next() {
                    Some(record) => return Some(record),
                    None => {
                        // current file exhausted, drop it (closes file handle)
                        self.current = None;
                    }
                }
            }
            // open next file
            let path = self.paths.next()?;
            match open_single(&path) {
                Ok(iter) => self.current = Some(iter),
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Open a single .bin file.
fn open_single(path: impl AsRef<Path>) -> Result<RecordIter<BufReader<File>>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open {}", path.as_ref().display()))?;
    Ok(RecordIter::new(BufReader::new(file)))
}

/// Open a .bin file or a directory of .bin files.
/// If path is a directory, all .bin files inside are lazily chained —
/// only one file is opened at a time.
pub fn open(path: impl AsRef<Path>) -> Result<ChainFileIter> {
    let path = path.as_ref();
    let paths: Vec<PathBuf> = if path.is_dir() {
        let mut bin_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(path)
            .with_context(|| format!("failed to read directory {}", path.display()))?
        {
            let entry = entry?;
            let p = entry.path();
            if p.extension().map_or(false, |ext| ext == "bin") {
                bin_files.push(p);
            }
        }
        if bin_files.is_empty() {
            anyhow::bail!("no .bin files found in {}", path.display());
        }
        bin_files.sort();
        bin_files
    } else {
        vec![path.to_path_buf()]
    };
    Ok(ChainFileIter {
        paths: paths.into_iter(),
        current: None,
    })
}

pub type LayerHistograms = HashMap<i32, Vec<u64>>;

pub fn compute_histograms<I>(records: I) -> LayerHistograms
where
    I: Iterator<Item = Result<Record>>,
{
    let mut histograms: LayerHistograms = HashMap::new();

    for record in records {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };

        let n = record.scores.len();
        let hist = histograms.entry(record.layer).or_default();
        if hist.len() < n {
            hist.resize(n, 0);
        }
        for (i, &score) in record.scores.iter().enumerate() {
            if score > 0.0 {
                hist[i] += 1;
            }
        }
    }

    histograms
}

pub fn print_first_records<I>(records: &mut I, n: usize)
where
    I: Iterator<Item = Result<Record>>,
{
    println!("idx\ttoken\tlayer\tbatch\tn_neurons\tactivated");
    for i in 0..n {
        match records.next() {
            Some(Ok(r)) => {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    i + 1,
                    r.token,
                    r.layer,
                    r.batch,
                    r.scores.len(),
                    r.activated_count(),
                );
            }
            Some(Err(e)) => {
                println!("{}\tERROR: {}", i + 1, e);
            }
            None => break,
        }
    }
}

pub fn print_histograms(histograms: &LayerHistograms) {
    println!("layer\tposition\tcount");
    let mut layers: Vec<_> = histograms.keys().copied().collect();
    layers.sort_unstable();
    for layer in layers {
        let hist = &histograms[&layer];
        for (pos, &count) in hist.iter().enumerate() {
            if count > 0 {
                println!("{}\t{}\t{}", layer, pos, count);
            }
        }
    }
}

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

/// Compute sparsity statistics from record iterator.
pub fn compute_sparsity<I>(records: I) -> SparsityStats
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
        let activated = record.activated_count() as u64;

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
    println!("Overall sparsity: {:.4}  (activated {}/{} total neurons)",
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
        println!("{}\t{}\t{}\t{:.4}", layer, ls.total_neurons, ls.activated_neurons, ls.sparsity());
    }
}

// ── PIM Simulation ───────────────────────────────────────────────

/// Fixed hardware parameters for PIM simulation.
#[derive(Debug, Clone)]
pub struct PimConfig {
    pub page_size: u64,
    pub banks: u64,
    pub channels: u64,
    pub banks_per_channel: u64,
    pub data_width: u64,
    pub activation_size: u64,
    pub neuron_size: u64,
}

impl Default for PimConfig {
    fn default() -> Self {
        Self {
            page_size: 1024,
            banks: 32 * 32, // 1024 banks = 32 channels × 32 banks each
            channels: 32,
            banks_per_channel: 32,
            data_width: 4,  // f32
            activation_size: 4 * 1024, // 4K
            neuron_size: 11008,
        }
    }
}

/// Pre-computed neuron-to-bank remapping for load balancing.
/// Loaded from a JSON file generated from histogram data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemapTable {
    /// Maps layer_id -> Vec<bank_index> (0..banks) for DOWN projection methods.
    pub down_remap: HashMap<i32, Vec<usize>>,
    /// Maps layer_id -> Vec<bank_in_channel> (0..banks_per_channel) for UP async.
    pub up_remap: HashMap<i32, Vec<usize>>,
}

impl RemapTable {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read remap file {}", path.as_ref().display()))?;
        serde_json::from_str(&s).context("failed to parse remap JSON")
    }
}

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
    pub up_total_iterleave_time: u64,
    pub down_total_interproduct_time_single: u64,
    pub down_total_interproduct_time_two: u64,
    pub down_total_rowwise_bitserial_time_method_1: u64,
    pub down_total_rowwise_bitserial_time_method_2: u64,
    // Balanced metrics (with remap)
    pub up_total_asnc_time_bal: u64,
    pub down_total_method_1_bal: u64,
    // Imbalance overhead stats
    pub up_async_imbalance_overhead: u64,
    pub down_method_1_imbalance_overhead: u64,
}

/// Accumulates PIM timing across records (two-record interleaving).
pub struct PimContext {
    config: PimConfig,
    remap: Option<RemapTable>,

    up_dense: u64,
    up_total_naive_time: u64,
    up_total_asnc_time: u64,
    up_total_iterleave_time: u64,

    down_dense: u64,
    down_total_interproduct_time_single: u64,
    down_total_interproduct_time_two: u64,
    down_total_rowwise_bitserial_time_method_1: u64,
    down_total_rowwise_bitserial_time_method_2: u64,

    // Balanced versions (with remap)
    up_total_asnc_time_bal: u64,
    down_total_method_1_bal: u64,

    // Imbalance overhead: sum of (actual - balanced_if_perfect) per record
    up_async_imbalance_overhead: u64,
    down_method_1_imbalance_overhead: u64,

    total_records: u64,
    total_neurons: u64,
    total_selected_neurons: u64,

    last_round_index: Option<Vec<usize>>,
    last_round_index_down: Option<Vec<usize>>,
}

impl PimContext {
    pub fn new(config: PimConfig, remap: Option<RemapTable>) -> Self {
        Self {
            config,
            remap,
            up_dense: 0,
            up_total_naive_time: 0,
            up_total_asnc_time: 0,
            up_total_iterleave_time: 0,
            down_dense: 0,
            down_total_interproduct_time_single: 0,
            down_total_interproduct_time_two: 0,
            down_total_rowwise_bitserial_time_method_1: 0,
            down_total_rowwise_bitserial_time_method_2: 0,
            up_total_asnc_time_bal: 0,
            down_total_method_1_bal: 0,
            up_async_imbalance_overhead: 0,
            down_method_1_imbalance_overhead: 0,
            total_records: 0,
            total_neurons: 0,
            total_selected_neurons: 0,
            last_round_index: None,
            last_round_index_down: None,
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
        let neuron_size = self.config.neuron_size as usize;

        // ── UP projection ──────────────────────────────────────
        let naive_single_neuron_size = activation_size * data_width; // 16 KB
        let up_rows_per_bank = naive_single_neuron_size / page_size; // 16
        assert_eq!(naive_single_neuron_size % page_size, 0);

        // dense baseline
        self.up_dense += ((neuron_size + banks - 1) / banks * up_rows_per_bank) as u64;

        // 1. naive layout — all banks in a channel activate the same row
        {
            let mut valid_rows: Vec<std::collections::HashSet<usize>> =
                (0..32).map(|_| std::collections::HashSet::new()).collect();
            for &i in indices {
                let channel_id = (i / 32) % 32;
                let row_id = i / 32 / 32;
                valid_rows[channel_id].insert(row_id);
            }
            let max_rows = valid_rows.iter().map(|s| s.len()).max().unwrap_or(0);
            self.up_total_naive_time += (max_rows * up_rows_per_bank) as u64;
        }

        // 2. async layout — each bank activates independently
        {
            let channels = self.config.channels as usize;
            let bpc = self.config.banks_per_channel as usize;
            let mut rows_per_each_bank = vec![vec![0usize; bpc]; channels];
            let up_remap = self.remap.as_ref().and_then(|r| r.up_remap.get(&layer));

            for &i in indices {
                let channel_id = (i / bpc) % channels;
                let bank_id = if let Some(remap) = up_remap {
                    if i < remap.len() { remap[i] } else { i % bpc }
                } else {
                    i % bpc
                };
                rows_per_each_bank[channel_id][bank_id] += 1;
            }
            let max_rows = rows_per_each_bank
                .iter()
                .map(|ch| ch.iter().max().copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            self.up_total_asnc_time += (max_rows * up_rows_per_bank) as u64;

            // Balanced time: ideal if all banks in each channel got equal work
            let mut balanced_max_rows = 0usize;
            for ch in rows_per_each_bank.iter() {
                let ch_total: usize = ch.iter().sum();
                let ch_bal = (ch_total + bpc - 1) / bpc;
                balanced_max_rows = balanced_max_rows.max(ch_bal);
            }
            self.up_total_asnc_time_bal += (balanced_max_rows * up_rows_per_bank) as u64;
            if max_rows > balanced_max_rows {
                self.up_async_imbalance_overhead += ((max_rows - balanced_max_rows) * up_rows_per_bank) as u64;
            }
        }

        // 3. interleave layout — pair two consecutive records
        if let Some(ref last) = self.last_round_index {
            let set1: std::collections::HashSet<usize> = last.iter().copied().collect();
            let set2: std::collections::HashSet<usize> = indices.iter().copied().collect();
            let all: std::collections::HashSet<usize> = set1.union(&set2).copied().collect();

            let mut rows_per_each_bank = vec![0usize; banks];
            for &i in &all {
                rows_per_each_bank[i % banks] += 1;
            }
            let max_rows = rows_per_each_bank.iter().max().copied().unwrap_or(0);
            self.up_total_iterleave_time += (max_rows * up_rows_per_bank) as u64;
            self.last_round_index = None;
        } else {
            self.last_round_index = Some(indices.to_vec());
        }

        // ── DOWN projection ────────────────────────────────────
        // dense baseline
        let down_rows = (neuron_size * data_width + page_size - 1) / page_size;
        self.down_dense += (down_rows * (activation_size / banks)) as u64;

        // 1. inner product — single batch
        {
            let mut row_index_count: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for &i in indices {
                row_index_count.insert(i * data_width / page_size);
            }
            self.down_total_interproduct_time_single +=
                (row_index_count.len() * (activation_size / banks)) as u64;
        }

        // 2. inner product — two batches interleaved
        if let Some(ref last) = self.last_round_index_down {
            let mut all_rows: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
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

        // 3. row-wise bitserial method 1
        {
            let rw_rows_per_bank = (activation_size * data_width) / page_size; // 16
            let mut tasks = vec![0usize; banks];
            let down_remap = self.remap.as_ref().and_then(|r| r.down_remap.get(&layer));

            for &i in indices {
                let bank = if let Some(remap) = down_remap {
                    if i < remap.len() { remap[i] } else { i % banks }
                } else {
                    i % banks
                };
                tasks[bank] += 1;
            }
            let max_tasks = tasks.iter().max().copied().unwrap_or(0);
            self.down_total_rowwise_bitserial_time_method_1 +=
                (max_tasks * rw_rows_per_bank) as u64;

            // Balanced time: ideal if all banks got equal work
            let total_active: usize = tasks.iter().sum();
            let balanced = (total_active + banks - 1) / banks;
            self.down_total_method_1_bal += (balanced * rw_rows_per_bank) as u64;
            if max_tasks > balanced {
                self.down_method_1_imbalance_overhead += ((max_tasks - balanced) * rw_rows_per_bank) as u64;
            }
        }

        // 4. row-wise bitserial method 2
        {
            let group_size = activation_size * data_width / page_size; // 16
            let num_groups = banks / group_size;
            let mut tasks = vec![0usize; num_groups];
            for &i in indices {
                tasks[i % num_groups] += 1;
            }
            let max_tasks = tasks.iter().max().copied().unwrap_or(0);
            self.down_total_rowwise_bitserial_time_method_2 += max_tasks as u64; // rows_per_bank = 1
        }
    }

    /// Flush any remaining unpaired interleaved records.
    pub fn finish(&mut self) {
        let banks = self.config.banks as usize;
        let up_rows_per_bank =
            (self.config.activation_size as usize * self.config.data_width as usize)
                / self.config.page_size as usize;

        // Flush up interleave
        if let Some(ref last) = self.last_round_index {
            let mut rows_per_each_bank = vec![0usize; banks];
            for &i in last {
                rows_per_each_bank[i % banks] += 1;
            }
            let max_rows = rows_per_each_bank.iter().max().copied().unwrap_or(0);
            self.up_total_iterleave_time += (max_rows * up_rows_per_bank) as u64;
            self.last_round_index = None;
        }

        // Flush down interleave (two-batch inner product)
        if let Some(ref last) = self.last_round_index_down {
            let mut row_index_count: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for &i in last {
                row_index_count.insert(
                    i * self.config.data_width as usize / self.config.page_size as usize,
                );
            }
            self.down_total_interproduct_time_single += (row_index_count.len()
                * (self.config.activation_size as usize / banks))
                as u64;
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
            up_total_iterleave_time: self.up_total_iterleave_time,
            down_total_interproduct_time_single: self.down_total_interproduct_time_single,
            down_total_interproduct_time_two: self.down_total_interproduct_time_two,
            down_total_rowwise_bitserial_time_method_1: self
                .down_total_rowwise_bitserial_time_method_1,
            down_total_rowwise_bitserial_time_method_2: self
                .down_total_rowwise_bitserial_time_method_2,
            up_total_asnc_time_bal: self.up_total_asnc_time_bal,
            down_total_method_1_bal: self.down_total_method_1_bal,
            up_async_imbalance_overhead: self.up_async_imbalance_overhead,
            down_method_1_imbalance_overhead: self.down_method_1_imbalance_overhead,
        }
    }
}

/// Run PIM simulation over a record iterator with the given activation threshold.
pub fn run_simulation<I>(records: I, threshold: f32, config: PimConfig, remap: Option<RemapTable>) -> PimResult
where
    I: Iterator<Item = Result<Record>>,
{
    let mut ctx = PimContext::new(config, remap);
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

// ── Filter ──────────────────────────────────────────────────────

/// Filter a record iterator, keeping only records matching the given layer and/or batch.
pub struct FilterIter<I: Iterator<Item = Result<Record>>> {
    inner: I,
    layer: Option<i32>,
    batch: Option<i32>,
}

impl<I: Iterator<Item = Result<Record>>> FilterIter<I> {
    pub fn new(inner: I, layer: Option<i32>, batch: Option<i32>) -> Self {
        Self {
            inner,
            layer,
            batch,
        }
    }
}

impl<I: Iterator<Item = Result<Record>>> Iterator for FilterIter<I> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let record = self.inner.next()?;
            let record = match record {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            if let Some(layer) = self.layer {
                if record.layer != layer {
                    continue;
                }
            }
            if let Some(batch) = self.batch {
                if record.batch != batch {
                    continue;
                }
            }
            return Some(Ok(record));
        }
    }
}

// ── Cycle computation ───────────────────────────────────────────

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
    pub up_total_iterleave_time_row_open: u64,
    pub up_total_iterleave_time_compute: u64,
    pub down_dense_row_open: u64,
    pub down_dense_compute: u64,
    pub down_total_interproduct_time_single_row_open: u64,
    pub down_total_interproduct_time_single_compute: u64,
    pub down_total_interproduct_time_two_row_open: u64,
    pub down_total_interproduct_time_two_compute: u64,
    pub down_total_rowwise_bitserial_time_method_1: u64,
    pub down_total_rowwise_bitserial_time_method_2: u64,
    // Balanced versions
    pub up_total_asnc_time_bal_row_open: u64,
    pub up_total_asnc_time_bal_compute: u64,
    pub down_total_method_1_bal: u64,
    // Imbalance overheads
    pub up_async_imbalance_overhead_row_open: u64,
    pub up_async_imbalance_overhead_compute: u64,
    pub down_method_1_imbalance_overhead: u64,
}

/// Convert PIM simulation stats to cycle counts.
pub fn compute_cycles(stat: &PimResult) -> CycleResult {
    let bandwidth: u64 = 128; // 1024/8 bytes/ns
    let data_width: u64 = 4;
    let row_open: u64 = 56;   // ns
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

    // UP interleave
    let up_total_iterleave_time_row_open = stat.up_total_iterleave_time * row_open;
    let up_total_iterleave_time_compute = stat.up_total_iterleave_time * row_compute;

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

    // DOWN row-wise bitserial
    // multiply + accumulate: 9 × 16 ops, each 56 ns
    let bitserial_factor: u64 = 9 * 16 * 56;
    let down_total_rowwise_bitserial_time_method_1 =
        stat.down_total_rowwise_bitserial_time_method_1 * bitserial_factor;
    let down_total_rowwise_bitserial_time_method_2 =
        stat.down_total_rowwise_bitserial_time_method_2 * bitserial_factor;

    // Balanced metrics
    let up_total_asnc_time_bal_row_open = stat.up_total_asnc_time_bal * row_open;
    let up_total_asnc_time_bal_compute = stat.up_total_asnc_time_bal * row_compute;
    let down_total_method_1_bal = stat.down_total_method_1_bal * bitserial_factor;

    // Imbalance overheads
    let up_async_imbalance_overhead_row_open = stat.up_async_imbalance_overhead * row_open;
    let up_async_imbalance_overhead_compute = stat.up_async_imbalance_overhead * row_compute;
    let down_method_1_imbalance_overhead = stat.down_method_1_imbalance_overhead * bitserial_factor;

    CycleResult {
        gpu_cycle,
        gpu_cycle_sparse,
        up_dense_row_open,
        up_dense_compute,
        up_total_naive_time_row_open,
        up_total_naive_time_compute,
        up_total_asnc_time_row_open,
        up_total_asnc_time_compute,
        up_total_iterleave_time_row_open,
        up_total_iterleave_time_compute,
        down_dense_row_open,
        down_dense_compute,
        down_total_interproduct_time_single_row_open,
        down_total_interproduct_time_single_compute,
        down_total_interproduct_time_two_row_open,
        down_total_interproduct_time_two_compute,
        down_total_rowwise_bitserial_time_method_1,
        down_total_rowwise_bitserial_time_method_2,
        up_total_asnc_time_bal_row_open,
        up_total_asnc_time_bal_compute,
        down_total_method_1_bal,
        up_async_imbalance_overhead_row_open,
        up_async_imbalance_overhead_compute,
        down_method_1_imbalance_overhead,
    }
}
