use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

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
