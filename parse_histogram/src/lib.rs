use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

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

pub fn open(path: impl AsRef<Path>) -> Result<RecordIter<BufReader<File>>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open {}", path.as_ref().display()))?;
    Ok(RecordIter::new(BufReader::new(file)))
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
