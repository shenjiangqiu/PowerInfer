use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// One (token, layer, batch) predictor-score record parsed from a `.bin`
/// dump. `scores[i]` is the raw pre-activation predictor score; whether
/// neuron `i` is actually selected depends on the model's own sparse
/// threshold (PowerInfer GGUF key `powerinfer.sparse_threshold`, read into
/// `hparams.sparse_pred_threshold` and compared as `score >= threshold` at
/// inference time — see `ggml.c`'s sparse mul_mat/axpy kernels). This
/// defaults to 0.0 for plain-ReLU models (e.g. ReluLLaMA), but techniques
/// like ProSparse deliberately train with a *nonzero* threshold to push
/// sparsity higher than vanilla ReLU would give — callers analyzing such a
/// model must pass that threshold explicitly, or every count derived here
/// will silently overcount "active" neurons (and thus undercount
/// sparsity) relative to what the model actually computes.
#[derive(Debug, Clone)]
pub struct Record {
    pub token: i32,
    pub layer: i32,
    pub batch: i32,
    pub scores: Vec<f32>,
}

impl Record {
    pub fn activated_count(&self, threshold: f32) -> usize {
        self.scores.iter().filter(|&&s| s > threshold).count()
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
            return Some(Err(anyhow::Error::from(e)
                .context(format!("failed to read data for record {}", self.index))));
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

/// List the individual `.bin` files a path resolves to: itself if it's a
/// single file, or every `.bin` file inside it (sorted) if it's a
/// directory. This is what [`open`] chains serially; callers that want to
/// process files in parallel (e.g. with rayon) should use this instead and
/// open+stream each file independently — every `.bin` file is a
/// self-contained, independent record stream, so there is nothing to
/// synchronize between them.
pub fn list_bin_files(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
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
    Ok(paths)
}

/// Open exactly one `.bin` file as a record stream (no directory chaining).
/// Pairs with [`list_bin_files`] for parallel, one-file-per-worker
/// processing.
pub fn open_one(path: impl AsRef<Path>) -> Result<RecordIter<BufReader<File>>> {
    open_single(path)
}

/// Open a .bin file or a directory of .bin files.
/// If path is a directory, all .bin files inside are lazily chained —
/// only one file is opened at a time. Serial by construction; see
/// [`list_bin_files`] to process the same files in parallel instead.
pub fn open(path: impl AsRef<Path>) -> Result<ChainFileIter> {
    let paths = list_bin_files(path)?;
    Ok(ChainFileIter {
        paths: paths.into_iter(),
        current: None,
    })
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
