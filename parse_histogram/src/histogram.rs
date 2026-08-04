use std::collections::HashMap;

use anyhow::Result;

use crate::record::Record;

pub type LayerHistograms = HashMap<i32, Vec<u64>>;

/// `threshold` must match the model's own `powerinfer.sparse_threshold`
/// (0.0 for plain ReLU models) — see [`Record`] docs.
pub fn compute_histograms<I>(records: I, threshold: f32) -> LayerHistograms
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
            if score > threshold {
                hist[i] += 1;
            }
        }
    }

    histograms
}

pub fn print_first_records<I>(records: &mut I, n: usize, threshold: f32)
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
                    r.activated_count(threshold),
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
