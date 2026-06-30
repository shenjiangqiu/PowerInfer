use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use parse_histogram::{self, compute_histograms, print_first_records, print_histograms, FilterIter};

#[derive(Parser)]
#[command(name = "parse_histogram", about = "Parse binary neuron scores and build per-layer histograms of positive-score positions.")]
struct Args {
    /// Path to the binary file
    filepath: PathBuf,

    /// Save histogram as JSON to this file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Print info of the first N records
    #[arg(short = 'p', long)]
    print_first: Option<usize>,

    /// Filter by layer (only process records with this layer id)
    #[arg(short, long)]
    layer: Option<i32>,

    /// Filter by batch (only process records with this batch id)
    #[arg(short, long)]
    batch: Option<i32>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let iter = parse_histogram::open(&args.filepath)?;
    let mut filtered = FilterIter::new(iter, args.layer, args.batch);

    if let Some(n) = args.print_first {
        print_first_records(&mut filtered, n);
        return Ok(());
    }

    let histograms = compute_histograms(filtered);

    eprintln!("layers: {}", histograms.len());

    if let Some(out_path) = &args.output {
        let json = serde_json::to_string_pretty(&histograms)
            .context("failed to serialize histograms to JSON")?;
        fs::write(out_path, json)
            .with_context(|| format!("failed to write {}", out_path.display()))?;
        eprintln!("saved to {}", out_path.display());
    } else {
        eprintln!();
        print_histograms(&histograms);
    }

    Ok(())
}
