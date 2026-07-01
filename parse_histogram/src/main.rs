mod cli;

use std::fs;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Args, Commands};
use parse_histogram::{self, compute_histograms, compute_sparsity, print_first_records, print_histograms, print_sparsity, run_simulation, FilterIter, PimConfig};

fn main() -> Result<()> {
    let args = Args::parse();

    let iter = parse_histogram::open(&args.file)?;
    let mut filtered = FilterIter::new(iter, args.layer, args.batch);

    match args.command {
        Commands::Print { count } => {
            print_first_records(&mut filtered, count);
        }
        Commands::Histogram { output } => {
            let histograms = compute_histograms(filtered);

            eprintln!("layers: {}", histograms.len());

            if let Some(out_path) = &output {
                let json = serde_json::to_string_pretty(&histograms)
                    .context("failed to serialize histograms to JSON")?;
                fs::write(out_path, json)
                    .with_context(|| format!("failed to write {}", out_path.display()))?;
                eprintln!("saved to {}", out_path.display());
            } else {
                eprintln!();
                print_histograms(&histograms);
            }
        }
        Commands::Sparsity => {
            let stats = compute_sparsity(filtered);
            print_sparsity(&stats);
        }
        Commands::Simulate { threshold, output } => {
            let result = run_simulation(filtered, threshold, PimConfig::default());

            if let Some(out_path) = &output {
                let json = serde_json::to_string_pretty(&result)
                    .context("failed to serialize simulation result to JSON")?;
                fs::write(out_path, json)
                    .with_context(|| format!("failed to write {}", out_path.display()))?;
                eprintln!("saved to {}", out_path.display());
            } else {
                let json = serde_json::to_string_pretty(&result)
                    .context("failed to serialize simulation result to JSON")?;
                println!("{}", json);
            }
        }
    }

    Ok(())
}
