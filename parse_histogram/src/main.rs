mod cli;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Args, Commands};
use parse_histogram::{
    self, compute_cycles, compute_histograms, compute_sparsity, derive_json_path,
    print_first_records, print_histograms, print_sparsity, run_simulation,
    FilterIter, PimConfig, PimResult,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // to-cycle handles its own file I/O (may run simulation first)
    if let Commands::ToCycle { threshold, output } = &args.command {
        return cmd_to_cycle(&args, *threshold, output.as_ref());
    }

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
            tracing::info!(
                "Running PIM simulation (threshold={})...",
                threshold
            );
            let result = run_simulation(filtered, threshold, PimConfig::default());

            let out_path = output.unwrap_or_else(|| derive_json_path(&args.file));
            let json = serde_json::to_string_pretty(&result)
                .context("failed to serialize simulation result to JSON")?;
            fs::write(&out_path, json)
                .with_context(|| format!("failed to write {}", out_path.display()))?;
            tracing::info!(
                "Simulation done — {} records, saved to {}",
                result.total_records,
                out_path.display()
            );
        }
        Commands::ToCycle { .. } => unreachable!(),
    }

    Ok(())
}

fn cmd_to_cycle(args: &Args, threshold: f32, output: Option<&PathBuf>) -> Result<()> {
    let sim_path = derive_json_path(&args.file);

    let pim_result: PimResult = if sim_path.exists() {
        tracing::info!("Using cached simulation JSON: {}", sim_path.display());
        let json_str = fs::read_to_string(&sim_path)
            .with_context(|| format!("failed to read {}", sim_path.display()))?;
        serde_json::from_str(&json_str)
            .context("failed to parse simulation JSON")?
    } else {
        tracing::info!(
            "Simulation JSON not found, running simulation (threshold={})...",
            threshold
        );
        let iter = parse_histogram::open(&args.file)?;
        let filtered = FilterIter::new(iter, args.layer, args.batch);
        let result = run_simulation(filtered, threshold, PimConfig::default());
        let json = serde_json::to_string_pretty(&result)
            .context("failed to serialize simulation result to JSON")?;
        fs::write(&sim_path, &json)
            .with_context(|| format!("failed to write {}", sim_path.display()))?;
        tracing::info!(
            "Simulation done — {} records, saved to {}",
            result.total_records,
            sim_path.display()
        );
        result
    };

    tracing::info!("Computing cycle counts...");
    let cycles = compute_cycles(&pim_result);

    if let Some(out_path) = output {
        let json = serde_json::to_string_pretty(&cycles)
            .context("failed to serialize cycle result to JSON")?;
        fs::write(out_path, json)
            .with_context(|| format!("failed to write {}", out_path.display()))?;
        tracing::info!("Cycle result saved to {}", out_path.display());
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&cycles)
                .context("failed to serialize cycle result to JSON")?
        );
    }

    Ok(())
}
