use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "parse_histogram", about = "Parse binary neuron scores and build per-layer histograms of positive-score positions.")]
pub struct Args {
    /// Path to the binary file
    #[arg(short = 'f', long)]
    pub file: PathBuf,

    /// Filter by layer (only process records with this layer id)
    #[arg(short, long)]
    pub layer: Option<i32>,

    /// Filter by batch (only process records with this batch id)
    #[arg(short, long)]
    pub batch: Option<i32>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Compute per-layer histograms of positive-score positions
    Histogram {
        /// Save histogram as JSON to this file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Print info of the first N records
    Print {
        /// Number of records to print
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
    },
    /// Compute sparsity statistics (overall and per-layer)
    Sparsity,
}
