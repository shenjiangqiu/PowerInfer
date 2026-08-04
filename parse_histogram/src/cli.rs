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
        /// Activation threshold — must match the model's own
        /// `powerinfer.sparse_threshold` (0.0 for plain ReLU models; some
        /// techniques like ProSparse use a nonzero threshold to push
        /// sparsity higher than vanilla ReLU, and passing 0.0 for those
        /// will overcount active neurons)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
    },
    /// Print info of the first N records
    Print {
        /// Number of records to print
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        /// Activation threshold (see `histogram --threshold`)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
    },
    /// Compute sparsity statistics (overall and per-layer)
    Sparsity {
        /// Activation threshold (see `histogram --threshold`)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
    },
    /// Run PIM simulation with given activation threshold
    Simulate {
        /// Activation threshold (default: 0.0)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
        /// Save result as JSON to this file (auto-derived if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Path to remap JSON for balanced bank placement
        #[arg(short = 'r', long)]
        remap: Option<PathBuf>,
        /// Union this many consecutive same-layer records into one
        /// dispatched round before computing the busiest bank (models N
        /// tokens/requests batched together). 1 = no batching.
        #[arg(long, default_value = "1")]
        batch_group: usize,
    },
    /// Convert simulation stats to cycle counts (auto-runs simulation if needed)
    ToCycle {
        /// Activation threshold for simulation (default: 0.0)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
        /// Save cycle result as JSON (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Path to remap JSON for balanced bank placement
        #[arg(short = 'r', long)]
        remap: Option<PathBuf>,
    },

    /// Generate a static, calibration-based neuron -> bank remap table
    /// from the same kind of activation-trace bins used everywhere else.
    GenMapping {
        /// Save the remap table (RemapTable JSON) to this file
        #[arg(short, long)]
        output: PathBuf,
        /// Activation threshold used to compute calibration probabilities
        /// (see `histogram --threshold`)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
        /// Packing strategy: modulo | round-robin | greedy (default: greedy)
        #[arg(short = 's', long, default_value = "greedy")]
        strategy: MappingStrategyArg,
        /// Total banks in the UP weight array
        #[arg(long, default_value = "1024")]
        up_banks: usize,
        /// Banks per placement unit for UP (must divide up_banks)
        #[arg(long, default_value = "1")]
        up_span: usize,
        /// Total banks in the DOWN weight array
        #[arg(long, default_value = "1024")]
        down_banks: usize,
        /// Banks per placement unit for DOWN (must divide down_banks)
        #[arg(long, default_value = "1")]
        down_span: usize,
        /// Soft storage-balance cap, as a multiple of the mean neurons/bank
        #[arg(long, default_value = "1.2")]
        capacity_slack: f64,
        /// Print per-layer mapping-quality stats (storage/load balance) to stderr
        #[arg(long, default_value_t = false)]
        report: bool,
    },

    /// Compare naive (i % banks) vs. calibrated-remap vs. ideal-balanced
    /// timing, to quantify how much the imbalance costs and how much of
    /// it the remap recovers. Runs the simulation twice (with and without
    /// `--remap`) over the same bins.
    Report {
        /// Activation threshold for simulation (default: 0.0)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
        /// Path to remap JSON to evaluate against the naive baseline
        #[arg(short = 'r', long)]
        remap: PathBuf,
        /// Union this many consecutive same-layer records into one
        /// dispatched round before computing the busiest bank (models N
        /// tokens/requests batched together). 1 = no batching.
        #[arg(long, default_value = "1")]
        batch_group: usize,
    },

    /// Compare naive vs. non-distributed-async vs. distributed-no-stagger
    /// vs. distributed-staggered-async layouts for UP and DOWN, using
    /// calibration data from `--calib` and measuring on `--file`. See
    /// `pim::layout` docs for the physical model this implements.
    LayoutReport {
        /// Calibration bins (dir or file) used to build the UP remap and
        /// the DOWN clustering permutation
        #[arg(long)]
        calib: PathBuf,
        /// Activation threshold (default: 0.0)
        #[arg(short = 't', long, default_value = "0.0")]
        threshold: f32,
        /// Banks to stripe one UP neuron's row across in the "distributed"
        /// configs (must divide up_rows_per_bank, e.g. 16 for the default config)
        #[arg(long, default_value = "4")]
        up_span: usize,
        /// Banks to stripe one DOWN tensor's row-groups across in the
        /// "distributed" configs
        #[arg(long, default_value = "4")]
        down_span: usize,
        /// Soft storage-balance cap for the UP remap, as a multiple of the
        /// mean neurons/bank
        #[arg(long, default_value = "1.2")]
        capacity_slack: f64,
    },

    ParseJson{

    }
}

// Kept crate-agnostic (no dependency on the `parse_histogram` lib crate):
// build.rs pulls this file in directly as a module to generate shell
// completions, without the lib crate available. The conversion to
// `parse_histogram::MappingStrategy` lives in main.rs instead.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum MappingStrategyArg {
    Modulo,
    RoundRobin,
    Greedy,
}
