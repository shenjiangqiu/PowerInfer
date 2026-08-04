use std::path::PathBuf;

use clap::Parser;

// Kept crate-agnostic (no dependency on ratatui/crossterm/parse_histogram):
// build.rs pulls this file in directly as a module to generate shell
// completions, the same way it does for the main CLI's `cli.rs`.
#[derive(Parser)]
#[command(
    name = "run_all",
    about = "Run the calibration + report + layout-report pipeline over multiple model bin dumps in parallel, with a live TUI."
)]
pub struct Args {
    /// Bin dump directories to process, e.g. dumpbins_ReluLLaMA-7B
    /// dumpbins_Bamboo-7B ... Each entry may append `:THRESHOLD` to
    /// override `--threshold` for that one model (e.g.
    /// `dumpbins_ProSparse-llama-7b:0.5`) — needed because different
    /// models can have different `powerinfer.sparse_threshold` values
    /// baked into their GGUF (0.0 for plain ReLU models; techniques like
    /// ProSparse train with a nonzero threshold specifically to push
    /// sparsity higher than vanilla ReLU gives).
    pub dirs: Vec<String>,
    /// Output directory; one subdirectory per model is created underneath
    #[arg(long, default_value = "results")]
    pub results: PathBuf,
    /// Default activation threshold for any directory that doesn't specify
    /// its own via `dir:THRESHOLD` — check the model's load-time log line
    /// `sparse_pred_threshold = ...` (or its GGUF/config.json) if unsure
    #[arg(short = 't', long, default_value = "0.0")]
    pub threshold: f32,
    /// Comma-separated span values to test in the layout report
    #[arg(long, default_value = "1,2,4,8")]
    pub spans: String,
    /// Soft storage-balance cap for the greedy packer, as a multiple of the mean neurons/bank
    #[arg(long, default_value = "1.2")]
    pub capacity_slack: f64,
    /// Max models processed concurrently (default: all of them at once)
    #[arg(long)]
    pub jobs: Option<usize>,
    /// Weight element width in bytes: 4 = f32 (default), 2 = f16, 1 = int8.
    /// Everything downstream derives from it: row-group capacity
    /// (1024/width coefficients), chunk size (16/width neurons), physical
    /// rows per neuron (4096*width/1024), and GPU roofline byte counts.
    #[arg(long, default_value = "4")]
    pub data_width: u64,
    /// Weight element width in BITS — overrides --data-width when set and
    /// allows sub-byte precisions: 32/16/8/6/4. int6 packs 21 neurons per
    /// 16-byte chunk (padded), int4 packs 32.
    #[arg(long)]
    pub weight_bits: Option<u64>,
}
