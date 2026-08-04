//! Run the full calibration + report + layout-report pipeline over several
//! model bin-dump directories in parallel, with a live ratatui progress UI.
//!
//! Unlike the main `parse_histogram` CLI's `report`/`layout-report`
//! commands, this tool does **not** split each directory into a
//! calibration half and a held-out evaluation half: the whole directory is
//! used both to build the calibration statistics and to measure the
//! result, since the point here is a full production-settings sweep across
//! several models rather than a generalization test.
//!
//! Usage:
//!   run_all <dir1> <dir2> ... [--results DIR] [--spans 1,2,4,8]
//!           [--capacity-slack 1.2] [--jobs N]

#[path = "../run_all_cli.rs"]
mod cli;

use std::collections::HashMap;
use std::fs;
use std::io::stdout;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use cli::Args;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use ratatui::{Frame, Terminal};
use rayon::prelude::*;

use parse_histogram::{
    build_balance_report, build_remap_table, cluster_slot_of, compute_activation_stats,
    generate_mapping, list_bin_files, merge_activation_tables, open_one, run_layout_comparison,
    LayerActivationTable, LayoutCalibration, LayoutComparisonResult, MappingParams,
    MappingStrategy, PimConfig, PimContext, PimResult, Record, SpanLayout,
};

// ── Progress tracking ────────────────────────────────────────────

#[derive(Clone)]
struct Progress {
    stage: String,
    /// Shared across every rayon worker processing the current stage's
    /// files concurrently — each worker adds its own delta via `fetch_add`,
    /// so this is a running sum rather than one stream's absolute count.
    bytes_done: Arc<AtomicU64>,
    total_bytes: u64,
    done: bool,
    error: Option<String>,
    started: Instant,
    stage_started: Instant,
    log: Vec<String>,
}

impl Progress {
    fn new(total_bytes: u64) -> Self {
        let now = Instant::now();
        Self {
            stage: "queued".to_string(),
            bytes_done: Arc::new(AtomicU64::new(0)),
            total_bytes,
            done: false,
            error: None,
            started: now,
            stage_started: now,
            log: Vec::new(),
        }
    }
}

type SharedProgress = Arc<Mutex<Progress>>;

fn set_stage(p: &SharedProgress, stage: &str) -> Arc<AtomicU64> {
    let mut g = p.lock().unwrap();
    if g.stage != "queued" {
        let elapsed = g.stage_started.elapsed().as_secs_f64();
        let prev = g.stage.clone();
        g.log.push(format!("{} done in {:.1}s", prev, elapsed));
        if g.log.len() > 8 {
            g.log.remove(0);
        }
    }
    g.stage = stage.to_string();
    g.bytes_done.store(0, Ordering::Relaxed);
    g.stage_started = Instant::now();
    g.bytes_done.clone()
}

fn mark_done(p: &SharedProgress) {
    let mut g = p.lock().unwrap();
    let elapsed = g.stage_started.elapsed().as_secs_f64();
    let prev = g.stage.clone();
    g.log.push(format!("{} done in {:.1}s", prev, elapsed));
    g.stage = "done".to_string();
    g.done = true;
}

fn mark_failed(p: &SharedProgress, err: String) {
    let mut g = p.lock().unwrap();
    g.error = Some(err);
    g.done = true;
}

/// Wraps a record iterator to periodically report bytes consumed into a
/// shared atomic counter, and to stop early (returning no more records)
/// once cancellation is requested. Multiple `ProgressIter`s — one per file
/// — run concurrently on separate rayon workers during a stage, each
/// contributing its own delta via `fetch_add` rather than overwriting a
/// single absolute count, since every file only knows its own progress.
struct ProgressIter<I> {
    inner: I,
    bytes_done: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    since_update: u64,
}

impl<I> ProgressIter<I> {
    fn new(inner: I, bytes_done: Arc<AtomicU64>, cancel: Arc<AtomicBool>) -> Self {
        Self { inner, bytes_done, cancel, since_update: 0 }
    }

    fn flush(&mut self) {
        if self.since_update > 0 {
            self.bytes_done.fetch_add(self.since_update, Ordering::Relaxed);
            self.since_update = 0;
        }
    }
}

impl<I: Iterator<Item = Result<Record>>> Iterator for ProgressIter<I> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cancel.load(Ordering::Relaxed) {
            self.flush();
            return None;
        }
        match self.inner.next() {
            Some(item) => {
                if let Ok(r) = &item {
                    self.since_update += 16 + (r.scores.len() as u64) * 4;
                }
                if self.since_update >= 8_000_000 {
                    self.flush();
                }
                Some(item)
            }
            None => {
                self.flush();
                None
            }
        }
    }
}

fn dir_total_bytes(dir: &Path) -> Result<u64> {
    let mut total = 0u64;
    if dir.is_dir() {
        for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "bin") {
                total += fs::metadata(&p)?.len();
            }
        }
    } else {
        total = fs::metadata(dir)?.len();
    }
    Ok(total)
}

// ── Tiny counting semaphore for --jobs throttling ───────────────

struct Sem {
    count: Mutex<usize>,
    cv: Condvar,
}

impl Sem {
    fn new(n: usize) -> Self {
        Self { count: Mutex::new(n), cv: Condvar::new() }
    }

    fn acquire(&self) {
        let mut guard = self.count.lock().unwrap();
        while *guard == 0 {
            guard = self.cv.wait(guard).unwrap();
        }
        *guard -= 1;
    }

    fn release(&self) {
        let mut guard = self.count.lock().unwrap();
        *guard += 1;
        self.cv.notify_one();
    }
}

struct SemGuard<'a>(&'a Sem);

impl Drop for SemGuard<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

// ── Per-model pipeline ───────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_model(
    name: &str,
    dir: &Path,
    results_root: &Path,
    progress: SharedProgress,
    cancel: Arc<AtomicBool>,
    sem: Arc<Sem>,
    spans: &[usize],
    capacity_slack: f64,
    threshold: f32,
    weight_bits: u64,
) -> Result<()> {
    sem.acquire();
    let _guard = SemGuard(&sem);

    if cancel.load(Ordering::Relaxed) {
        mark_failed(&progress, "cancelled before starting".to_string());
        return Ok(());
    }

    let model_dir = results_root.join(name);
    fs::create_dir_all(&model_dir)?;
    let config = PimConfig {
        weight_bits,
        data_width: (weight_bits / 8).max(1),
        ..PimConfig::default()
    };

    // Stage 1: activation stats over the *entire* directory (no calib/eval
    // split — this run is meant to score every model as a whole).
    // `threshold` must match this model's own `powerinfer.sparse_threshold`
    // (0.0 for plain ReLU models; see Record docs) or every count below
    // silently overcounts "active" neurons and undercounts sparsity.
    let bytes_done = set_stage(&progress, &format!("activation stats (threshold={})", threshold));
    let files = list_bin_files(dir)?;
    let stats: LayerActivationTable = files
        .par_iter()
        .map(|p| -> Result<LayerActivationTable> {
            let iter = open_one(p)?;
            let iter = ProgressIter::new(iter, bytes_done.clone(), cancel.clone());
            Ok(compute_activation_stats(iter, threshold))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(merge_activation_tables)
        .unwrap_or_default();
    if cancel.load(Ordering::Relaxed) {
        mark_failed(&progress, "cancelled".to_string());
        return Ok(());
    }

    let mut total_neurons: u64 = 0;
    let mut total_activated: u64 = 0;
    for s in stats.values() {
        total_neurons += s.total_records * s.activated_counts.len() as u64;
        total_activated += s.activated_counts.iter().sum::<u64>();
    }
    let sparsity = if total_neurons > 0 {
        1.0 - (total_activated as f64 / total_neurons as f64)
    } else {
        0.0
    };
    fs::write(
        model_dir.join("sparsity.txt"),
        format!(
            "layers={}\nthreshold={}\nsparsity={:.4}\nactivated={}\ntotal={}\n",
            stats.len(),
            threshold,
            sparsity,
            total_activated,
            total_neurons
        ),
    )?;

    // Stage 2: build remap tables (span=1 for the report, one per
    // requested span>1 for the layout report) — in-memory, no I/O.
    set_stage(&progress, "building remaps");
    let up_params_1 = MappingParams { n_banks: config.banks as usize, span: 1, capacity_slack };
    let down_params_1 = MappingParams { n_banks: config.banks as usize, span: 1, capacity_slack };
    let remap_1 = build_remap_table(&stats, &up_params_1, &down_params_1, MappingStrategy::GreedyLpt);
    remap_1.save(model_dir.join("remap_span1.json"))?;

    let mut layers: Vec<i32> = stats.keys().copied().collect();
    layers.sort_unstable();

    let mut calib_span1 = LayoutCalibration::default();
    let mut calib_cluster = LayoutCalibration::default();
    let mut calib_by_span: HashMap<usize, LayoutCalibration> = HashMap::new();
    for &span in spans {
        if span > 1 {
            calib_by_span.insert(span, LayoutCalibration::default());
        }
    }
    let mut calib_random = LayoutCalibration::default();
    for &layer in &layers {
        let probs = stats[&layer].probabilities();
        calib_span1
            .up_remap
            .insert(layer, generate_mapping(&probs, &up_params_1, MappingStrategy::GreedyLpt));
        calib_cluster.down_slot_of.insert(layer, cluster_slot_of(&probs));
        // Random (uncalibrated) placement baseline: deterministic
        // Fisher-Yates shuffle of neuron ids (xorshift64, layer-seeded),
        // dealt round-robin to banks — storage-balanced but blind to
        // firing probability. Isolates what calibration itself buys.
        let n = probs.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut s: u64 = 0x9e3779b97f4a7c15 ^ (layer as u64).wrapping_mul(0xff51afd7ed558ccd);
        for i in (1..n).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            perm.swap(i, (s % (i as u64 + 1)) as usize);
        }
        let mut bank_of = vec![0usize; n];
        for (pos, &neuron) in perm.iter().enumerate() {
            bank_of[neuron] = pos % config.banks as usize;
        }
        calib_random.up_remap.insert(layer, bank_of);
        for &span in spans {
            if span <= 1 {
                continue;
            }
            let up_params_n = MappingParams { n_banks: config.banks as usize, span, capacity_slack };
            let mapped = generate_mapping(&probs, &up_params_n, MappingStrategy::GreedyLpt);
            calib_by_span.get_mut(&span).unwrap().up_remap.insert(layer, mapped);
        }
    }
    if cancel.load(Ordering::Relaxed) {
        mark_failed(&progress, "cancelled".to_string());
        return Ok(());
    }

    // Stage 3: naive vs. calibrated-remap report, both computed in one pass
    // per file (two PimContext accumulators fed the same record stream),
    // with each file on its own rayon worker and the per-file results
    // merged (map-reduce — every file is an independent record stream).
    let bytes_done = set_stage(&progress, "naive+remap report");
    let (naive_result, remap_result): (Vec<PimResult>, Vec<PimResult>) = files
        .par_iter()
        .map(|p| -> Result<(PimResult, PimResult)> {
            let iter = open_one(p)?;
            let iter = ProgressIter::new(iter, bytes_done.clone(), cancel.clone());
            let mut ctx_naive = PimContext::new(config.clone(), None);
            let mut ctx_remap = PimContext::new(config.clone(), Some(remap_1.clone()));
            for record in iter {
                let record = match record {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[{}] warning: {}", name, e);
                        continue;
                    }
                };
                let indices: Vec<usize> = record
                    .scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s > threshold)
                    .map(|(i, _)| i)
                    .collect();
                ctx_naive.compute_time(record.layer, record.scores.len(), &indices);
                ctx_remap.compute_time(record.layer, record.scores.len(), &indices);
            }
            Ok((ctx_naive.into_result(), ctx_remap.into_result()))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    if cancel.load(Ordering::Relaxed) {
        mark_failed(&progress, "cancelled".to_string());
        return Ok(());
    }
    let naive_result = naive_result.into_iter().reduce(PimResult::merge).context("no .bin files processed")?;
    let remap_result = remap_result.into_iter().reduce(PimResult::merge).context("no .bin files processed")?;
    let comparison = build_balance_report(&naive_result, &remap_result);

    fs::write(
        model_dir.join("report.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "naive": naive_result,
            "remap": remap_result,
            "comparison": comparison,
        }))?,
    )?;
    let mut report_txt = format!("records={}\n", naive_result.total_records);
    for r in &comparison {
        report_txt.push_str(&format!(
            "{}: naive={} remap={} ideal={} speedup={:.3}x remap/ideal={:.3}x\n",
            r.name, r.naive_rows, r.remap_rows, r.ideal_rows, r.speedup_vs_naive, r.remap_vs_ideal
        ));
    }
    fs::write(model_dir.join("report.txt"), report_txt)?;

    // Stage 4: layout report — async-only / random-remap attribution
    // baselines, non-distributed calibrated, down-clustered, and every
    // requested span (no-stagger / staggered / staggered+clustered), all
    // computed together in one pass per file, files processed in parallel.
    let mut names: Vec<String> = vec![
        "async only (modulo, no remap)".into(),
        "async + random remap".into(),
        "non-distributed async".into(),
        "non-distributed + down-clustered".into(),
    ];
    let nd = SpanLayout::non_distributed();
    let mut up_layouts: Vec<SpanLayout> = vec![nd, nd, nd, nd];
    let mut down_layouts: Vec<SpanLayout> = vec![nd, nd, nd, nd];
    let mut up_cals: Vec<Option<&LayoutCalibration>> =
        vec![None, Some(&calib_random), Some(&calib_span1), Some(&calib_span1)];
    let mut down_cals: Vec<Option<&LayoutCalibration>> = vec![None, None, None, Some(&calib_cluster)];

    for &span in spans {
        if span <= 1 {
            continue;
        }
        let cal = calib_by_span.get(&span);

        names.push(format!("distributed span={} no-stagger", span));
        up_layouts.push(SpanLayout { span, stagger: false });
        down_layouts.push(SpanLayout { span, stagger: false });
        up_cals.push(cal);
        down_cals.push(None);

        names.push(format!("distributed span={} staggered", span));
        up_layouts.push(SpanLayout { span, stagger: true });
        down_layouts.push(SpanLayout { span, stagger: true });
        up_cals.push(cal);
        down_cals.push(None);

        names.push(format!("distributed span={} staggered+clustered", span));
        up_layouts.push(SpanLayout { span, stagger: true });
        down_layouts.push(SpanLayout { span, stagger: true });
        up_cals.push(cal);
        down_cals.push(Some(&calib_cluster));
    }

    let configs: Vec<(&str, SpanLayout, Option<&LayoutCalibration>, SpanLayout, Option<&LayoutCalibration>)> = (0..names.len())
        .map(|i| (names[i].as_str(), up_layouts[i], up_cals[i], down_layouts[i], down_cals[i]))
        .collect();

    let bytes_done = set_stage(&progress, "layout report");
    let layout_result: LayoutComparisonResult = files
        .par_iter()
        .map(|p| -> Result<LayoutComparisonResult> {
            let iter = open_one(p)?;
            let iter = ProgressIter::new(iter, bytes_done.clone(), cancel.clone());
            run_layout_comparison(iter, threshold, &config, &configs)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(LayoutComparisonResult::merge)
        .context("no .bin files processed")?;
    if cancel.load(Ordering::Relaxed) {
        mark_failed(&progress, "cancelled".to_string());
        return Ok(());
    }

    fs::write(model_dir.join("layout.json"), serde_json::to_string_pretty(&layout_result)?)?;
    let n_neurons = stats.values().map(|s| s.activated_counts.len()).max().unwrap_or(0);
    let mut layout_txt = format!(
        "records={} n={} weight_bits={} naive_up_ns={} naive_down_ns={} (naive_up_rows={} naive_down_rows={})\n",
        layout_result.records,
        n_neurons,
        weight_bits,
        layout_result.naive_up_time_ns,
        layout_result.naive_down_time_ns,
        layout_result.naive_up_rows,
        layout_result.naive_down_rows
    );
    layout_txt.push_str(&format!(
        "oracle: up_ns={} (up_rows={}) down_ns={} (down_rows={} down_chunks={})\n",
        layout_result.oracle_up_time_ns,
        layout_result.oracle_up_rows,
        layout_result.oracle_down_time_ns,
        layout_result.oracle_down_rows,
        layout_result.oracle_down_chunks,
    ));
    // Per-token latency percentiles for the best config (min up+down total).
    if let Some(best_idx) = (0..layout_result.configs.len()).min_by_key(|&i| {
        // best = independently-best up and down aren't necessarily one
        // config; for percentiles use the config with the lowest combined
        // total, which matches how a single deployed layout would behave.
        layout_result.configs[i].up_time_ns + layout_result.configs[i].down_time_ns
    }) {
        let mut lat = layout_result.per_token_latency[best_idx].clone();
        if !lat.is_empty() {
            lat.sort_unstable();
            let pct = |p: f64| lat[(((lat.len() - 1) as f64) * p) as usize];
            layout_txt.push_str(&format!(
                "per-token latency (config '{}', {} tokens): P50={}ns P95={}ns P99={}ns mean={}ns\n",
                layout_result.configs[best_idx].name,
                lat.len(),
                pct(0.50),
                pct(0.95),
                pct(0.99),
                lat.iter().sum::<u64>() / lat.len() as u64,
            ));
        }
    }
    let pim_naive_total = layout_result.naive_up_time_ns + layout_result.naive_down_time_ns;
    let pim_best_total = layout_result
        .configs
        .iter()
        .map(|c| c.up_time_ns)
        .min()
        .unwrap_or(0)
        + layout_result.configs.iter().map(|c| c.down_time_ns).min().unwrap_or(0);
    layout_txt.push_str(&format!(
        "gpu@{}GB/s: dense_ns={} sparse_ideal_ns={} | pim_naive_ns={} pim_best_ns={} | pim_best vs gpu_dense={:.3}x vs gpu_sparse={:.3}x\n",
        config.gpu_bandwidth_gbps,
        layout_result.gpu_dense_time_ns,
        layout_result.gpu_sparse_time_ns,
        pim_naive_total,
        pim_best_total,
        layout_result.gpu_dense_time_ns as f64 / pim_best_total.max(1) as f64,
        layout_result.gpu_sparse_time_ns as f64 / pim_best_total.max(1) as f64,
    ));
    for c in &layout_result.configs {
        let up_speedup = if c.up_time_ns > 0 {
            layout_result.naive_up_time_ns as f64 / c.up_time_ns as f64
        } else {
            1.0
        };
        let down_speedup = if c.down_time_ns > 0 {
            layout_result.naive_down_time_ns as f64 / c.down_time_ns as f64
        } else {
            1.0
        };
        let subrow_gain = if c.down_time_ns > 0 {
            c.down_time_ns_full_row as f64 / c.down_time_ns as f64
        } else {
            1.0
        };
        layout_txt.push_str(&format!(
            "{}: up_ns={} (speedup={:.3}x) down_ns_skip={} down_ns_full={} (speedup={:.3}x, subrow_gain={:.3}x) [up_rows={} down_rows={} down_chunks_skip={} down_chunks_full={}]\n",
            c.name,
            c.up_time_ns,
            up_speedup,
            c.down_time_ns,
            c.down_time_ns_full_row,
            down_speedup,
            subrow_gain,
            c.up_rows,
            c.down_rows,
            c.down_chunk_compute,
            c.down_chunk_compute_full_row,
        ));
    }
    fs::write(model_dir.join("layout.txt"), layout_txt)?;

    mark_done(&progress);
    Ok(())
}

// ── TUI ──────────────────────────────────────────────────────────

struct Job {
    name: String,
    progress: SharedProgress,
}

fn draw_ui(f: &mut Frame, jobs: &[Job], start_all: Instant, results_dir: &Path) {
    let area = f.area();
    let mut constraints: Vec<Constraint> = jobs.iter().map(|_| Constraint::Length(5)).collect();
    constraints.push(Constraint::Min(1));
    constraints.push(Constraint::Length(1));
    let chunks = Layout::default().direction(Direction::Vertical).constraints(constraints).split(area);

    for (job, chunk) in jobs.iter().zip(chunks.iter()) {
        draw_job(f, job, *chunk);
    }

    let footer = Paragraph::new(Line::from(vec![
        Span::raw(format!("elapsed {:.0}s — results in {}  ", start_all.elapsed().as_secs_f64(), results_dir.display())),
        Span::styled("[q] quit", Style::default().fg(Color::DarkGray)),
    ]));
    f.render_widget(footer, chunks[chunks.len() - 1]);
}

fn draw_job(f: &mut Frame, job: &Job, area: Rect) {
    let g = job.progress.lock().unwrap();
    let bytes_done = g.bytes_done.load(Ordering::Relaxed);
    let ratio = if g.total_bytes > 0 {
        (bytes_done as f64 / g.total_bytes as f64).min(1.0)
    } else {
        0.0
    };
    let color = if g.error.is_some() {
        Color::Red
    } else if g.done {
        Color::Green
    } else {
        Color::Cyan
    };
    let label = format!(
        "{}  ({:.2}/{:.2} GB)",
        g.error.as_deref().unwrap_or(&g.stage),
        bytes_done as f64 / 1e9,
        g.total_bytes as f64 / 1e9
    );

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!(" {} ", job.name)))
        .gauge_style(Style::default().fg(color))
        .ratio(ratio)
        .label(label);
    f.render_widget(gauge, rows[0]);

    let log_line = g.log.last().cloned().unwrap_or_default();
    let log = Paragraph::new(Line::from(Span::styled(log_line, Style::default().fg(Color::DarkGray))));
    f.render_widget(log, rows[1]);
}

/// Parse one `--dirs` entry: `path` or `path:threshold`. A bare trailing
/// `:N` is only treated as a threshold override if it parses as a float;
/// this keeps plain paths containing `:` (rare, but e.g. Windows drive
/// letters) working by falling back to the whole string as the path.
fn parse_dir_arg(raw: &str, default_threshold: f32) -> (PathBuf, f32) {
    if let Some((path, thr)) = raw.rsplit_once(':') {
        if let Ok(t) = thr.parse::<f32>() {
            return (PathBuf::from(path), t);
        }
    }
    (PathBuf::from(raw), default_threshold)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.dirs.is_empty() {
        eprintln!("usage: run_all <dir1[:threshold]> <dir2[:threshold]> ... [--threshold 0.0] [--results DIR] [--spans 1,2,4,8] [--jobs N]");
        std::process::exit(1);
    }
    let spans: Vec<usize> = args
        .spans
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .context("invalid --spans (expected comma-separated integers)")?;

    fs::create_dir_all(&args.results)?;

    let mut jobs: Vec<Job> = Vec::new();
    let mut job_dirs: Vec<PathBuf> = Vec::new();
    let mut job_thresholds: Vec<f32> = Vec::new();
    for raw in &args.dirs {
        let (dir, threshold) = parse_dir_arg(raw, args.threshold);
        if !dir.exists() {
            eprintln!("warning: {} does not exist, skipping", dir.display());
            continue;
        }
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| dir.display().to_string());
        let total_bytes = dir_total_bytes(&dir)?;
        jobs.push(Job { name, progress: Arc::new(Mutex::new(Progress::new(total_bytes))) });
        job_dirs.push(dir);
        job_thresholds.push(threshold);
    }
    if jobs.is_empty() {
        anyhow::bail!("no valid directories to process");
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let max_jobs = args.jobs.unwrap_or(jobs.len()).max(1);
    let sem = Arc::new(Sem::new(max_jobs));

    let handles: Vec<_> = jobs
        .iter()
        .zip(job_dirs.iter())
        .zip(job_thresholds.iter())
        .map(|((job, dir), &threshold)| {
            let name = job.name.clone();
            let dir = dir.clone();
            let results_root = args.results.clone();
            let progress = job.progress.clone();
            let cancel = cancel.clone();
            let sem = sem.clone();
            let spans = spans.clone();
            let capacity_slack = args.capacity_slack;
            let weight_bits = args.weight_bits.unwrap_or(args.data_width * 8);
            thread::spawn(move || {
                if let Err(e) = run_model(
                    &name,
                    &dir,
                    &results_root,
                    progress.clone(),
                    cancel,
                    sem,
                    &spans,
                    capacity_slack,
                    threshold,
                    weight_bits,
                ) {
                    mark_failed(&progress, e.to_string());
                }
            })
        })
        .collect();

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;
    let start_all = Instant::now();

    loop {
        terminal.draw(|f| draw_ui(f, &jobs, start_all, &args.results))?;

        if event::poll(Duration::from_millis(150))? {
            if let Event::Key(k) = event::read()? {
                if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        }

        if jobs.iter().all(|j| j.progress.lock().unwrap().done) {
            terminal.draw(|f| draw_ui(f, &jobs, start_all, &args.results))?;
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    for h in handles {
        let _ = h.join();
    }

    println!("Done in {:.0}s. Results in {}", start_all.elapsed().as_secs_f64(), args.results.display());
    for job in &jobs {
        let g = job.progress.lock().unwrap();
        if let Some(e) = &g.error {
            println!("  {}: ERROR/CANCELLED: {}", job.name, e);
        } else {
            println!("  {}: done in {:.1}s", job.name, g.started.elapsed().as_secs_f64());
        }
    }

    Ok(())
}
