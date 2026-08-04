mod cli;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Args, Commands};
use parse_histogram::{
    self, build_balance_report, build_remap_table, cluster_slot_of, compute_activation_stats,
    compute_activation_stats_parallel, compute_cycles, compute_histograms, compute_sparsity,
    derive_json_path, derive_remap_json_path, generate_mapping, mapping_quality,
    print_balance_report, print_first_records, print_histograms, print_sparsity,
    run_layout_comparison, run_layout_comparison_parallel, run_naive_and_remap_parallel,
    run_simulation, run_simulation_batched, FilterIter, LayoutCalibration, MappingParams,
    MappingStrategy, PimConfig, PimResult, RemapTable, SpanLayout,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // to-cycle handles its own file I/O (may run simulation first)
    if let Commands::ToCycle {
        threshold,
        output,
        remap,
    } = &args.command
    {
        return cmd_to_cycle(&args, *threshold, output.as_ref(), remap.as_ref());
    }
    // gen-mapping and report each open the bins themselves (gen-mapping
    // needs the raw stream twice-removed through compute_activation_stats;
    // report runs the simulator twice), so handle them before the shared
    // FilterIter setup below.
    if let Commands::GenMapping {
        output,
        threshold,
        strategy,
        up_banks,
        up_span,
        down_banks,
        down_span,
        capacity_slack,
        report,
    } = &args.command
    {
        let strategy = match strategy {
            cli::MappingStrategyArg::Modulo => parse_histogram::MappingStrategy::Modulo,
            cli::MappingStrategyArg::RoundRobin => {
                parse_histogram::MappingStrategy::SortedRoundRobin
            }
            cli::MappingStrategyArg::Greedy => parse_histogram::MappingStrategy::GreedyLpt,
        };
        return cmd_gen_mapping(
            &args,
            output,
            *threshold,
            strategy,
            *up_banks,
            *up_span,
            *down_banks,
            *down_span,
            *capacity_slack,
            *report,
        );
    }
    if let Commands::Report {
        threshold,
        remap,
        batch_group,
    } = &args.command
    {
        return cmd_report(&args, *threshold, remap, *batch_group);
    }
    if let Commands::LayoutReport {
        calib,
        threshold,
        up_span,
        down_span,
        capacity_slack,
    } = &args.command
    {
        return cmd_layout_report(&args, calib, *threshold, *up_span, *down_span, *capacity_slack);
    }

    let iter = parse_histogram::open(&args.file)?;
    let mut filtered = FilterIter::new(iter, args.layer, args.batch);

    match args.command {
        Commands::Print { count, threshold } => {
            print_first_records(&mut filtered, count, threshold);
        }
        Commands::Histogram { output, threshold } => {
            let histograms = compute_histograms(filtered, threshold);

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
        Commands::Sparsity { threshold } => {
            let stats = compute_sparsity(filtered, threshold);
            print_sparsity(&stats);
        }
        Commands::Simulate {
            threshold,
            output,
            remap,
            batch_group,
        } => {
            let remap_table = if let Some(ref rp) = remap {
                Some(parse_histogram::RemapTable::load(rp)?)
            } else {
                None
            };
            tracing::info!(
                "Running PIM simulation (threshold={}, remap={}, batch_group={})...",
                threshold,
                remap.is_some(),
                batch_group
            );
            let result = run_simulation_batched(
                filtered,
                threshold,
                PimConfig::default(),
                remap_table,
                batch_group,
            );

            let out_path = output.unwrap_or_else(|| {
                if remap.is_some() {
                    derive_remap_json_path(&args.file)
                } else {
                    derive_json_path(&args.file)
                }
            });
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
        Commands::GenMapping { .. } => unreachable!(),
        Commands::Report { .. } => unreachable!(),
        Commands::LayoutReport { .. } => unreachable!(),
        Commands::ParseJson {} => {}
    }

    Ok(())
}

fn cmd_to_cycle(
    args: &Args,
    threshold: f32,
    output: Option<&PathBuf>,
    remap: Option<&PathBuf>,
) -> Result<()> {
    let remap_table = if let Some(rp) = remap {
        Some(parse_histogram::RemapTable::load(rp)?)
    } else {
        None
    };
    let sim_path = if remap.is_some() {
        derive_remap_json_path(&args.file)
    } else {
        derive_json_path(&args.file)
    };

    let pim_result: PimResult = if sim_path.exists() {
        tracing::info!("Using cached simulation JSON: {}", sim_path.display());
        let json_str = fs::read_to_string(&sim_path)
            .with_context(|| format!("failed to read {}", sim_path.display()))?;
        serde_json::from_str(&json_str).context("failed to parse simulation JSON")?
    } else {
        tracing::info!(
            "Simulation JSON not found, running simulation (threshold={})...",
            threshold
        );
        let iter = parse_histogram::open(&args.file)?;
        let filtered = FilterIter::new(iter, args.layer, args.batch);
        let result = run_simulation(filtered, threshold, PimConfig::default(), remap_table);
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

#[allow(clippy::too_many_arguments)]
fn cmd_gen_mapping(
    args: &Args,
    output: &PathBuf,
    threshold: f32,
    strategy: parse_histogram::MappingStrategy,
    up_banks: usize,
    up_span: usize,
    down_banks: usize,
    down_span: usize,
    capacity_slack: f64,
    report: bool,
) -> Result<()> {
    tracing::info!(
        "Computing per-layer activation stats from calibration set (threshold={})...",
        threshold
    );
    // The parallel path processes each .bin file on its own rayon worker,
    // so it can't apply a layer/batch filter mid-stream — fall back to the
    // serial FilterIter path when one is requested.
    let stats = if args.layer.is_none() && args.batch.is_none() {
        compute_activation_stats_parallel(&args.file, threshold)?
    } else {
        let iter = parse_histogram::open(&args.file)?;
        let filtered = FilterIter::new(iter, args.layer, args.batch);
        compute_activation_stats(filtered, threshold)
    };
    tracing::info!("{} layers observed", stats.len());

    let up_params = MappingParams {
        n_banks: up_banks,
        span: up_span,
        capacity_slack,
    };
    let down_params = MappingParams {
        n_banks: down_banks,
        span: down_span,
        capacity_slack,
    };

    let table = build_remap_table(&stats, &up_params, &down_params, strategy);

    if report {
        let mut layers: Vec<_> = stats.keys().copied().collect();
        layers.sort_unstable();
        eprintln!(
            "{:<6}{:>10}{:>10}{:>10}{:>10}{:>14}{:>10}{:>10}{:>14}",
            "layer",
            "up_min",
            "up_max",
            "up_mean",
            "up_std",
            "up_imbal",
            "dn_min",
            "dn_max",
            "dn_imbal"
        );
        for layer in layers {
            let probs = stats[&layer].probabilities();
            let up_bank_of = &table.up_remap[&layer];
            let down_bank_of = &table.down_remap[&layer];
            let up_q = mapping_quality(&probs, up_bank_of, &up_params);
            let down_q = mapping_quality(&probs, down_bank_of, &down_params);
            eprintln!(
                "{:<6}{:>10}{:>10}{:>10.2}{:>10.2}{:>13.3}x{:>10.4}{:>10.4}{:>13.3}x",
                layer,
                up_q.neuron_count_min,
                up_q.neuron_count_max,
                up_q.neuron_count_mean,
                up_q.neuron_count_stddev,
                up_q.load_imbalance_ratio,
                down_q.load_min,
                down_q.load_max,
                down_q.load_imbalance_ratio,
            );
        }
    }

    table.save(output)?;
    tracing::info!("Remap table saved to {}", output.display());

    Ok(())
}

fn cmd_report(args: &Args, threshold: f32, remap_path: &PathBuf, batch_group: usize) -> Result<()> {
    let remap_table = RemapTable::load(remap_path)?;

    // Same fallback rule as cmd_gen_mapping: the parallel path can't apply
    // a layer/batch filter mid-stream, since each file is processed whole
    // on its own rayon worker.
    let (naive, remapped) = if args.layer.is_none() && args.batch.is_none() {
        tracing::info!(
            "Running naive + remap simulation in parallel across files (batch_group={})...",
            batch_group
        );
        run_naive_and_remap_parallel(&args.file, threshold, PimConfig::default(), remap_table, batch_group)?
    } else {
        tracing::info!("Running naive baseline simulation (no remap, batch_group={})...", batch_group);
        let naive_iter = parse_histogram::open(&args.file)?;
        let naive_filtered = FilterIter::new(naive_iter, args.layer, args.batch);
        let naive = run_simulation_batched(
            naive_filtered,
            threshold,
            PimConfig::default(),
            None,
            batch_group,
        );

        tracing::info!("Running simulation with calibrated remap...");
        let remap_iter = parse_histogram::open(&args.file)?;
        let remap_filtered = FilterIter::new(remap_iter, args.layer, args.batch);
        let remapped = run_simulation_batched(
            remap_filtered,
            threshold,
            PimConfig::default(),
            Some(remap_table),
            batch_group,
        );
        (naive, remapped)
    };

    let rows = build_balance_report(&naive, &remapped);
    println!(
        "Bank-imbalance report — busiest-bank row count per record stream ({} records, batch_group={})",
        naive.total_records, batch_group
    );
    print_balance_report(&rows);

    Ok(())
}

fn cmd_layout_report(
    args: &Args,
    calib_path: &PathBuf,
    threshold: f32,
    up_span: usize,
    down_span: usize,
    capacity_slack: f64,
) -> Result<()> {
    let config = PimConfig::default();

    tracing::info!(
        "Computing per-layer activation stats from calibration set (threshold={})...",
        threshold
    );
    let stats = if args.layer.is_none() && args.batch.is_none() {
        compute_activation_stats_parallel(calib_path, threshold)?
    } else {
        let calib_iter = parse_histogram::open(calib_path)?;
        let calib_filtered = FilterIter::new(calib_iter, args.layer, args.batch);
        compute_activation_stats(calib_filtered, threshold)
    };
    tracing::info!("{} layers observed", stats.len());

    let up_params_1 = MappingParams {
        n_banks: config.banks as usize,
        span: 1,
        capacity_slack,
    };
    let up_params_n = MappingParams {
        n_banks: config.banks as usize,
        span: up_span,
        capacity_slack,
    };

    let mut calib_span1 = LayoutCalibration::default();
    let mut calib_span_n = LayoutCalibration::default();
    let mut calib_cluster = LayoutCalibration::default();

    let mut layers: Vec<_> = stats.keys().copied().collect();
    layers.sort_unstable();
    for layer in layers {
        let probs = stats[&layer].probabilities();
        calib_span1
            .up_remap
            .insert(layer, generate_mapping(&probs, &up_params_1, MappingStrategy::GreedyLpt));
        calib_span_n
            .up_remap
            .insert(layer, generate_mapping(&probs, &up_params_n, MappingStrategy::GreedyLpt));
        calib_cluster.down_slot_of.insert(layer, cluster_slot_of(&probs));
    }

    let up_span1 = SpanLayout { span: 1, stagger: false };
    // `stagger` is included on the UP side purely to empirically confirm
    // it makes no difference (UP neurons are independent — see pim::layout
    // docs); it should give identical up_rows to the no-stagger variant.
    let up_span_n_nostagger = SpanLayout { span: up_span, stagger: false };
    let up_span_n_stagger = SpanLayout { span: up_span, stagger: true };

    let down_span1 = SpanLayout { span: 1, stagger: false };
    let down_span_n_nostagger = SpanLayout { span: down_span, stagger: false };
    let down_span_n_stagger = SpanLayout { span: down_span, stagger: true };

    let configs = vec![
        ("non-distributed async", up_span1, Some(&calib_span1), down_span1, None),
        (
            "distributed, no stagger",
            up_span_n_nostagger,
            Some(&calib_span_n),
            down_span_n_nostagger,
            None,
        ),
        (
            "distributed + staggered",
            up_span_n_stagger,
            Some(&calib_span_n),
            down_span_n_stagger,
            None,
        ),
        (
            "non-distributed + down-clustered",
            up_span1,
            Some(&calib_span1),
            down_span1,
            Some(&calib_cluster),
        ),
        (
            "distributed+staggered+clustered",
            up_span_n_stagger,
            Some(&calib_span_n),
            down_span_n_stagger,
            Some(&calib_cluster),
        ),
    ];

    tracing::info!("Running layout comparison on {}...", args.file.display());
    let result = if args.layer.is_none() && args.batch.is_none() {
        run_layout_comparison_parallel(&args.file, threshold, &config, &configs)?
    } else {
        let iter = parse_histogram::open(&args.file)?;
        let filtered = FilterIter::new(iter, args.layer, args.batch);
        run_layout_comparison(filtered, threshold, &config, &configs)?
    };

    println!(
        "Layout report — busiest-unit time (ns) per record stream ({} records, up_span={}, down_span={})",
        result.records, up_span, down_span
    );
    let pim_best_total = result.configs.iter().map(|c| c.up_time_ns).min().unwrap_or(0)
        + result.configs.iter().map(|c| c.down_time_ns).min().unwrap_or(0);
    println!(
        "GPU reference @ {} GB/s: dense={} ns, sparse-ideal={} ns; best PIM (up+down) = {} ns → {:.3}x vs GPU dense, {:.3}x vs GPU sparse-ideal",
        config.gpu_bandwidth_gbps,
        result.gpu_dense_time_ns,
        result.gpu_sparse_time_ns,
        pim_best_total,
        result.gpu_dense_time_ns as f64 / pim_best_total.max(1) as f64,
        result.gpu_sparse_time_ns as f64 / pim_best_total.max(1) as f64,
    );
    println!(
        "{:<36}{:>16}{:>10}{:>16}{:>16}{:>10}{:>12}",
        "config", "up_ns", "up_spdup", "down_ns(skip)", "down_ns(full)", "dn_spdup", "subrow_gain"
    );
    println!(
        "{:<36}{:>16}{:>10}{:>16}{:>16}{:>10}{:>12}",
        "naive", result.naive_up_time_ns, "1.000x", result.naive_down_time_ns, result.naive_down_time_ns, "1.000x", "1.000x"
    );
    for c in &result.configs {
        let up_speedup = if c.up_time_ns > 0 {
            result.naive_up_time_ns as f64 / c.up_time_ns as f64
        } else {
            1.0
        };
        let down_speedup = if c.down_time_ns > 0 {
            result.naive_down_time_ns as f64 / c.down_time_ns as f64
        } else {
            1.0
        };
        let subrow_gain = if c.down_time_ns > 0 {
            c.down_time_ns_full_row as f64 / c.down_time_ns as f64
        } else {
            1.0
        };
        println!(
            "{:<36}{:>16}{:>9.3}x{:>16}{:>16}{:>9.3}x{:>11.3}x",
            c.name, c.up_time_ns, up_speedup, c.down_time_ns, c.down_time_ns_full_row, down_speedup, subrow_gain
        );
    }

    Ok(())
}
