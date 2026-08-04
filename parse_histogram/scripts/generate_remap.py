#!/usr/bin/env python3
"""
Generate constrained-swap remap JSON from histogram data.

The algorithm:
1. Starts with cyclic assignment (bank[i] = i % banks)
2. Iteratively swaps hot/cold neurons between max/min banks
3. Preserves per-bank neuron count constraint

Usage:
    python3 scripts/generate_remap.py <histogram.json> <output_remap.json>
"""

import json
import sys
import os

BANKS = 1024
CHANNELS = 32
BANKS_PER_CHANNEL = 32


def build_initial_assignment(data, banks):
    """Initial cyclic assignment: bank[i] = i % banks"""
    return [i % banks for i in range(len(data))]


def bank_totals_from_assignment(data, assignment, banks):
    totals = [0] * banks
    for i, count in enumerate(data):
        totals[assignment[i]] += count
    return totals


def constrained_swap_rebalance(data, assignment, banks, max_iters=20000):
    """
    Constrained swap rebalancing: swap neurons between banks to minimize max bank total.
    Preserves the per-bank neuron count (each bank keeps same number of neurons).
    """
    n = len(data)
    totals = bank_totals_from_assignment(data, assignment, banks)

    bank_neurons = [[] for _ in range(banks)]
    for i, bank in enumerate(assignment):
        bank_neurons[bank].append((i, data[i]))

    for b in range(banks):
        bank_neurons[b].sort(key=lambda x: x[1])

    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        max_bank = max(range(banks), key=lambda b: totals[b])
        min_bank = min(range(banks), key=lambda b: totals[b])
        max_total, min_total = totals[max_bank], totals[min_bank]

        if max_total - min_total <= 1:
            break

        # Find best swap: hottest from max_bank with coldest from min_bank
        best_reduction = 0
        best_pair = None
        for hi, hot_count in sorted(bank_neurons[max_bank], key=lambda x: -x[1]):
            for ci, cold_count in bank_neurons[min_bank]:
                if hot_count <= cold_count:
                    continue
                new_max = max_total - hot_count + cold_count
                new_min = min_total - cold_count + hot_count
                if new_max < max_total and new_min > min_total:
                    reduction = max_total - max(new_max, new_min)
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_pair = (hi, ci, hot_count, cold_count)

        if best_pair is not None:
            hi, ci, hot_count, cold_count = best_pair
            new_max = max_total - hot_count + cold_count
            new_min = min_total - cold_count + hot_count

            totals[max_bank] = new_max
            totals[min_bank] = new_min
            assignment[hi] = min_bank
            assignment[ci] = max_bank

            bank_neurons[max_bank] = [(n, c) for n, c in bank_neurons[max_bank] if n != hi]
            bank_neurons[min_bank] = [(n, c) for n, c in bank_neurons[min_bank] if n != ci]
            bank_neurons[min_bank].append((hi, hot_count))
            bank_neurons[max_bank].append((ci, cold_count))
            bank_neurons[min_bank].sort(key=lambda x: x[1])
            bank_neurons[max_bank].sort(key=lambda x: x[1])
            improved = True
        else:
            break

    return assignment, totals


def constrained_swap_up(data, channels, banks_per_channel):
    """Within each channel, rebalance banks using constrained swaps."""
    n = len(data)
    remap = [0] * n

    for ch in range(channels):
        ch_neurons = [(i, data[i]) for i in range(n)
                      if (i // banks_per_channel) % channels == ch]
        ch_data = [data[i] for i, _ in ch_neurons]
        ch_assignment = build_initial_assignment(ch_data, banks_per_channel)

        ch_new_assignment, _ = constrained_swap_rebalance(
            ch_data, ch_assignment, banks_per_channel
        )

        for (neuron_idx, _), bank in zip(ch_neurons, ch_new_assignment):
            remap[neuron_idx] = bank

    return remap


def generate_remap(hist_path, out_path):
    with open(hist_path) as f:
        hist = json.load(f)

    down_remap = {}
    up_remap = {}

    for layer_str, data in hist.items():
        # DOWN (flat constrained swap across all banks)
        down_assignment = build_initial_assignment(data, BANKS)
        down_assignment, down_totals = constrained_swap_rebalance(
            data, down_assignment, BANKS
        )
        down_remap[layer_str] = down_assignment

        # UP (per-channel constrained swap)
        up_remap[layer_str] = constrained_swap_up(
            data, CHANNELS, BANKS_PER_CHANNEL
        )

        max_dn = max(down_totals)
        avg_dn = sum(down_totals) / BANKS
        imb = (max_dn / avg_dn - 1) * 100 if avg_dn > 0 else 0
        print(f"  layer {layer_str}: DOWN max={max_dn}, avg={avg_dn:.1f}, "
              f"imb={imb:.2f}%")

    remap_data = {
        "down_remap": down_remap,
        "up_remap": up_remap,
        "banks": BANKS,
        "channels": CHANNELS,
        "banks_per_channel": BANKS_PER_CHANNEL,
    }

    with open(out_path, 'w') as f:
        json.dump(remap_data, f)
    print(f"\nGenerated {out_path} ({len(down_remap)} layers)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <histogram.json> <output_remap.json>")
        sys.exit(1)
    generate_remap(sys.argv[1], sys.argv[2])
