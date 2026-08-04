#!/usr/bin/env python3
"""
Analyze PowerInfer dump.bin - per-layer activation histograms and sparsity stats.

Binary format (from llama.cpp:6987-6990):
    int32 token_id
    int32 layer_id
    int32 batch_id
    int32 n_neurons
    float32[n_neurons] predictor scores  (activated when score > threshold, default >0)

Usage:
    python scripts/analyze_dump_bin.py dump.bin --topk 20 --threshold 0.0
    python scripts/analyze_dump_bin.py dump.bin -o histograms/ --capacity 2048
"""

import argparse
import os
import struct
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

HEADER_FMT = "<iiii"  # token_id, layer_id, batch_id, n_neurons
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def read_dump_bin(filepath: str, threshold: float = 0.0):
    """
    Read dump.bin and return:
      - per_layer_hist:  dict[layer_id] -> Counter(neuron_idx -> activation_count)
      - per_layer_stats: dict[layer_id] -> list of (n_active, n_neurons) per record
    """
    per_layer_hist: Dict[int, Counter] = defaultdict(Counter)
    per_layer_stats: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    total_records = 0

    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        while True:
            hdr_bytes = f.read(HEADER_SIZE)
            if len(hdr_bytes) < HEADER_SIZE:
                break

            token_id, layer_id, batch_id, n_neurons = struct.unpack(HEADER_FMT, hdr_bytes)

            if n_neurons <= 0:
                print(f"WARNING: record {total_records}: n_neurons={n_neurons}, skipping")
                continue

            data_bytes = f.read(n_neurons * 4)
            if len(data_bytes) < n_neurons * 4:
                print(f"WARNING: record {total_records}: truncated data, "
                      f"expected {n_neurons * 4} bytes, got {len(data_bytes)}")
                break

            scores = np.frombuffer(data_bytes, dtype=np.float32)
            active_mask = scores > threshold
            active_indices = np.where(active_mask)[0]

            per_layer_hist[layer_id].update(active_indices.tolist())
            per_layer_stats[layer_id].append((len(active_indices), n_neurons))
            total_records += 1

    total_activations = sum(c.total() for c in per_layer_hist.values())
    print(f"Read {total_records} records, {total_activations} total activations "
          f"(threshold={threshold}), {len(per_layer_hist)} layers, "
          f"{file_size / 1024 / 1024:.1f} MiB")
    return dict(per_layer_hist), dict(per_layer_stats)


def print_sparsity_stats(per_layer_stats: Dict[int, List[Tuple[int, int]]]) -> None:
    """Print per-layer average sparsity (fraction of inactive neurons)."""
    all_active = []
    all_total = []
    for lid in sorted(per_layer_stats):
        stats = per_layer_stats[lid]
        actives = [s[0] for s in stats]
        totals = [s[1] for s in stats]
        avg_active = np.mean(actives)
        avg_total = np.mean(totals)
        n_records = len(stats)
        all_active.extend(actives)
        all_total.extend(totals)
        print(f"  layer {lid:3d}: {n_records:4d} records, "
              f"avg active={avg_active:7.1f}/{int(avg_total)} "
              f"(sparsity={1 - avg_active / avg_total:.2%})")

    if all_active:
        overall_avg = np.mean(all_active)
        overall_total = np.mean(all_total)
        print(f"\n  overall: avg active={overall_avg:.1f}/{int(overall_total)} "
              f"(sparsity={1 - overall_avg / overall_total:.2%})")


def print_histogram(per_layer_hist: Dict[int, Counter], topk: int = 20) -> None:
    """Print per-layer top-k neuron activation frequency."""
    for lid in sorted(per_layer_hist):
        counter = per_layer_hist[lid]
        n_neurons = max(counter.keys()) + 1 if counter else 0
        total_activations = counter.total()
        n_records = sum(1 for idx, cnt in counter.items())  # total activations across records
        unique_activated = len(counter)

        print(f"\n--- Layer {lid} ---")
        print(f"  Total activations (sum of counts): {total_activations}")
        print(f"  Unique neurons ever activated: {unique_activated}")
        if n_neurons:
            print(f"  Coverage: {unique_activated}/{n_neurons} ({unique_activated / n_neurons:.2%})")

        print(f"  Top-{topk} most frequent neurons:")
        for neuron_idx, count in counter.most_common(topk):
            bar = "#" * min(count, 60)
            print(f"    neuron {neuron_idx:5d}: {count:6d} {bar}")


def compute_gpu_splits(
    per_layer_hist: Dict[int, Counter],
    capacity_per_layer: int,
) -> Dict[int, np.ndarray]:
    """
    Compute gpu_idx for each layer given a per-layer neuron capacity.
    Selects the `capacity_per_layer` most frequently activated neurons.

    Returns: dict[layer_id] -> gpu_idx array (int32, 1=GPU, 0=CPU)
    """
    result = {}
    for lid in sorted(per_layer_hist):
        counter = per_layer_hist[lid]
        n_neurons = max(counter.keys()) + 1 if counter else 1

        top_indices = [idx for idx, _ in counter.most_common(capacity_per_layer)]

        gpu_idx = np.zeros(n_neurons, dtype=np.int32)
        gpu_idx[top_indices] = 1
        result[lid] = gpu_idx

        selected = gpu_idx.sum()
        print(f"  layer {lid}: {selected}/{n_neurons} selected ({selected / n_neurons:.2%})")

    return result


def save_histogram(per_layer_hist: Dict[int, Counter], output_dir: str) -> None:
    """Save per-layer activation histograms as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    for lid, counter in sorted(per_layer_hist.items()):
        n_neurons = max(counter.keys()) + 1 if counter else 1
        hist = np.zeros(n_neurons, dtype=np.int64)
        for idx, cnt in counter.items():
            hist[idx] = cnt
        fpath = os.path.join(output_dir, f"activation_{lid}.npy")
        np.save(fpath, hist)
        print(f"  Saved {fpath} ({hist.sum()} total activations, {n_neurons} neurons)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PowerInfer dump.bin - per-layer neuron activation histograms"
    )
    parser.add_argument("dump_bin", help="Path to dump.bin")
    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="Activation threshold, neuron active when score > threshold (default: 0.0)")
    parser.add_argument("--topk", type=int, default=20,
                        help="Top-k neurons to print per layer")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Directory to save per-layer activation_<N>.npy histograms")
    parser.add_argument("--capacity", type=int, default=0,
                        help="If set, compute gpu_idx with given per-layer neuron capacity")
    args = parser.parse_args()

    if not os.path.exists(args.dump_bin):
        raise SystemExit(f"File not found: {args.dump_bin}")

    per_layer_hist, per_layer_stats = read_dump_bin(args.dump_bin, args.threshold)

    print("\n=== Per-layer sparsity (avg active neurons per token) ===")
    print_sparsity_stats(per_layer_stats)

    print("\n=== Per-layer neuron frequency histogram ===")
    print_histogram(per_layer_hist, topk=args.topk)

    if args.output:
        print("\n=== Saving histograms ===")
        save_histogram(per_layer_hist, args.output)

    if args.capacity > 0:
        print(f"\n=== GPU split with capacity={args.capacity} per layer ===")
        compute_gpu_splits(per_layer_hist, args.capacity)


if __name__ == "__main__":
    main()
