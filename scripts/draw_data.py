#!/usr/bin/env python3
"""Draw bar chart from PowerInfer cycle JSON.

Usage:
    python draw_data.py cycle.json
    python draw_data.py cycle.json -o perf.png
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# hardcoded: single-value keys
SINGLES = [
    "gpu_cycle",
    "gpu_cycle_sparse",
    "down_total_rowwise_bitserial_time_method_1",
    "down_total_rowwise_bitserial_time_method_2",
]

# hardcoded: stacked pair prefixes → display name
PAIRS = {
    "up_total_naive_time": "up naive",
    "up_total_asnc_time": "up async",
    "up_total_iterleave_time": "up interleave",
    "down_total_interproduct_time_single": "down single",
    "down_total_interproduct_time_two": "down two",
    "up_dense": "up dense",
    "down_dense": "down dense",
    
}

# keys that are visually capped (short bar + number label)
CAPPED_KEYS = {
    "gpu_cycle",
    "gpu_cycle_sparse",
    "down_total_rowwise_bitserial_time_method_1",
    "down_total_rowwise_bitserial_time_method_2",
}


def format_value(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.2f}G"
    elif v >= 1e6:
        return f"{v / 1e6:.1f}M"
    elif v >= 1e3:
        return f"{v / 1e3:.0f}K"
    else:
        return f"{v:.0f}"


def draw(data: dict, output: str, title: str, cap_height_ratio: float = 0.15):
    """Draw hardcoded bars. Capped keys get short gray bar + red annotation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # find a reasonable reference height for capped bars
    all_capped = []
    for k in CAPPED_KEYS:
        if k in data:
            all_capped.append(data[k])
    for prefix in PAIRS:
        o = data.get(f"{prefix}_row_open", 0)
        c = data.get(f"{prefix}_compute", 0)
        if o or c:
            all_capped.append(o + c)
    if all_capped:
        max_normal = max(v for v in all_capped if v < 2e9)
        if max_normal == 0:
            max_normal = 1e8
    else:
        max_normal = 1e8

    cap_visual = max_normal * cap_height_ratio
    idx = 0
    labels = []

    # ---- pair bars ----
    for prefix, label in PAIRS.items():
        o = data.get(f"{prefix}_row_open", 0)
        c = data.get(f"{prefix}_compute", 0)
        total = o + c
        if total == 0:
            continue

        ax.bar(idx, o, color="#5B9BD5", edgecolor="white", linewidth=0.5)
        ax.bar(idx, c, bottom=o, color="#ED7D31", edgecolor="white", linewidth=0.5)
        ax.text(
            idx,
            total,
            format_value(total),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        labels.append(label)
        idx += 1

    # ---- single bars ----
    for key in SINGLES:
        v = data.get(key, 0)
        if v == 0:
            continue
        if key in CAPPED_KEYS:
            ax.bar(idx, cap_visual, color="#CCCCCC", edgecolor="white", linewidth=0.5)
            ax.text(
                idx,
                cap_visual,
                f"↓ {format_value(v)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#C00000",
            )
        else:
            ax.bar(idx, v, color="#70AD47", edgecolor="white", linewidth=0.5)
            ax.text(
                idx,
                v,
                format_value(v),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        labels.append(key.replace("_", " ").replace("  ", " "))
        idx += 1

    ax.set_xticks(range(idx))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cycles / Time")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.margins(y=0.12)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#5B9BD5", label="row open"),
            Patch(facecolor="#ED7D31", label="compute"),
            Patch(facecolor="#70AD47", label="single value"),
            Patch(facecolor="#CCCCCC", label="capped (actual in red)"),
        ],
        loc="upper right",
        fontsize=7,
    )

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Draw PowerInfer cycle chart")
    parser.add_argument("input", help="JSON file (e.g. cycle.json)")
    parser.add_argument("-o", "--output", help="Output PNG (default: <input>.png)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output = args.output or (os.path.splitext(args.input)[0] + ".png")
    title = os.path.splitext(os.path.basename(args.input))[0]
    draw(data, output, title)


if __name__ == "__main__":
    main()
