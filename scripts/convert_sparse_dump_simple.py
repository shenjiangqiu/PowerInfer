#!/usr/bin/env python3
"""
Convert PowerInfer binary sparse dump to JSONL with a configurable threshold.

Binary format (per record):
    int32 token_id
    int32 layer_id
    int32 batch_seq
    int32 n_neurons
    float[n_neurons] scores

Usage:
    # single file → output to stdout or -o
    python convert_sparse_dump_simple.py scores.bin -t 0.0 -o dump.jsonl

    # directory → convert all .bin files, write .jsonl next to each
    python convert_sparse_dump_simple.py dumpbins/ -t 0.5
"""

import argparse
import json
import os
import struct
import sys


def read_binary(filepath: str):
    """Yield (token, layer, batch, n_neurons, scores_ndarray) tuples."""
    import numpy as np

    with open(filepath, "rb") as f:
        while True:
            hdr = f.read(16)
            if not hdr:
                break
            if len(hdr) < 16:
                print(f"Warning: truncated header at offset {f.tell() - len(hdr)}, stopping.",
                      file=sys.stderr)
                break
            token, layer, batch, n_neurons = struct.unpack("<iiii", hdr)
            raw = f.read(n_neurons * 4)
            if len(raw) < n_neurons * 4:
                print(f"Warning: truncated data for token {token} layer {layer}, stopping.",
                      file=sys.stderr)
                break
            scores = np.frombuffer(raw, dtype=np.float32)
            yield token, layer, batch, n_neurons, scores


def convert_bin(bin_path: str, threshold: float, out_path: str = None):
    """Convert a single .bin file to JSONL."""
    out = open(out_path, "w") if out_path else sys.stdout
    records = 0
    try:
        for token, layer, batch, n_neurons, scores in read_binary(bin_path):
            active_count = int((scores > threshold).sum())
            record = {
                "token": token,
                "layer": layer,
                "batch": batch,
                "total": n_neurons,
                "active": active_count,
            }
            out.write(json.dumps(record) + "\n")
            records += 1
    except BrokenPipeError:
        pass
    finally:
        if out_path:
            out.close()
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert PowerInfer binary sparse dump to JSONL")
    parser.add_argument(
        "input", help="Binary dump file or directory of .bin files")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.0,
        help="Neuron activation threshold (default: 0.0)")
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file (ignored when input is a directory)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        bins = sorted(
            f for f in os.listdir(args.input) if f.endswith(".bin"))
        if not bins:
            print(f"No .bin files found in {args.input}", file=sys.stderr)
            sys.exit(1)

        for name in bins:
            bin_path = os.path.join(args.input, name)
            jsonl_path = os.path.join(args.input, name.rsplit(".bin", 1)[0] + ".jsonl")
            records = convert_bin(bin_path, args.threshold, jsonl_path)
            print(f"{name} -> {os.path.basename(jsonl_path)}  ({records} records)")

        print(f"\nDone: {len(bins)} files converted in {args.input}")
    else:
        if args.output is None and sys.stdout.isatty():
            records = convert_bin(args.input, args.threshold, args.input.rsplit(".bin", 1)[0] + ".jsonl")
            print(f"Wrote {records} records", file=sys.stderr)
        else:
            convert_bin(args.input, args.threshold, args.output)


if __name__ == "__main__":
    main()
