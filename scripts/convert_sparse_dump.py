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
    python convert_sparse_dump.py scores.bin -t 0.0 -o dump.jsonl
    python convert_sparse_dump.py scores.bin --threshold 0.5

Default threshold is 0.0 (same as the original JSONL dump behavior).
"""

import argparse
import json
import struct
import sys


def read_binary(filepath: str):
    """Yield (token, layer, batch, scores_ndarray) tuples from binary dump."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PowerInfer binary sparse dump to JSONL")
    parser.add_argument("input", help="Binary dump file (from POWERINFER_DUMP_BINARY)")
    parser.add_argument("-t", "--threshold", type=float, default=0.0,
                        help="Neuron activation threshold (default: 0.0)")
    parser.add_argument("-o", "--output", help="Output JSONL file (default: stdout)")
    args = parser.parse_args()

    out = open(args.output, "w") if args.output else sys.stdout

    try:
        for token, layer, batch, n_neurons, scores in read_binary(args.input):
            active = (scores > args.threshold).nonzero()[0].tolist()
            record = {
                "token": token,
                "layer": layer,
                "batch": batch,
                "total": n_neurons,
                "active": len(active),
                "indices": active,
            }
            out.write(json.dumps(record) + "\n")
    except BrokenPipeError:
        pass
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
