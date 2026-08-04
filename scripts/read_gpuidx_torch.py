#!/usr/bin/env python3
"""
Read a .gpuidx.json + .gpuidx.bin pair (exported by gpuidx-export tool)
and reconstruct torch tensors.

Usage:
    python scripts/read_gpuidx_torch.py <prefix>

Example:
    gpuidx-export bamboo-7b-v0.1.powerinfer.gguf.generated.gpuidx
    python scripts/read_gpuidx_torch.py bamboo-7b-v0.1.powerinfer.gguf

This produces:
    - .gpuidx.json  (metadata)
    - .gpuidx.bin   (raw concatenated tensor data)

You can also pass the .gpuidx.json path directly:
    python scripts/read_gpuidx_torch.py bamboo-7b-v0.1.powerinfer.gguf.gpuidx.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


GGML_DTYPE_TO_NUMPY: Dict[str, np.dtype] = {
    "i32": np.int32,
    "f32": np.float32,
    "f16": np.float16,
    "i8": np.int8,
    "i16": np.int16,
    "i64": np.int64,
    "f64": np.float64,
    "u8": np.uint8,
    "u32": np.uint32,
}

GGML_DTYPE_TO_TORCH: Dict[str, torch.dtype] = {
    "i32": torch.int32,
    "f32": torch.float32,
    "f16": torch.float16,
    "i8": torch.int8,
    "i16": torch.int16,
    "i64": torch.int64,
    "f64": torch.float64,
    "u8": torch.uint8,
    "u32": torch.int32,  # torch has no uint32, use int32
}


def parse_gpuidx_export(json_path: str) -> Tuple[List[dict], np.ndarray]:
    """
    Parse a gpuidx.json file and its corresponding .bin file.

    Returns:
        tensors_meta: list of dicts with keys: name, dtype, shape, offset, nbytes
        full_data: np.ndarray (uint8) of the entire binary blob (memory-mapped)
    """
    json_path = Path(json_path)
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".gpuidx.json")
    bin_path = json_path.with_suffix("")

    # Auto-detect: if json path given ends with .gpuidx.json, strip it for bin
    json_stem = json_path.stem  # e.g. "foo.gpuidx"
    if json_stem.endswith(".gpuidx"):
        bin_path = json_path.with_name(json_stem)
    else:
        bin_path = json_path.with_suffix(".gpuidx.bin")
        if not bin_path.exists():
            bin_path = json_path.with_suffix("")  # try without suffix
            if not bin_path.exists():
                # try same name minus .json
                bin_path = Path(str(json_path)[:-5])

    # Find the bin: try multiple variants
    candidates = [
        bin_path,
        Path(str(json_path).replace(".gpuidx.json", ".gpuidx.bin")),
        Path(str(json_path)[:-5]),  # strip .json
    ]
    bin_path = None
    for c in candidates:
        if c.exists():
            bin_path = c
            break

    if bin_path is None:
        # Last resort: look for any .gpuidx.bin next to the json
        base = str(json_path)
        for suffix in [".gpuidx.json", ".json"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        candidate = Path(base + ".gpuidx.bin")
        if candidate.exists():
            bin_path = candidate

    if bin_path is None:
        raise FileNotFoundError(
            f"Could not find .bin file for {json_path}. "
            f"Tried: {candidates}"
        )

    print(f"Loading metadata from: {json_path}")
    print(f"Loading binary data from: {bin_path}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    full_data = np.memmap(str(bin_path), dtype=np.uint8, mode="r")

    return meta["tensors"], full_data


def load_tensors(
    json_path: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load all tensors from a gpuidx export and return a dict of torch tensors.

    Args:
        json_path: Path to the .gpuidx.json file
        device: torch device to place tensors on

    Returns:
        Dict mapping tensor name (e.g. "blk.0.gpu_idx") to torch.Tensor
    """
    tensors_meta, full_data = parse_gpuidx_export(json_path)
    tensors: Dict[str, torch.Tensor] = {}

    for tmeta in tensors_meta:
        name = tmeta["name"]
        ggml_dtype = tmeta["dtype"]
        shape = tuple(tmeta["shape"])
        offset = tmeta["offset"]
        nbytes = tmeta["nbytes"]

        np_dtype = GGML_DTYPE_TO_NUMPY.get(ggml_dtype)
        torch_dtype = GGML_DTYPE_TO_TORCH.get(ggml_dtype)

        if np_dtype is None or torch_dtype is None:
            print(f"  WARNING: unknown dtype '{ggml_dtype}' for tensor '{name}', skipping")
            continue

        raw = full_data[offset : offset + nbytes]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch_dtype)
        tensors[name] = tensor

    return tensors


def analyze_layers(tensors: Dict[str, torch.Tensor]) -> None:
    """Print per-layer statistics for gpu_idx / gpu_bucket tensors."""
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        match = re.match(r"blk\.(\d+)\.(gpu_idx|gpu_bucket)$", name)
        if not match:
            continue
        layer_id = int(match.group(1))
        kind = match.group(2)
        layers.setdefault(layer_id, {})[kind] = tensor

    total_selected = 0
    total_neurons = 0

    print(f"\nLayer analysis ({len(layers)} layers):")
    for lid in sorted(layers):
        items = layers[lid]
        gpu_idx = items.get("gpu_idx")
        gpu_bucket = items.get("gpu_bucket")
        if gpu_idx is None or gpu_bucket is None:
            print(f"  layer {lid:3d}: missing gpu_idx or gpu_bucket")
            continue

        selected = int(gpu_idx.sum().item())
        neurons = int(gpu_idx.numel())
        ratio = selected / neurons if neurons else 0.0
        total_selected += selected
        total_neurons += neurons

        bucket_head = gpu_bucket[:8].tolist()
        print(
            f"  layer {lid:3d}: neurons={neurons}, selected={selected}, "
            f"ratio={ratio:.2%}, bucket[:8]={bucket_head}"
        )

    if total_neurons:
        print(
            f"\n  overall: selected={total_selected}, neurons={total_neurons}, "
            f"ratio={total_selected / total_neurons:.2%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Load PowerInfer gpuidx export into torch tensors"
    )
    parser.add_argument(
        "prefix",
        help="Path to the .gpuidx.json file, or the prefix (without .gpuidx.json/.bin)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device to place tensors on (default: cpu)",
    )
    args = parser.parse_args()

    json_path = args.prefix
    if not json_path.endswith(".json"):
        json_path = json_path + ".gpuidx.json"

    if not os.path.exists(json_path):
        raise SystemExit(f"File not found: {json_path}")

    tensors = load_tensors(json_path, device=args.device)

    print(f"\nLoaded {len(tensors)} tensors:")
    for name, t in tensors.items():
        shape_str = "x".join(str(s) for s in t.shape)
        print(f"  {name}: shape=({shape_str}), dtype={t.dtype}, device={t.device}")

    analyze_layers(tensors)


if __name__ == "__main__":
    main()
