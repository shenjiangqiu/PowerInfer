#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

from gguf import GGUFReader, GGUFValueType  # noqa: E402


DEFAULT_GPUIDX = (
    "/home/sjq/sjq10t/hfhome/hub/models--PowerInfer--Bamboo-base-v0.1-gguf/"
    "snapshots/da6530bc54af41183383df7923701f8c6333b3e7/"
    "bamboo-7b-v0.1.powerinfer.gguf.generated.gpuidx"
)


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = "LITTLE" if sys.byteorder == "little" else "BIG"
    if reader.byte_order == "S":
        file_endian = "BIG" if host_endian == "LITTLE" else "LITTLE"
    else:
        file_endian = host_endian
    return host_endian, file_endian


def tensor_as_i32(tensor) -> np.ndarray:
    if tensor.data.nbytes % 4 != 0:
        raise ValueError(f"tensor {tensor.name} has unexpected byte length {tensor.data.nbytes}")
    return tensor.data.view(np.int32)


def field_value(field) -> str:
    if field is None:
        return "<missing>"
    if not field.types:
        return "<unknown>"
    if field.types[0] == GGUFValueType.STRING:
        return str(bytes(field.parts[-1]), encoding="utf-8")
    if field.types[0] == GGUFValueType.ARRAY:
        if field.types[-1] == GGUFValueType.STRING:
            return "[" + ", ".join(str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data) + "]"
        return str([pv for idx in field.data for pv in field.parts[idx].tolist()])
    return str(field.parts[-1].tolist()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PowerInfer generated .gpuidx file")
    parser.add_argument("model", nargs="?", default=DEFAULT_GPUIDX, help="Path to the .generated.gpuidx file")
    parser.add_argument("--topk", type=int, default=8, help="How many bucket entries to print per layer")
    args = parser.parse_args(None if len(sys.argv) > 1 else [DEFAULT_GPUIDX])

    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    reader = GGUFReader(path, "r")
    host_endian, file_endian = get_file_host_endian(reader)

    print(f"file: {path}")
    print(f"endian: file={file_endian}, host={host_endian}")
    print(f"gguf version: {field_value(reader.get_field('GGUF.version'))}")
    print(f"kv count: {field_value(reader.get_field('GGUF.kv_count'))}")
    print(f"tensor count: {field_value(reader.get_field('GGUF.tensor_count'))}")
    print(f"split.vram_capacity: {field_value(reader.get_field('split.vram_capacity'))}")
    print()

    layers: dict[int, dict[str, object]] = {}
    for tensor in reader.tensors:
        match = re.match(r"blk\.(\d+)\.(gpu_idx|gpu_bucket)$", tensor.name)
        if not match:
            continue
        layer_id = int(match.group(1))
        tensor_kind = match.group(2)
        layers.setdefault(layer_id, {})[tensor_kind] = tensor

    total_layers = len(layers)
    total_selected = 0
    total_neurons = 0

    print(f"layers detected: {total_layers}")
    for layer_id in sorted(layers):
        items = layers[layer_id]
        gpu_idx = items.get("gpu_idx")
        gpu_bucket = items.get("gpu_bucket")
        if gpu_idx is None or gpu_bucket is None:
            print(f"layer {layer_id:3d}: missing gpu_idx or gpu_bucket")
            continue

        idx = tensor_as_i32(gpu_idx)
        bucket = tensor_as_i32(gpu_bucket)
        selected = int(idx.sum())
        neurons = int(idx.size)
        ratio = selected / neurons if neurons else 0.0
        total_selected += selected
        total_neurons += neurons

        bad_values = np.setdiff1d(np.unique(idx), np.array([0, 1], dtype=np.int32))
        warn_suffix = f", unexpected={bad_values.tolist()}" if bad_values.size else ""
        bucket_preview = bucket[: args.topk].tolist()
        print(
            f"layer {layer_id:3d}: neurons={neurons}, selected={selected}, "
            f"ratio={ratio:.2%}, bucket={bucket.size}{warn_suffix}"
        )
        print(f"           bucket[:{args.topk}] = {bucket_preview}")

    print()
    if total_neurons:
        print(f"overall: selected={total_selected}, neurons={total_neurons}, ratio={total_selected / total_neurons:.2%}")


if __name__ == "__main__":
    main()