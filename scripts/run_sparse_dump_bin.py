#!/usr/bin/env python3
"""Batch run PowerInfer binary dump across models and datasets.

Usage:
    python run_sparse_dump_bin.py
    python run_sparse_dump_bin.py --models ReluLLaMA-7B,OPT-6.7B --datasets wiki,c4 --max-prompts 10
    python run_sparse_dump_bin.py --models /path/to/model.gguf --datasets wiki --dumpdir /tmp/dumpbins
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
POWERINFER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from download_dataset import (
    download_wiki,
    download_c4,
    download_alpaca,
    download_chatgpt,
)

MODEL_MAP = {
    "ReluLLaMA-7B": (
        "/home/sjq/sjq10t/hfhome/hub/models--PowerInfer--ReluLLaMA-7B-PowerInfer-GGUF/snapshots/17b5d8a28f0377e05758a74989ce9326f5905860/"
        + "llama-7b-relu.powerinfer.gguf"
    ),
    "Bamboo-7B": "/home/sjq/sjq10t/hfhome/hub/models--PowerInfer--Bamboo-base-v0.1-gguf/snapshots/da6530bc54af41183383df7923701f8c6333b3e7/bamboo-7b-v0.1.powerinfer.gguf",
    "Bamboo-dpo-7B": "/home/sjq/sjq10t/hfhome/hub/models--PowerInfer--Bamboo-DPO-v0.1-gguf/snapshots/c3847e080bd8664f2ea299400f74665ca5b13824/bamboo-7b-dpo-v0.1.powerinfer.gguf",
    "ProSparse-llama-7b": "/home/sjq/sjq10t/hfhome/hub/models--PowerInfer--ProSparse-LLaMA-2-7B-GGUF/snapshots/72f4a41882ede0cbc745150559b4db268a142b36/prosparse-llama-2-7b-clip15.gguf",
}

DATASET_HANDLERS = {
    "wiki": download_wiki,
    "c4": download_c4,
    "alpaca": download_alpaca,
    "chatgpt": download_chatgpt,
}


def resolve_model(key: str, model_dir: str) -> str:
    if os.path.isfile(key):
        return key
    handler = MODEL_MAP.get(key)
    if handler is None:
        sys.exit(f"ERROR: unknown model key '{key}' and not a file path")
    if isinstance(handler, str):
        return handler
    return handler(model_dir)


def ensure_dataset(name: str, max_prompts: int) -> Path:
    datasets_dir = POWERINFER_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = datasets_dir / f"{name}_prompts.jsonl"

    if prompt_file.exists():
        return prompt_file

    handler = DATASET_HANDLERS.get(name)
    if handler is None:
        print(f"Unknown dataset '{name}', writing default prompt", file=sys.stderr)
        prompt_file.write_text(json.dumps("once upon a time") + "\n")
        return prompt_file

    print(f"=== Downloading dataset: {name} ===", file=sys.stderr)
    handler(str(prompt_file), max_prompts)
    return prompt_file


def main():
    parser = argparse.ArgumentParser(description="Batch PowerInfer binary sparse dump")
    parser.add_argument(
        "--models",
        default="ReluLLaMA-7B",
        help="Comma-separated model keys or paths (default: ReluLLaMA-7B)",
    )
    parser.add_argument(
        "--datasets",
        default="wiki",
        help="Comma-separated datasets: wiki,c4,alpaca,chatgpt (default: wiki)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=100,
        help="Max prompts per dataset (default: 100)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=1,
        help="Tokens to generate per prompt (default: 1)",
    )
    parser.add_argument(
        "--threads", type=int, default=20, help="CPU threads (default: 20)"
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        help="Output directory for binary dumps (default: <repo>/dumpbins)",
    )
    parser.add_argument(
        "--main-bin",
        default=None,
        help="Path to main binary (default: <repo>/build_release/bin/main)",
    )
    parser.add_argument(
        "--model-dir", default=None, help="Directory containing model files"
    )
    args = parser.parse_args()

    powerinfer_dir = POWERINFER_DIR
    dumpdir = Path(args.dumpdir) if args.dumpdir else powerinfer_dir / "dumpbins"
    dumpdir.mkdir(parents=True, exist_ok=True)

    default_bin = powerinfer_dir / "build_release" / "bin" / "main"
    main_bin = Path(args.main_bin) if args.main_bin else default_bin
    if not main_bin.exists():
        sys.exit(f"ERROR: main binary not found: {main_bin}")

    model_dir = args.model_dir or "."

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    failed = 0
    total = 0

    for model_key in models:
        model_path = resolve_model(model_key, model_dir)
        if not model_path or not os.path.isfile(model_path):
            print(f"SKIP: model not found: {model_path}", file=sys.stderr)
            continue

        model_short = Path(model_key).stem

        for ds in datasets:
            prompt_file = ensure_dataset(ds, args.max_prompts)
            if not prompt_file:
                continue

            print(f"=== Model: {model_short}  Dataset: {ds} ===")

            prompts = []
            for line in prompt_file.read_text().strip().split("\n"):
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
            for i, prompt in enumerate(prompts, 1):
                if not prompt.strip():
                    continue
                if i > args.max_prompts:
                    break

                dumpfile = dumpdir / f"{model_short}-{ds}-{i}.bin"

                if dumpfile.exists() and dumpfile.stat().st_size > 0:
                    print(f"  [{i}] SKIP (exists: {dumpfile})")
                    continue

                print(f"  [{i}] {prompt[:80]}")

                env = os.environ.copy()
                env["POWERINFER_DUMP_BINARY"] = str(dumpfile)

                proc = subprocess.run(
                    [
                        str(main_bin),
                        "-m",
                        model_path,
                        "-p",
                        prompt,
                        "-n",
                        str(args.n_predict),
                        "-t",
                        str(args.threads),
                        "-c",
                        "2048",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                total += 1

                if proc.returncode != 0:
                    print(f"    ** FAILED (rc={proc.returncode}) **")
                    last_line = (
                        proc.stderr.strip().split("\n")[-1] if proc.stderr else ""
                    )
                    if last_line:
                        print(f"    {last_line[:200]}")
                    failed += 1
                    dumpfile.unlink(missing_ok=True)
                else:
                    size = dumpfile.stat().st_size if dumpfile.exists() else 0
                    print(f"    -> {dumpfile}  ({size} bytes)")

    print()
    print("=== Done ===")
    print(f"Total runs: {total}  Failed: {failed}")
    print(f"Outputs: {dumpdir}/")
    bins = sorted(dumpdir.iterdir())
    for f in bins[:20]:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
    if len(bins) > 20:
        print(f"  ... and {len(bins) - 20} more")


if __name__ == "__main__":
    main()
