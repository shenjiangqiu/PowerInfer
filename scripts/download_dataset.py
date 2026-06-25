#!/usr/bin/env python3
"""Download dataset prompts for PowerInfer sparse dump.

Saves as JSONL: one JSON-encoded string per line (preserves newlines etc.).

Usage:
    python download_dataset.py wiki    -o datasets/wiki_prompts.jsonl    --max 100
    python download_dataset.py c4      -o datasets/c4_prompts.jsonl
    python download_dataset.py alpaca  -o datasets/alpaca_prompts.jsonl
    python download_dataset.py chatgpt -o datasets/chatgpt_prompts.jsonl
"""

import argparse
import json
import sys

MIN_WORDS = 200
MAX_CHARS = 2048


def _valid(text: str) -> bool:
    return text.strip() and len(text.split()) >= MIN_WORDS


def _save(prompts: list, output_file: str):
    with open(output_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def download_wiki(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")
    prompts = []
    for row in ds:
        text = row["page"]
        if _valid(text):
            prompts.append(text.strip()[:MAX_CHARS])
        if len(prompts) >= max_prompts:
            break
    _save(prompts, output_file)
    print(f"wiki: {len(prompts)} prompts saved to {output_file}", file=sys.stderr)


def download_c4(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for row in ds:
        text = row["text"].strip()
        if _valid(text):
            prompts.append(text[:MAX_CHARS])
        if len(prompts) >= max_prompts:
            break
    _save(prompts, output_file)
    print(f"c4: {len(prompts)} prompts saved to {output_file}", file=sys.stderr)


def download_alpaca(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    prompts = []
    for row in ds:
        q = row["instruction"]
        if row.get("input"):
            q += " " + row["input"]
        if _valid(q):
            prompts.append(q.strip()[:MAX_CHARS])
        if len(prompts) >= max_prompts:
            break
    _save(prompts, output_file)
    print(f"alpaca: {len(prompts)} prompts saved to {output_file}", file=sys.stderr)


def download_chatgpt(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True)
    prompts = []
    for row in ds:
        q = row["conversation_a"][0]["content"] if row["conversation_a"] else ""
        if _valid(q):
            prompts.append(q.strip()[:MAX_CHARS])
        if len(prompts) >= max_prompts:
            break
    _save(prompts, output_file)
    print(f"chatgpt: {len(prompts)} prompts saved to {output_file}", file=sys.stderr)


DATASETS = {
    "wiki": download_wiki,
    "c4": download_c4,
    "alpaca": download_alpaca,
    "chatgpt": download_chatgpt,
}


def main():
    parser = argparse.ArgumentParser(description="Download dataset prompts")
    parser.add_argument("dataset", choices=list(DATASETS), help="Dataset name")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--max", type=int, default=100, help="Max prompts (default: 100)")
    args = parser.parse_args()

    DATASETS[args.dataset](args.output, args.max)


if __name__ == "__main__":
    main()
