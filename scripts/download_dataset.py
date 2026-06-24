#!/usr/bin/env python3
"""Download dataset prompts for PowerInfer sparse dump.

Usage:
    python download_dataset.py wiki    -o datasets/wiki_prompts.txt    --max 100
    python download_dataset.py c4      -o datasets/c4_prompts.txt
    python download_dataset.py alpaca  -o datasets/alpaca_prompts.txt
    python download_dataset.py chatgpt -o datasets/chatgpt_prompts.txt
"""

import argparse
import sys


def download_wiki(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")
    prompts = []
    for row in ds:
        text = row["text"]
        if text.strip():
            prompts.append(text.strip()[:2048])
        if len(prompts) >= max_prompts:
            break

    with open(output_file, "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"wiki: {len(prompts)} prompts saved to {output_file}", file=sys.stderr)


def download_c4(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    with open(output_file, "w") as f:
        count = 0
        for row in ds:
            if count >= max_prompts:
                break
            f.write(row["text"].strip()[:2048] + "\n")
            count += 1
    print(f"c4: {count} prompts saved to {output_file}", file=sys.stderr)


def download_alpaca(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    with open(output_file, "w") as f:
        count = 0
        for row in ds:
            if count >= max_prompts:
                break
            q = row["instruction"]
            if row.get("input"):
                q += " " + row["input"]
            if q.strip():
                f.write(q.strip()[:2048] + "\n")
                count += 1
    print(f"alpaca: {count} prompts saved to {output_file}", file=sys.stderr)


def download_chatgpt(output_file: str, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True)
    with open(output_file, "w") as f:
        count = 0
        for row in ds:
            if count >= max_prompts:
                break
            q = row["conversation_a"][0]["content"] if row["conversation_a"] else ""
            if q.strip():
                f.write(q.strip()[:2048] + "\n")
                count += 1
    print(f"chatgpt: {count} prompts saved to {output_file}", file=sys.stderr)


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
