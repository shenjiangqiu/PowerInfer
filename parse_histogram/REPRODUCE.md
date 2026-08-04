# Reproducing the async-bank-PIM mapping experiments

This is a step-by-step log of everything needed to regenerate the data,
mappings, and reports behind the `paper/aysnc_bank_pim` paper and the
follow-up analyses (gpuidx comparison, batch-group sensitivity). Run
commands from the repo root unless noted.

## 0. Prerequisites

- `build_release/bin/main` built with the `POWERINFER_DUMP_BINARY` /
  `POWERINFER_DUMP_SPARSE` sparse-dump feature (already in `llama.cpp`).
  Rebuild if you've changed the C++ side:
  ```
  cmake --build build_release/ --config Release -j
  ```
- The `parse_histogram` Rust CLI installed:
  ```
  just install
  ```
  Re-run this after any change under `parse_histogram/src/`.
- A ReluLLaMA-7B GGUF checkpoint with its `activation/` directory alongside
  it (used in step 5) — this ships with the HuggingFace download and is
  what PowerInfer's own GPU/CPU split solver reads.
- Python with `numpy` for the analysis snippets; the repo's `.venv` has
  `torch` (needed only for step 5's gpuidx comparison — the main gguf-py
  based reader in `scripts/read_gpuidx.py` does not need torch).

## 1. Regenerate the calibration/evaluation traces

`runall_mac.sh` drives `scripts/run_sparse_dump_bin.py`, which runs `main`
once per prompt with `POWERINFER_DUMP_BINARY` set, producing one `.bin` per
prompt in `./dumpbins_ReluLLaMA-7B/`. As configured it processes 20
WikiText passages:

```
bash runall_mac.sh
```

This takes on the order of an hour and produces ~14 GB of `.bin` files
(20 prompts × 32 layers × however many prefill positions each passage
tokenizes to). Do not edit `runall_mac.sh` itself — if you need different
prompts/counts, change the flags via a copy or pass them straight to
`scripts/run_sparse_dump_bin.py`.

Split the 20 files by parity into a calibration half (used to *build* the
mapping) and a disjoint evaluation half (used to *measure* it), so the
reported numbers show generalization rather than fitting the test set:

```bash
mkdir -p /tmp/calib_bins /tmp/eval_bins
for f in dumpbins_ReluLLaMA-7B/ReluLLaMA-7B-wiki-*.bin; do
  n=$(basename "$f" .bin | sed -E 's/.*-([0-9]+)$/\1/')
  if [ $((n % 2)) -eq 1 ]; then ln -sf "$PWD/$f" /tmp/calib_bins/; else ln -sf "$PWD/$f" /tmp/eval_bins/; fi
done
```

## 2. Sanity-check the traces

```bash
parse_histogram -f dumpbins_ReluLLaMA-7B sparsity   # overall + per-layer sparsity
parse_histogram -f dumpbins_ReluLLaMA-7B print -n 20  # first 20 records raw
```

Expect ~80% sparsity (~20% of 11008 neurons active per token) and, if you
inspect a `.bin` file directly, no two layers should ever be byte-identical
(that was the original PIM-dump bug — see the C++ history for the
`llama_sparse_dump_reset` / per-layer-buffer fix).

## 3. Generate the calibration-based bank mapping

```bash
parse_histogram -f /tmp/calib_bins gen-mapping \
  -o /tmp/remap_greedy.json \
  --strategy greedy \
  --capacity-slack 1.2 \
  --report 2> /tmp/mapping_quality.log
```

`--strategy` also accepts `modulo` (baseline, ignores calibration data) and
`round-robin` (sorted-by-probability round robin) — regenerate with each to
reproduce the strategy-comparison table:

```bash
parse_histogram -f /tmp/calib_bins gen-mapping -o /tmp/remap_modulo.json -s modulo --report 2> /tmp/gen_modulo.log
parse_histogram -f /tmp/calib_bins gen-mapping -o /tmp/remap_rr.json     -s round-robin --report 2> /tmp/gen_rr.log
parse_histogram -f /tmp/calib_bins gen-mapping -o /tmp/remap_greedy.json -s greedy --report 2> /tmp/gen_greedy.log
```

`--report` prints one line per layer to stderr: per-bank neuron count
(min/max/mean/stddev) and the load-imbalance ratio (busiest bank's expected
load / mean expected load — 1.0 is perfect). This is what produced the
paper's Table I numbers (greedy ≈1.01x worst-layer, round-robin ≈1.41x,
modulo ≈2.53x).

For the span-aligned (multi-bank-per-neuron) placement mode, add
`--up-span 4` (must divide `--up-banks`, default 1024):

```bash
parse_histogram -f /tmp/calib_bins gen-mapping -o /tmp/remap_span4.json --up-span 4 --report
```

Sanity check the alignment on the resulting JSON — every UP bank index
should be a multiple of the span:

```bash
python3 -c "
import json
d = json.load(open('/tmp/remap_span4.json'))
banks = d['up_remap']['0']
print('all multiples of 4:', all(b % 4 == 0 for b in banks))
"
```

## 4. Measure the imbalance: naive vs. remap vs. ideal

```bash
parse_histogram -f /tmp/eval_bins report -r /tmp/remap_greedy.json
```

This runs the simulator twice over the held-out evaluation bins — once
with no remap (`i % banks`) and once with the calibrated table — and
prints the busiest-bank row count for both `up_async` and `down_async`,
plus the per-record dynamic-oracle `ideal` (best any *static* mapping
could ever do, since it's allowed a different assignment per token) and
the resulting speedup / remap-vs-ideal ratios. Expect roughly:

```
metric        naive(rows)   remap(rows)    ideal(rows)   speedup   remap/ideal
up_async      18848144      17806464       7456736       1.059x    2.388x
down_async    18848144      17806464       7456736       1.059x    2.388x
```

`up_async` and `down_async` come out identical here because both arrays
use the same row-cost constants and were packed by the same algorithm on
the same probabilities — that's expected, not a bug (verified by feeding
the `modulo` table back in as `--remap` and confirming it reproduces the
`naive` column exactly, speedup 1.000x).

To reproduce the batch-group sensitivity study (grouping 2-4 consecutive
same-layer records into one dispatched round before computing the busiest
bank):

```bash
for bg in 1 2 3 4; do
  echo "=== batch_group=$bg ==="
  parse_histogram -f /tmp/eval_bins report -r /tmp/remap_greedy.json --batch-group $bg
done
```

Expect the `up_batch`/`down_batch` rows (only printed when `--batch-group
> 1`) to show: total work shrinking as batch size grows (more requests
share the same resident row), `remap/ideal` improving (2.39x → 1.83x) as
pooling more active neurons per round reduces relative sampling variance,
but the naive-vs-remap `speedup` itself *shrinking slightly* (1.059x →
1.040x) — batching dilutes the skew signal remap corrects for, since
larger unions increasingly saturate toward "most neurons appear at least
once regardless of which specific tokens you grouped."

## 4b. Span-distributed placement (striping a neuron's row across banks)

`report`'s span=1 numbers (1.06x) look unimpressive next to the
gen-mapping report's near-perfect static balance (1.01x) because at
this sparsity level (~2 active neurons/bank), the busiest bank on any
single token is dominated by sampling variance, not skew — see the
`layout-report` command below, which stripes each neuron's row across
`--up-span` physical banks (pooling more neurons per calibrated placement
unit *and* letting the group stream a row in parallel) and separately
tests down-projection's *clustering* + *staggering* levers:

```bash
for s in 1 2 4 8 16; do
  echo "=== up_span=$s down_span=$s ==="
  parse_histogram -f /tmp/eval_bins layout-report --calib /tmp/calib_bins --up-span $s --down-span $s
done
```

Expect up-projection speedup (vs. the true channel-lockstep naive
baseline) to climb monotonically: 1.65x (calibration alone, span=1) →
2.17x → 2.73x → 3.28x → 3.78x at span=16, with the "distributed, no
stagger" and "distributed + staggered" rows *identical* at every span
(up-projection neurons share no structure, so staggering is a no-op for
them — this is itself a validation of the model, not a bug). For
down-projection, expect: non-distributed to already match the dense
(no-skip) reference almost exactly (256-neuron row-groups are active with
near-certainty at ~20% sparsity); "distributed, no stagger" to get
*worse* as span grows (0.977x at span 4, 0.896x at span 8 — i.e. 10%
*more* work); "distributed + staggered" to land back exactly on the
non-distributed baseline at every span (staggering is damage control, not
a net win); and "down-clustered" (sorting neurons by probability before
assigning row-groups, independent of span) to be the only lever that
actually beats the baseline, by a modest ~0.4%.

## 5. Compare against PowerInfer's own precomputed hot-neuron index

PowerInfer ships a `.generated.gpuidx` file per model (a GGUF container of
per-layer `gpu_idx`/`gpu_bucket` tensors) alongside a `activation/`
directory of `activation_{layer}.pt` tensors — raw per-neuron activation
*counts* from PowerInfer's own (larger, different) calibration run, used by
`powerinfer-py/powerinfer/export_split.py`'s `torch.topk` to decide which
neurons are GPU-resident under a given VRAM budget (see
`llm_load_gpu_split_with_budget` in `llama.cpp` for the calling code).

Inspect a `.generated.gpuidx` file directly:

```bash
python3 scripts/read_gpuidx.py gpuidx/llama-7b-relu.powerinfer.gguf.generated.gpuidx
```

Note: if `split.vram_capacity` in that file is larger than the whole model
(check the printed value), every layer will show `selected=100%` — the
split is trivial (everything fits in VRAM) and carries no ranking
information. That was the case for the shipped `llama-7b-relu` gpuidx in
this repo; the ranking signal instead comes directly from the
`activation_{layer}.pt` files, not from `gpu_idx` itself.

To compare PowerInfer's own per-neuron activation ranking against our
calibration histogram (Spearman rank correlation + top-K overlap):

```bash
parse_histogram -f dumpbins_ReluLLaMA-7B histogram -o /tmp/full_histogram.json

MODEL_DIR="<path to the model's snapshot dir containing activation/>"
.venv/bin/python3 - << EOF
import torch, json
import numpy as np
from pathlib import Path

act_dir = Path("$MODEL_DIR/activation")
our_hist = json.load(open('/tmp/full_histogram.json'))

def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

rhos = []
for layer in range(32):
    pt = torch.load(act_dir / f"activation_{layer}.pt", weights_only=False).numpy().astype(np.float64)
    ours = np.array(our_hist[str(layer)], dtype=np.float64)
    rhos.append(spearman(pt, ours))
print("mean spearman rho:", np.mean(rhos))
EOF
```

Expect mean Spearman rho around 0.85 (range ~0.66-0.94 across layers) —
strong agreement between a 10-document wiki calibration set and
PowerInfer's own, much larger and differently-sourced profiling data. This
is external validation that neuron "hotness" is a fairly stable, corpus-
independent property of the model rather than an artifact of our specific
calibration prompts.

## 6. Histogram shape (power-law vs. exponential check)

```bash
parse_histogram -f dumpbins_ReluLLaMA-7B histogram -o /tmp/full_histogram.json
```

Then, in Python, sort each layer's counts descending, average the
sorted-by-rank curves across all 32 layers, and fit both a power law
(`log(count) ~ a + b*log(rank)`) and an exponential
(`log(count) ~ a + k*rank`) via `numpy.polyfit`; compare R². On our data the
exponential fit (R²≈0.89) clearly beats the power-law fit (R²≈0.72), and
the top ranks are tied at the maximum possible count (some neurons are
active on ~100% of tokens) — a "thick, saturating head" rather than the
smoothly-decreasing head a true Zipf/power-law distribution would show.

## Notes / things that are environment-specific

- Absolute paths above (model snapshot dirs, `/tmp/...`) will differ on
  your machine — `scripts/mac_path.py` / `scripts/husky5_path.py` hold the
  per-machine model path maps `run_sparse_dump_bin.py` reads from.
- `gen-mapping`'s greedy packer is O(n_neurons × n_slots) per layer
  (~11M comparisons/layer at 1024 banks) — a few seconds total, not a
  bottleneck; don't be surprised it's much faster than the simulation
  steps, which re-read every `.bin` record.
- `report` always re-runs the simulator twice from scratch (no caching);
  `simulate` + `to-cycle` do cache their JSON output next to the input
  path if you need faster iteration on downstream analysis only.
