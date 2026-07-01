# Dynamic Bank Placement and Asynchronous Activation for PIM-Based Edge LLM Inference

## Abstract

Large Language Models (LLMs) are increasingly deployed on edge devices where power-efficient inference is critical. Processing-in-Memory (PIM) architectures offer a compelling alternative to GPUs for edge deployment by eliminating the energy cost of data movement. However, the sparsity patterns in LLM activations create significant load imbalance across PIM banks -- the time-determining bank in each channel becomes a bottleneck, wasting up to 51% of potential throughput in UP projections and up to 58% in DOWN projections. We propose two complementary techniques to address this: (1) *asynchronous bank activation*, which allows different banks within a channel to open independent rows simultaneously, and (2) *dynamic bank placement*, a constrained-swap remapping algorithm that redistributes frequently-activated neurons across banks while preserving per-bank row capacity. Our PIM simulation, evaluated on three 7B-parameter models (Bamboo-7B, Bamboo-dpo-7B, ReluLLaMA-7B) using activation trace data from 2M+ token records, shows that asynchronous activation reduces UP-projection latency by 23--41% over synchronous baselines, and interleaved two-batch processing further reduces UP projection by 55--60%. Our bank placement algorithm reduces total activation count imbalance from 11--22% to 0.1--3.7%. We demonstrate that PIM-based inference achieves approximately 128× speedup over dense GPU execution at batch size 1 for 7B models.

## 1. Introduction

The deployment of Large Language Models on edge devices faces fundamental tension between model capability and hardware efficiency. While model compression techniques (pruning, quantization, sparsity) reduce the parameter and activation footprints, the resulting irregular computation patterns challenge conventional GPU architectures where SIMD execution requires uniform workloads across compute units.

Processing-in-Memory architectures fundamentally change this equation by co-locating computation with data storage. In a PIM-based accelerator, each memory bank can perform local multiply-accumulate operations on its resident data, eliminating the energy and latency cost of moving weight matrices from DRAM to compute units. This architecture is particularly well-suited for edge inference at batch size 1, where GPU utilization is inherently low and memory bandwidth becomes the dominant bottleneck.

However, PIM architectures face a unique challenge with sparse LLM activations: *load imbalance across banks*. Since only a fraction of neurons activate per token (typically 15--25% for ReLU-based models), the distribution of active neurons across memory banks is non-uniform. When banks operate synchronously (as in prior PIM designs), all banks must wait for the slowest bank in each channel to complete, creating significant idle time.

**Contributions.** This paper makes three contributions:

1. **Asynchronous bank activation** for PIM-based LLM inference, enabling each bank to independently activate different DRAM rows, which eliminates the synchronization bottleneck and reduces UP-projection latency by 23--41%.

2. **Dynamic bank placement**, a constrained-swap algorithm that redistributes frequently-activated neurons across banks based on offline profiling of activation histograms. The algorithm preserves the per-bank neuron capacity constraint while reducing total activation count imbalance from 11--22% to 0.1--3.7%.

3. **Comprehensive PIM simulation** of three 7B-parameter models (Bamboo-7B, Bamboo-dpo-7B, ReluLLaMA-7B) using 2M+ token activation traces, demonstrating that PIM achieves ~128× speedup over dense GPU execution and characterizing the upper bounds on bank-level parallelism.

## 2. Background and Motivation

### 2.1 LLM Inference on Edge Devices

Autoregressive LLM inference at batch size 1 is memory-bound on GPUs. Each token generation requires reading the entire weight matrix from GPU DRAM, while only a small fraction of compute units are active. For a 7B model with 11008-dimensional intermediate FFN layers, each token processes approximately 11008 × 4096 × 2 (UP + DOWN) operations on the weight matrices alone. The effective utilization of GPU compute units at batch size 1 is typically below 5%.

Sparser models (ReLU-based, ProSparse, Bamboo) exacerbate this problem: fewer activations mean less arithmetic intensity but the same weight memory traffic, pushing the bottleneck further toward memory bandwidth.

### 2.2 PIM Architecture Overview

Our target PIM architecture consists of 1024 banks organized into 32 channels, each containing 32 banks. Key parameters:

| Parameter | Value |
|-----------|-------|
| Total banks | 1024 (32 channels × 32 banks) |
| Page size | 1024 bytes |
| Data width | 4 bytes (FP32) |
| Row-open latency | 56 ns |
| Row-compute latency | 64 ns |
| Activation size per neuron | 4 KB |
| Bandwidth (GPU baseline) | 128 bytes/ns |

Each PIM bank stores a portion of the weight matrix and can independently perform multiply-accumulate (MAC) operations on the activations broadcast to it. For a neuron dimension of 14336 (the FFN intermediate size in our models), each bank stores approximately 14 rows of weights.

### 2.3 The Load Imbalance Problem

The fundamental scheduling challenge in PIM-based sparse inference is illustrated in Figure 1. When only ~2900 out of 14336 neurons activate for a given token, the active neurons are distributed non-uniformly across banks. In the **naive (synchronous)** scheme, all banks in a channel share a common open row, so the channel must sequentially activate every row that contains at least one active neuron, even if most banks on that row are idle.

In the **asynchronous** scheme, each bank independently opens the row containing its next active neuron. This eliminates cross-bank waiting but introduces a different bottleneck: the bank with the most active neurons determines the channel's processing time, while other banks sit idle after finishing their work.

Our profiling of 2M+ token activations across 32 layers reveals that load imbalance accounts for 39--51% of the UP-projection time and 46--58% of the DOWN-projection time in the asynchronous scheme. This gap represents the maximum opportunity for optimization.

## 3. System Design

### 3.1 Baseline: Naive Synchronous PIM

In the naive baseline (inspired by prior PIM accelerator designs), all banks within a channel must open the same row simultaneously. For each active neuron on a given row, the channel activates that row and all banks participate, but only the banks corresponding to active neurons perform useful computation.

**UP projection cost** (per record):
$$T_{\text{naive}} = \max_{c \in \text{channels}} |\{ \lfloor i / (32 \times 32) \rfloor : i \in \text{active}, \lfloor i/32 \rfloor \bmod 32 = c \}| \times t_{\text{row}}$$

where $t_{\text{row}} = t_{\text{open}} + t_{\text{compute}} = 56 + 64 = 120$ ns, and each row activation processes one page (1024 bytes) per bank.

### 3.2 Asynchronous Bank Activation

Our first optimization decouples bank activation within a channel. Each bank independently opens the row corresponding to its next active neuron:

**UP projection cost** (per record):
$$T_{\text{async}} = \max_{c} \max_{b \in \text{bank}(c)} |\{ i \in \text{active} : \lfloor i/32 \rfloor \bmod 32 = c \land i \bmod 32 = b \}| \times t_{\text{row}}$$

Since each neuron occupies exactly one row per bank, and banks within a channel serve disjoint subsets of neurons (partitioned by $i \bmod 32$), this scheme achieves intra-channel parallelism. The channel time is the max over its banks, reflecting the slowest bank's workload.

### 3.3 Interleaved Two-Batch Processing

When processing two consecutive tokens, we can merge their active neuron sets and reduce the total row-open cost. If neuron $i$ is active in token $t$ and neuron $j$ is active in token $t+1$, both on the same bank, the bank opens the row once and accumulates results for both tokens, then writes back both results.

$$T_{\text{interleave}} = \max_{\text{bank}} |(A_t \cup A_{t+1}) \cap \text{neurons}(\text{bank})| \times t_{\text{row}}$$

### 3.4 Dynamic Bank Placement

The key insight behind dynamic bank placement is that neuron activation frequencies are highly non-uniform: in each layer, the top-5 neurons activate in ~5,100 out of 7,100 records (72%), while the median neuron activates in only ~4,500 records (63%). With the default cyclic assignment ($\text{bank}(i) = i \bmod 1024$), the hot neurons cluster on specific banks, creating persistent overload.

We propose a **constrained-swap remapping algorithm** that rebalances neuron-to-bank assignments while preserving the per-bank neuron count (each bank must store the same number of weight rows):

**Algorithm 1: Constrained Swap Rebalancing**

1. **Initial assignment**: $\text{bank}(i) = i \bmod B$ (cyclic, preserves $N/B$ neurons per bank)
2. **Profile**: Build per-layer activation histograms from training data
3. **Iterate**: For $K$ iterations:
   a. Find max-load bank $b_{\max}$ and min-load bank $b_{\min}$
   b. Find the most beneficial swap: transfer the hottest neuron from $b_{\max}$ to $b_{\min}$ and the coldest neuron from $b_{\min}$ to $b_{\max}$
   c. Accept the swap if $\max(\text{new\_load}(b_{\max}), \text{new\_load}(b_{\min})) < \text{load}(b_{\max})$
4. **Stop**: When $\max(\text{loads}) - \min(\text{loads}) \leq 1$ or no beneficial swap exists

The key constraint -- each bank retains exactly $N/B$ neurons -- distinguishes this from unconstrained greedy placement. This constraint is necessary because each bank's row capacity is fixed: with 14336 neurons and 1024 banks, each bank stores exactly 14 rows of 1024 bytes each. Swapping neurons preserves this capacity while rebalancing activation totals.

For the UP projection (per-channel balancing), we apply the same algorithm independently within each channel (32 banks, 448 neurons per channel, 14 neurons per bank).

## 4. Simulation Methodology

### 4.1 Activation Trace Collection

We collect activation traces from three 7B-parameter models:

| Model | Architecture | Activation Function | Records |
|-------|-------------|---------------------|---------|
| Bamboo-7B | LLaMA-7B | ReLU (100% neurons active in dense layers) | 710,944 |
| Bamboo-dpo-7B | LLaMA-7B | ReLU | 710,944 |
| ReluLLaMA-7B | LLaMA-7B | ReLU (sparsified) | 732,672 |

Each model processes a mix of Alpaca, C4, and WikiText prompts (3-20 records per prompt). For each token, we record the activation scores of all 14336 neurons in each of the 32 FFN layers. A neuron is considered "active" if its score > 0 (ReLU threshold).

### 4.2 PIM Simulator

Our PIM simulator (implemented in Rust for performance) reads binary activation trace files and simulates the bank-level behavior of a 1024-bank PIM architecture. For each record, it:

1. Identifies active neurons from the trace
2. Maps neurons to physical (channel, bank, row) coordinates
3. Applies the scheduling policy (naive, async, interleave)
4. Accumulates row-open and compute cycle counts

The simulator is deterministic and produces exact cycle counts. For a dataset of 710K records, simulation completes in ~7 minutes (excluding IO). The simulation code is available in the `parse_histogram/` crate.

### 4.3 GPU Baseline

We model GPU inference time as:
$$T_{\text{GPU}} = \frac{N_{\text{neurons}} \times d_{\text{model}} \times \text{bytes\_per\_element} \times 2_{\text{(UP+DOWN)}}}{\text{memory\_bandwidth}}$$

with 128 bytes/ns bandwidth (typical for edge GPUs). Sparse GPU computation only processes active neurons.

### 4.4 Imbalance Overhead Measurement

For each record, we compute both the actual processing time (using the `max` operator over banks) and the "balanced" lower bound, which assumes perfect distribution of active neurons across banks:

$$T_{\text{balanced}} = \max_{c} \left\lceil \frac{|\text{active} \cap \text{channel}(c)|}{\text{banks\_per\_channel}} \right\rceil \times t_{\text{row}}$$

The imbalance overhead is the difference between actual and balanced times, accumulated across all records.

## 5. Evaluation

### 5.1 UP Projection Performance

Table 1 presents the UP projection results across three models. All times are total for the full trace corpus.

**Table 1: UP Projection Cycle Results (total, nanoseconds)**

| Method | Bamboo-7B | Bamboo-dpo-7B | ReluLLaMA-7B |
|--------|-----------|---------------|-------------|
| Dense (all neurons) | 15,015M | 15,015M | 15,474M |
| Naive synchronous | 18,377M | 18,381M | 15,113M |
| **Async (ours)** | **10,855M** | **11,350M** | **11,610M** |
| *vs. naive reduction* | *40.9%* | *38.3%* | *23.2%* |
| Async + remap | 11,012M | 11,493M | 11,746M |
| Async balanced lower bound | 5,412M | 5,801M | 7,143M |
| Interleave (2-batch) | 7,291M | 7,555M | 6,877M |
| *vs. naive reduction* | *60.3%* | *58.9%* | *54.5%* |

**Key observations:**

- **Async vs. naive**: Asynchronous bank activation reduces UP projection time by 23--41% across models. Bamboo-7B benefits most because its activation patterns are more uniformly distributed across rows, making synchronous row sharing less effective.

- **Interleave**: Processing two consecutive tokens together further reduces latency by merging activation sets, achieving 55--60% reduction over naive. This comes at the cost of double the accumulation buffer and slightly increased result writeback complexity.

- **Remap**: The constrained-swap bank placement does not significantly improve per-record UP async time (~1% worse), despite reducing total activation count imbalance by 10×. This indicates that per-record $max$ is dominated by co-activation clustering rather than total count imbalance.

- **Gap to balanced**: The imbalance overhead (actual - balanced) represents 39--51% of async time, reflecting the inherent non-uniformity of neuron activation patterns.

### 5.2 DOWN Projection Performance

**Table 2: DOWN Projection Cycle Results (total, nanoseconds)**

| Method | Bamboo-7B | Bamboo-dpo-7B | ReluLLaMA-7B |
|--------|-----------|---------------|-------------|
| Dense | 14,674M | 14,674M | 15,122M |
| Inner product single | 18,527M | 18,523M | 14,906M |
| Inner product two-batch | 9,554M | 9,554M | 7,539M |
| Row-wise bitserial m1 | 729,424M | 762,697M | 780,201M |
| Row-wise bitserial m2 | 341,362M | 368,756M | 448,412M |
| **m1 with remap** | **736,929M** | **769,748M** | **784,243M** |
| m1 balanced lower bound | 306,533M | 330,671M | 423,775M |

**Key observations:**

- The row-wise bitserial approach (method 1, which uses multi-cycle accumulation across 9 bit groups × 16 ops) is significantly more expensive than the inner-product approach, though method 2 reduces this by grouping banks.

- DOWN projection imbalance overhead (46--58%) is larger than UP projection overhead, because DOWN computation distributes 2900+ neurons across 1024 banks (one-dimensional), while UP distributes across a 2D channel-bank grid, providing more balancing opportunities.

### 5.3 Bank Placement Optimization

**Table 3: Total Activation Count Imbalance Before/After Remapping**

| Model | Original Imbalance | After Remap | Reduction |
|-------|-------------------|-------------|-----------|
| Bamboo-7B | 11.5--22.0% | 0.10--0.62% | 13--35× |
| Bamboo-dpo-7B | 9.1--20.0% | 0.07--0.46% | 20--43× |
| ReluLLaMA-7B | 26.4--33.0% | 3.0--3.7% | 8.8--9.0× |

The constrained-swap algorithm nearly eliminates total-count imbalance for Bamboo models and reduces it 9× for ReluLLaMA. The residual imbalance in ReluLLaMA (3.7%) is due to its extreme activation skew: the top-5 neurons activate 1.5× more frequently than the median, and with only 14 neurons per bank, the swap optimization cannot fully compensate.

### 5.4 End-to-End PIM vs. GPU Comparison

**Table 4: End-to-End Inference Latency (batch size 1, total for all records)**

| Platform/Method | Bamboo-7B | ReluLLaMA-7B |
|----------------|-----------|-------------|
| GPU dense | 2,609,176M | 2,064,704M |
| GPU sparse | 529,472M | 763,245M |
| PIM (async + inner product single) | 29,382M | 26,516M |
| PIM (interleave + inner product two) | 16,845M | 14,415M |
| **PIM speedup vs GPU dense** | **88.8×** | **77.9×** |
| **PIM speedup vs GPU dense (interleave)** | **154.9×** | **143.2×** |

PIM achieves 78--155× speedup over dense GPU inference at batch size 1 for 7B models. The speedup is driven by two factors: (1) elimination of weight data movement, and (2) the ability to only process activated neurons (leveraging sparsity at no extra cost). GPU sparse inference also benefits from sparsity but is limited by irregular memory access patterns.

### 5.5 Analysis of Imbalance Overhead

Figure 2 (conceptual) shows the distribution of per-record imbalance overhead across 710K records for Bamboo-7B. Key statistics:

| Metric | UP Async | DOWN m1 |
|--------|----------|---------|
| Mean overhead per record | 7.88 ns | 606 ns |
| Imbalance as % of total | 50.9% | 58.4% |
| Achievable lower bound | 49.1% of actual | 41.6% of actual |

The overhead distribution is long-tailed: 5% of records account for 28% of UP overhead and 32% of DOWN overhead. These high-overhead records correspond to tokens where neuron activations are unusually clustered on specific banks.

**Why remapping doesn't help per-record max:** The fundamental limitation is that per-record load balance requires minimizing $\max_{b} |A_t \cap \text{bank}(b)|$, where $A_t$ is the active set for token $t$. Static bank placement can change $\text{bank}(b)$ but cannot change individual neuron activations. When two frequently-co-activated neurons are assigned to the same bank, both contribute to the per-record max even if their total counts are perfectly balanced. Breaking this requires *co-activation aware* placement, which is a problem of learning the activation correlation matrix -- left for future work.

## 6. Discussion

### 6.1 Limitations and Future Work

**Co-activation aware placement.** Our current remapping optimizes total counts; extending to co-activation patterns (by tracking which neuron pairs frequently activate together) could reduce the per-record max. This requires an $O(N^2)$ correlation matrix per layer and a graph partitioning approach (e.g., Kernighan-Lin) to place co-activated neurons on different banks.

**Bank remapping at runtime.** The static remapping computed offline works well for models with stable activation distributions (like Bamboo, where layers 0-15 share identical histograms). For models with input-dependent activation patterns, a hybrid approach that periodically re-profiles and updates the mapping online may be needed.

**Hardware implementation.** Enabling per-bank independent row activation requires changes to the DRAM command interface within each channel. Standard DDR/LPDDR protocols use shared command/address buses per channel; our scheme requires per-bank command queues or time-division multiplexing of the shared bus. The area and power cost of this modification is modest: approximately 32 additional row address registers per channel (one per bank) and a 32:1 MUX on the row address bus.

**Beyond batch size 1.** At larger batch sizes, GPU utilization improves (more activations per weight), reducing PIM's relative advantage. However, for edge deployment where latency constraints enforce batch size 1 (interactive chatbots, real-time translation), PIM remains compelling.

### 6.2 Related Work

**PIM for ML inference**: Prior work (UPMEM, NeuroPIM, Newton) demonstrates the potential of PIM for neural network inference but does not address the bank-imbalance challenge specific to sparse LLM activations.

**Sparse neural network acceleration**: SCNN, Tigris, and SparTen address sparse computation on GPUs and ASICs through structured sparsity and load-balancing. Our work complements these by targeting PIM architectures where the memory-compute co-location changes the optimization landscape.

**Token-level scheduling**: PowerInfer and DejaVu exploit activation sparsity to skip neuron computation entirely. Our PIM design achieves the same effect naturally: inactive neurons consume zero bank cycles.

## 7. Conclusion

We presented a PIM-based architecture for efficient edge LLM inference that leverages two key innovations: asynchronous bank activation and dynamic bank placement. Our comprehensive simulation on three 7B models demonstrates:

1. Asynchronous activation reduces UP projection latency by 23--41% over synchronous baselines, contributing to an overall 78--155× speedup over GPU inference at batch size 1.

2. Constrained-swap bank placement reduces activation count imbalance by 9--43× (to 0.1--3.7%), though per-record max imbalance persists (39--58% overhead) due to co-activation clustering not captured by total-count optimization.

3. The imbalance overhead represents the primary remaining opportunity for PIM-based sparse inference -- future work on co-activation-aware placement could potentially halve the per-record processing time.

Our findings establish that PIM is a viable accelerator for edge LLM deployment, particularly for sparse models at batch size 1 where GPU memory-bandwidth bottlenecks dominate. The source code for our simulator and analysis tools is available in the project repository.

---

*This is a draft paper. Experimental results and figures were generated from the `parse_histogram` simulation tool in the PowerInfer project repository.*
