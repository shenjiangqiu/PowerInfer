/// Fixed hardware parameters for PIM simulation.
#[derive(Debug, Clone)]
pub struct PimConfig {
    pub page_size: u64,
    pub banks: u64,
    pub channels: u64,
    pub banks_per_channel: u64,
    pub data_width: u64,
    /// Weight element width in BITS (32=f32, 16=f16, 8=int8, 6=int6,
    /// 4=int4). The layout simulation derives everything from this —
    /// neurons per 16-byte chunk = floor(128/bits) (sub-byte widths pack
    /// with padding), neurons per 1KB row = 64 chunks × that, physical
    /// rows per neuron = ceil(4096·bits / 8·1024) — allowing sub-byte
    /// precisions that `data_width` (whole bytes) cannot express.
    pub weight_bits: u64,
    pub activation_size: u64,
    pub neuron_size: u64,
    /// Aggregate memory bandwidth of the GPU used as the non-PIM reference,
    /// in GB/s (== bytes/ns). A GPU's DRAM banks (RTX 4090: 384 = 12 GDDR6X
    /// dies × 2 channels × 16 banks) all feed one shared bus, so unlike the
    /// PIM array — where each bank computes locally and bank count sets the
    /// aggregate throughput — the only number that matters for the GPU is
    /// this bus bandwidth. The GPU model is `time = weight bytes streamed /
    /// bandwidth`, deliberately optimistic for the GPU (peak bandwidth,
    /// perfectly coalesced sparse gathers, zero row-activation penalty):
    /// each 7B FFN layer's weights (~360 MB at f32) far exceed the 4090's
    /// 72 MB L2, so every decoded token must re-stream weights from DRAM.
    pub gpu_bandwidth_gbps: f64,
}

impl Default for PimConfig {
    fn default() -> Self {
        Self {
            page_size: 1024,
            banks: 32 * 32, // 1024 banks = 32 channels × 32 banks each
            channels: 32,
            banks_per_channel: 32,
            data_width: 4,             // f32 (bytes; legacy, used by pim::simulate)
            weight_bits: 32,           // f32 (bits; used by pim::layout)
            activation_size: 4 * 1024, // 4K
            neuron_size: 11008,
            gpu_bandwidth_gbps: 1008.0, // RTX 4090: 384-bit GDDR6X @ 21 Gbps
        }
    }
}
