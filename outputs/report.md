# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

## Extensions ("Your Turn")

### 1. Purchasing policy v2 (GPU-aware interrupt rate + 1yr/3yr reserved)

- v1 monthly: $15,627 -> v2 monthly: $15,905 (delta $-277, -1.8%)
  - `job-infer-chat`: reserved ($2,592) -> reserved_3yr ($2,592) — GPU-specific risk/duration check changed the pick
  - `job-infer-rag`: reserved ($2,160) -> reserved_3yr ($2,160) — GPU-specific risk/duration check changed the pick
  - `job-infer-search`: reserved ($972) -> reserved_3yr ($972) — GPU-specific risk/duration check changed the pick
  - `job-dev-sandbox`: spot ($203) -> on_demand ($480) — GPU-specific risk/duration check changed the pick

### 2. Right-sizing memory-bound GPUs by MBU/bandwidth

| GPU | current | $/GB-VRAM | suggested | $/GB-VRAM | BW kept | $/month saved |
|---|---|---|---|---|---|---|
| gpu-h100-0 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |
| gpu-h100-1 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |
| gpu-h100-2 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |
| gpu-h100-3 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |
| gpu-h100-4 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |
| gpu-h100-5 | H100 | 0.0312 | MI300X | 0.0102 | 158.2% | $396 |

Total if all memory-bound GPUs are right-sized: **$2,376/month**

### 3. `cache_is_worth_it()` — is the cache write premium repaid?

- **small**: avg cache reads/prefix = 237.75, break-even = 1.39 reads -> **worth it**
- **large**: avg cache reads/prefix = 62.25, break-even = 1.39 reads -> **worth it**

### 4. Reasoning traffic budget

- Reasoning: 201 requests (8.4% of traffic), $1.3965/day (16.5% of $ spend), 29787.74 Wh/day
- Normal: 2199 requests, $7.0882/day, 1887.56 Wh/day
- Capping reasoning to 5% of traffic would save **$0.5628/day** and **12004.02 Wh/day** (40% of reasoning's energy draw)
- Note the asymmetry: reasoning is a much bigger energy problem than a $ problem under today's per-token pricing — routing rules justified on cost alone will under-value it.

### 5. Carbon-aware scheduling for interruptible jobs

| Job | Wh | gCO2 us-east-1 | gCO2 best region | saved |
|---|---|---|---|---|
| job-train-llm | 1,568,000 | 595,840 | 47,040 (europe-north1) | 92.1% |
| job-train-embed | 80,000 | 30,400 | 2,400 (europe-north1) | 92.1% |
| job-finetune | 25,200 | 9,576 | 756 (europe-north1) | 92.1% |
| job-dev-sandbox | 52,800 | 20,064 | 1,584 (europe-north1) | 92.1% |
| job-batch-eval | 63,000 | 23,940 | 1,890 (europe-north1) | 92.1% |

Total carbon saved by moving all interruptible jobs to the cleanest region: **626.1 kg CO2e** over their run window.
