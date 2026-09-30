# Project Overview: Hardware Cost of KV-Cache Retention Policies

*Proposed revised scope, 30 September 2026. The FPGA platform, deadline, and supervisor approval still need to be confirmed. See `LITERATURE_REVIEW.md` and `ROADMAP.md`.*

## Project in one sentence

Implement a small, verified FPGA KV-cache manager and compare simple and importance-based retention policies under matched cache budgets, measuring both the quality they preserve and the hardware cost of maintaining and accessing the retained state.

## Why this question

KV-cache research commonly changes software policy, compression, attention, or serving behavior to work within hardware constraints. That is a valid hardware-software approach. The hardware project is sensible only if it measures what those policy choices cost in a real implementation. “We put a known policy on an FPGA” is not enough by itself.

Recent work already includes dedicated hardware for policy/selection: VEDA proposes voting-based eviction and a dataflow-flexible accelerator; HiKV combines token eviction with finer-grained selection and a hardware importance sorter. An IIT Jodhpur team including Prof. Binod Kumar has also published a graph-based dynamic KV-compression method. These make a new eviction algorithm or a generic eviction accelerator an unsafe novelty claim. The proposed contribution is narrower: a controlled comparison of policy-manager overhead and memory organization on a named, resource-constrained FPGA, with quality and hardware cost reported together. See the literature review for the required novelty caveat.

## Research question

> At the same retained-cache budget and attention datapath, how much model quality does a score-based policy recover over deterministic retention, and what additional controller cycles, memory traffic, address irregularity, and FPGA resources does it require?

**Candidate title:** *From Retention Policy to Memory Traffic: An FPGA Study of KV-Cache Management Costs.*

This is a candidate for a defensible B.Tech research project, not a guaranteed publication claim. The exact publication gap must be confirmed with the supervisor after reading VEDA, HiKV, and the IIT Jodhpur work in full.

## What will be built

One common attention/cache interface with three policy modes:

| Mode | Policy | Hardware behavior to study |
|---|---|---|
| Reference | Full cache | Quality and latency reference; no eviction |
| Static | Sink plus sliding window (StreamingLLM) | Fixed sink slots and circular recent-token buffer |
| Dynamic | H2O-style heavy-hitter plus recent window | Per-token importance state, score updates, victim selection, and slot reuse |

All modes use the same FP16 K/V representation in the core study. This deliberately removes quantization as a confound. The same attention datapath, clock target, bus, cache budget, input trace, and measurement boundaries are used for both bounded policies. Quantization is out of scope unless there is time after the policy comparison is complete.

The hardware scope is a **single attention head initially**, with one-query decode steps and bounded sequence lengths. The RTL includes cache slots, valid/position metadata, append/evict control, the policy score state and selector for H2O, a simple attention read path, and counters. It does not include the transformer layers, weight matrices, or a complete LLM accelerator. If board memory supports it, cache data resides in external DDR; otherwise results must be labelled as on-chip-memory controller results.

## What will be measured

### Quality

- FP reference vs policy outputs: cosine similarity, relative L2, max absolute error.
- Autoregressive model-level quality on a small open model where feasible: perplexity change and at least one long-context retrieval task/benchmark. Teacher-forced trace error alone is not a model-quality result.
- Compare bounded policies at equal retained-token count and report the actual metadata-inclusive footprint.

### Hardware and memory

- Policy decision cycles and total attention cycles per decode step.
- External-memory bytes read/written, burst count, burst utilization, and non-sequential accesses. Validate counters against bus transactions.
- Extra policy state and writes: importance-score storage/update traffic, selector latency, and cache-manager stalls.
- LUT, FF, BRAM/URAM, DSP, achieved Fmax/timing slack. Power/energy only if the board provides a defensible measurement method.
- Repeat measurements and report spread; retain raw logs and build/config metadata.

### Primary comparison

The central comparison is **sink+window vs H2O at equal cache capacity**. Full cache anchors quality and latency. The key result is a quality-versus-hardware-cost curve, not a claim that one method wins every metric. Report separately (1) policy-controller overhead and (2) total decode-step effect; do not hide the controller behind an attention-only timing boundary.

## Hypotheses to test (not assumed results)

1. Sink+window has low state and regular append/evict addressing.
2. H2O may preserve quality better at a fixed token budget, but maintaining scores and selecting victims adds state, writes, and control work.
3. Under dense attention over the same number of retained tokens, both bounded policies read approximately the same K/V volume; their hardware difference may come mostly from update and addressing overhead, not fewer read bytes. Measure this rather than claiming otherwise.
4. H2O is worthwhile only in regions where its quality benefit exceeds its controller and memory-system cost.

## Out of scope

- New eviction/quantization algorithms, reproducing VEDA/HiKV, DeepSeek V4 system reproduction, multi-request serving, multi-GPU scheduling, 1M-token execution, full-model FPGA inference, and a general-purpose cache framework.
- Quantization in the primary experiments; it would confound policy costs with unpack/dequantization costs.
- Claims of state of the art or that policy comparisons alone are novel.

## Current repository status

The repository currently contains `kvgold.py`, `sweep.py`, `dump_kv.py`, and one saved Q/K/V trace. Its Python model includes an H2O-like score-eviction simulation and StreamingLLM-like sink/window simulation. This provides a starting point, not hardware evidence. There is currently no RTL, board specification, fixed-point policy model, board measurement, or model-level quality evaluation in the repository. Existing trace checks are teacher-forced attention checks.

## Feasibility gates

1. **Platform gate:** name the board, external-memory interface, tools, deadline, and lab access. Without an accessible board/memory path, make the deliverable RTL synthesis/controlled simulation and remove claims of measured FPGA memory-system behavior.
2. **Policy correctness gate:** specify the exact H2O variant, score aggregation, recent-token protection, tie-breaking, and update/eviction order; match software and RTL bit-for-bit for policy decisions.
3. **Quality gate:** verify that the chosen small model and evaluation fit the schedule and hardware/software access.
4. **Novelty gate:** compare against VEDA, HiKV, and the IIT Jodhpur work. If the same policy-manager/control/dataflow question is already answered, the project remains a thesis implementation study but not a novel-paper claim until the research question changes.

## Immediate student/supervisor decisions

1. Confirm board + external memory + toolchain, deadline, and project expectations.
2. Confirm whether an autoregressive small-model quality evaluation is required/feasible.
3. Agree that FP16-only policies are the core, with quantization deferred.
4. Read the closest hardware papers and agree on a one-sentence novelty statement before RTL.

