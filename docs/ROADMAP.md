# Roadmap: FPGA KV-Cache Policy Cost Study

*Working plan, 30 September 2026. Dates are relative and must be rescheduled once the actual deadline and board are known. Project definition: `PROJECT_OVERVIEW.md`; paper audit: `LITERATURE_REVIEW.md`.*

## 1. Research objective

Answer this question with a common, verified implementation:

> At equal retained-cache capacity, what quality benefit does H2O-style importance retention provide over sink-plus-window, and what extra update, selector, address-generation, memory-traffic, cycle, and FPGA-resource cost does it impose?

The project studies existing retention policies. It does not propose a new policy. The proposed research contribution is the controlled hardware cost/quality characterization. The novelty gate remains open because VEDA, HiKV, and related IIT work are close prior art.

## 2. Fixed experimental scope

### Core policies

| ID | Behavior | Role |
|---|---|---|
| FULL | Retain all tokens | Reference quality and latency |
| SW | Retain first `n_sink` tokens and most recent tokens | Regular-addressing bounded baseline |
| H2O | Retain high accumulated attention-score tokens plus protected recent window | Dynamic-score policy |

Use FP16 K/V for every core run to isolate policy-management cost. A quantized cache is a possible future extension after core evaluation, not a parallel workstream.

### Frozen controls

- Same model-derived Q/K/V inputs, attention implementation, bus width, memory tier, clock target, and timing boundary across policy modes.
- Compare SW and H2O at the same cache capacity, including position/valid bits and H2O score metadata in total storage accounting.
- Report at least three board-feasible sequence lengths and two cache budgets after board selection.
- One head first. Add GQA only after one-head integration is correct and timing/resources allow it.
- Freeze the H2O variant: score definition/aggregation, recent window, eviction timing, tie-breaking, and slot reuse. Mirror the exact variant in Python and RTL.

### Two evaluation tracks

1. **Hardware track:** replay Q/K/V/attention traces through the same one-head FPGA attention/cache path. Measures cycles, transactions, policy state, and PPA.
2. **Quality track:** software autoregressive decoding with the same policies on a small open model. Measures perplexity and a long-context retrieval task if feasible. This establishes whether local trace similarity reflects actual behavior. Never label teacher-forced local error as end-to-end model quality.

## 3. Implementation specification

### Common datapath

The attention engine consumes one query and iterates over valid cache slots, computes QK scores, softmax and weighted V output. The K/V read interface remains identical for FULL, SW, and H2O. The initial design uses one head and bounded context lengths.

### Policy manager

- **SW:** append to circular recent buffer; pin configured sink slots; advance the oldest recent slot on overflow.
- **H2O:** maintain a score per retained token; update scores from current attention weights; protect the configured recent region; select the lowest-scoring eligible token; reuse its slot or use a free slot if available. Specify when score values are renormalized/reset to avoid overflow.
- Keep physical storage slot IDs separate from logical token age. Attention is invariant to the iteration order if score/value associations and positional treatment are correct; verify this property.
- Add counters for policy update cycles, selector cycles, K/V reads/writes, score-state reads/writes, stalls, burst length, and total query cycles.

Important design question: H2O’s score update may require many score writes per query, and victim selection may need a reduction tree or maintained priority structure. Compare the actual chosen architecture, not only asymptotic complexity. A controller that requires an extra full K/V compaction pass must charge that traffic and latency.

## 4. Metrics and protocol

### Quality

- Attention output vs FULL: mean cosine, mean relative L2, max absolute error.
- Autoregressive perplexity delta on held-out text for the small model.
- Retrieval accuracy on a named long-context dataset/task if model/tool access permits.
- Quality by sequence length and cache budget, not one averaged point only.

### Hardware

- Policy-only cycles and whole attention-step cycles.
- K/V bytes and score/metadata bytes transferred at the actual interface; transaction count; average burst utilization; sequential vs irregular accesses.
- LUT/FF/BRAM/URAM/DSP; achieved Fmax, slack; board power/energy if directly measurable.
- Validate cycle and traffic counters on hand-computable short traces and against bus/logic-analyzer transactions.
- Repeat each hardware point at least five times where run variation exists; publish median and spread plus raw logs.

### Fairness

- Identical cache capacity and retained-token budget; report all overhead bytes.
- Same hardware and timing boundary. Report initialization and host/DMA costs separately from steady-state query time.
- Full cache is a quality reference, not an equal-budget competitor.
- If external DDR is not in the implemented path, make no DDR bandwidth or external-memory conclusion.

## 5. Phases and deliverables

### Phase 0 — Feasibility and literature gate (Week 1)

1. Fill in `docs/platform.md`: exact FPGA board, memory type/capacity, usable bus, tools, constraints, host flow, and sustainable-bandwidth measurement method.
2. Confirm deadline, available hours/week, and supervisor requirements.
3. Read VEDA, HiKV, IIT Jodhpur's graph-based compression paper, H2O, StreamingLLM, and PagedAttention in full. Use `LITERATURE_REVIEW.md`; correct unresolved/invalid references in `docs/Papers` before citing.
4. Write a claim-to-prior-work table. State what this project measures that the closest hardware papers do not.

**Exit:** accessible board path and supervisor-approved scope. If not available, decide explicitly whether a synthesis/simulation project is acceptable; otherwise change project topic before RTL.

### Phase 1 — Software reference and quality baseline (Weeks 2–4)

1. Keep existing `kvgold.py` and refactor/extend only as needed for exact SW/H2O retention semantics.
2. Add deterministic policy tests for fill, overflow, sink protection, recent protection, tie-breaking, score updates, and slot reuse.
3. Preserve and document model/tokenizer/dataset/version and input token IDs for trace creation.
4. Run FULL/SW/H2O software comparisons for at least two budgets and the longest software-feasible sequence.
5. Implement autoregressive small-model evaluation with the cache policy active during generation. Report perplexity and retrieval result if feasible.

**Deliverables:** frozen policy spec, software regression, `results/policy_quality.csv`, input provenance. **Exit:** independent reruns reproduce outputs; chosen settings give a meaningful quality-vs-budget comparison.

### Phase 2 — Cycle and storage model (Weeks 3–5)

1. Derive exact bytes and state size per policy, including score memory, valid/age maps, padding, and physical slot layout.
2. Develop a cycle model for SW and H2O controller choices (parallel score update, serial/parallel min-select, slot write, memory stalls).
3. Benchmark actual sustainable board-memory bandwidth and minimum burst behavior.
4. Predict policy-only and full-query cycles; identify expected bottleneck and choose one H2O selection architecture.

**Deliverables:** `experiments/policy_cost_model.py`, `results/policy_cost_model.csv`, architecture sketch. **Exit:** predicted byte/cycle accounting agrees with hand-counted short cases and board constraints.

### Phase 3 — RTL policy unit and verification (Weeks 5–9)

1. Build a board smoke design with clock/reset, control, memory transaction, and counter readback by Week 6 if practical.
2. Implement common slot table/interface and SW ring buffer.
3. Implement H2O score state and selector.
4. Verify policy decisions against software golden model bit-for-bit.
5. Include assertions for valid/ready stability, slot uniqueness, sink/recent protection, correct overwrite, counter consistency, and no stale score association.
6. Synthesize both policies under the same constraints and retain reports.

**Deliverables:** synthesizable controller RTL, testbench, regression logs, synthesis reports. **Exit:** all directed/random policy tests pass and resource requirements fit selected board.

### Phase 4 — Integrate attention and memory (Weeks 8–12)

1. Integrate one-head attention read/score/softmax/value path and policy manager.
2. Compare RTL outputs with fixed-point/software reference on short vectors and trace slices.
3. Connect to external DDR if available; validate transactions, addresses, bursts, and data.
4. Confirm counter counts from manually verified runs.

**Deliverables:** integrated bitstream or documented synthesis/simulation fallback; bit-accurate vector results. **Exit:** outputs and memory behavior verified before performance collection.

### Phase 5 — Controlled measurements (Weeks 12–16)

Run FULL, SW, H2O with identical input, sequence lengths, cache budgets, timing boundary and clock. Capture all metrics above, repeated trials, raw logs, FPGA/tool metadata, and failed runs. Make policy-only cost and total decode cost separate plots. Explain any mismatch between the cycle model and board data.

**Deliverables:** `results/hardware_measurements.csv`, raw logs, scripts and figures. **Exit:** conclusions are traceable to reproducible data and uncertainty is reported.

### Phase 6 — Claims, writing, and decision (Weeks 17–20 assumed)

1. Complete the related-work comparison; novelty statement must survive supervisor review.
2. Create a claim/evidence/limitation table.
3. Ask another student to reproduce one software and one hardware result from instructions.
4. Decide with supervisor whether the outcome supports a conference/workshop paper or a thesis/report. Do not inflate a platform-specific study into a general claim.

## 6. Stop/go rules

- No named board or memory path: no external-bandwidth claims; continue software and literature only.
- No external DDR: report on-chip-memory controller behavior and synthesis; do not present it as DDR measurements.
- H2O score-update/selector cannot meet timing or area: keep this as a measured limitation, reduce budget/head scope, or report only policy-controller synthesis if scope approved.
- No reliable autoregressive evaluator: restrict claims to attention-output fidelity and explicitly say model-level quality is unmeasured.
- VEDA/HiKV/IITJ fully cover the proposed exact question: withdraw the paper novelty claim and refine the question before investing in full RTL.
- Under schedule pressure, drop multi-head/GQA, quantization, extra policies and broad sweeps before dropping verification or fairness controls.

## 7. Required result tree

```text
docs/       PROJECT_OVERVIEW.md, ROADMAP.md, LITERATURE_REVIEW.md,
            platform.md, policy_spec.md, decisions.md
experiments/ policy_cost_model.py, quality evaluator, export scripts
rtl/         cache slots, SW manager, H2O manager, attention path, counters
tb/          unit/integration tests, vectors, regression logs
fpga/        constraints, build files, host and memory interface
results/     raw logs, CSVs, run metadata, plots
```

Every result records source commit, model/trace/token IDs, policy settings, budget, board, tools, bitstream, and exact command.

