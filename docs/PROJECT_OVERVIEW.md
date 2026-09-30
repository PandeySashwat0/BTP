# KV-Cache Compression on FPGA: Project Overview for Collaborators

*Last updated: 2026-09-30. Items marked **[TBD]** are not decided yet and are listed in Section 12.*

**Read this first.** This document explains what we are building, why, how we will do it, and how you can help. You do not need to know anything about large language models (LLMs) to start. Section 2 explains the background from scratch. A more technical companion, `ROADMAP.md`, holds the detailed math and design notes; you only need it if you work on the golden models or RTL.

---

## 1. The project in one minute

Modern chatbots (LLMs) generate text one word-piece ("token") at a time. To do that they keep a growing memory of everything said so far, called the **KV cache**. For long conversations this memory becomes huge, and reading it again for every new token is the main thing that slows generation down.

**One way to shrink it is to store the numbers with fewer bits ("quantization")**, for example 4 bits instead of 16. That saves memory, but on real hardware some of the saving is lost, because the hardware must unpack and convert those small numbers back before using them.

**We are building a small piece of custom hardware on an FPGA that reads the compressed cache and does the attention computation, and we are measuring how much of the theoretical saving we actually keep.** We will build three versions of the same hardware and compare them:

| Version | What it does |
|---|---|
| **A: Baseline** | Stores the cache uncompressed. This is our reference. |
| **B: Naive compressed** | Stores it compressed, converts every number back to full precision, then computes. |
| **C: Fused** | Stores it compressed and computes directly on the compressed numbers, applying the conversion cleverly (Section 4.3). |

The result of the project is a set of measured numbers and plots: how many bytes move, how many clock cycles it takes, how much FPGA area and power it uses, and how accurate the output stays.

---

## 2. Background from scratch

### 2.1 What is the KV cache?

An LLM produces text token by token. For each new token it "looks back" at all previous tokens (this is called **attention**). For every past token the model stores two vectors, a **Key (K)** and a **Value (V)**. Think of them as index cards: the Key says "what this token is about" and the Value says "what it contributes".

Without saving these cards, the model would recompute them for every previous token at every step, which is wasteful. So it stores them: that store is the **KV cache**. Every new token adds two more cards, and every new token must read *all* the previous cards.

### 2.2 Why is it a problem?

- The cache grows with the length of the conversation. For a typical 8-billion-parameter model, each token adds about 128 KiB, so **a 128,000-token context needs about 16 GiB** just for the cache.
- Producing each new token requires reading the entire cache once, but does little arithmetic on each number. The computation is limited by **memory bandwidth** (how fast data can be read), not by the number of calculators. This is called being **memory-bound**.

So making the cache smaller directly helps speed and energy. This is why so much current research (KIVI, BitDecoding, Omni-LUT and others, see Section 10) targets it.

### 2.3 Quantization in plain terms

Normally each number takes 16 bits. **Quantization** stores each number with fewer bits, say 4 or 2, by mapping it onto a small set of levels. To map back we need two extra numbers per group of values: a **scale** and a **zero-point** ("metadata"). To recover a number:

> real value ≈ scale × stored code + zero-point

Because of that metadata, the saving is **less** than the bit count suggests. Our own measurements (head dimension 64) give:

| Format | Bytes per token per head | Real compression |
|---|---|---|
| FP16 (uncompressed) | 256 | 1.00× |
| INT8 | 136 | 1.88× |
| INT4, groups of 32 | 80 | 3.20× |
| KIVI-style mixed 4-bit keys / 2-bit values | 64 | 4.00× |
| KIVI-style 2-bit | 48 | 5.33× |

So "4-bit" gives about 3.2×, not 4×. Whenever someone quotes a theoretical speedup in this project, it must include metadata.

### 2.4 KIVI, the algorithm we build on

KIVI is a published method that quantizes Keys and Values *differently*, because they behave differently:

- **Keys** are quantized **per channel** (each of the 64 or 128 feature positions gets its own scale, shared across a block of consecutive tokens). Keys have a few "outlier" channels with very large values, and per-channel scaling handles them well.
- **Values** are quantized **per token** (each token's value vector gets its own scales).
- The most recent few tokens are kept in full precision in a small **residual buffer**, and are quantized in a batch once enough have accumulated.

### 2.5 Why hardware, not just software?

On GPUs, the extra work of unpacking and converting the small numbers can eat much of the gain. A recent paper, BitDecoding (HPCA 2026), studies exactly this on GPUs. It reports that naive low-bit implementations struggle to deliver the expected speedup, and it presents a solution for GPUs. We are asking the same question for a **custom datapath on an FPGA**, which is a very different platform where we control every part of the pipeline and can measure every clock cycle.

---

## 3. Why this project is still worthwhile

The field is active and there are strong existing systems, so we want to be honest about what we are and are not doing.

**We are not trying to beat GPUs or published accelerators.** An FPGA will not match datacenter throughput, and we will test at the level of one attention step on small models, not full chatbots.

**What we can contribute, and why it is still valuable:**

1. **A measured, reproducible study on real hardware.** Most published work reports GPU results or simulated accelerators. A carefully verified FPGA design with honest measurements of where the time goes is a useful, checkable result.
2. **A designed-in comparison.** Baseline, naive, and fused versions built with the same interfaces, same memory system, and same clock target give a fair answer to "how much of the theoretical saving survives?" That fairness is often missing in reports.
3. **A parameterized design.** Compression formats keep changing (4-bit, 2-bit, FP4, new methods). A datapath whose bit-width, group size, and layout are parameters remains useful as formats change.
4. **Edge relevance.** Devices with limited memory bandwidth are where KV compression matters most.
5. **Skills and verifiable engineering.** The project covers memory-bound datapath design, verification against a golden model, timing closure, and power/area reporting.

**Limits we accept.** The novelty is modest, and results may show the design is simply limited by memory bandwidth. That is a valid outcome, and we will report it either way. We will not claim state-of-the-art results.

---

## 4. What exactly we are doing

### 4.1 Scope

**In scope**
- One quantization scheme (KIVI-style) on one model family, in one attention head first, then several heads in parallel.
- Three datapaths (A, B, C) in synthesizable RTL, verified against a software model, running on a real FPGA board.
- Measured metrics: DDR bytes moved, cycles, area, power, Fmax, and accuracy.
- Eviction of old tokens (dropping unimportant cache entries) as a later extension.

**Out of scope**
- Inventing a new quantization algorithm.
- Building a full LLM accelerator, or running a whole model end to end on the FPGA.
- Supporting many models, or 128K-token runs on the board.
- Reproducing several papers.

### 4.2 The three datapaths

All three share the same interface and memory system so the comparison is fair. For each new token, the hardware reads the cached keys, computes scores against the current query, applies softmax, and combines the values.

- **A (baseline):** everything in FP16, no compression.
- **B (naive):** read packed low-bit data → **unpack** → **dequantize** to real numbers → compute.
- **C (fused):** read packed low-bit data → compute directly on the raw codes, applying scale and zero-point in a smarter place.

### 4.3 The key idea behind version C, explained simply

Dequantizing every stored number, then multiplying, is wasteful, because the scale and zero-point are the same for a whole group. Algebra lets us pull them outside the sum. A tiny example:

- Query q = (1, 2). Stored codes c = (3, 1). Scale s = 0.5. Zero-point z = 1.
- **Naive:** dequantize k = (0.5·3 + 1, 0.5·1 + 1) = (2.5, 1.5), then q·k = 2.5 + 3 = **5.5**.
- **Folded:** s·(q·c) + z·(q₁ + q₂) = 0.5·(3 + 2) + 1·3 = 2.5 + 3 = **5.5**.

Same answer, but the folded version multiplies the query by *raw small integers* and applies the scale and zero-point once per group. We checked this numerically on our golden model, and it matches to within rounding error.

**Caveats:** we verified this only in floating point. In real hardware, precision limits may cost accuracy, or the saving may vanish, and that is part of what we are testing. We also have not yet completed a literature search to see how far related work (for example, BitDecoding's query transformation) already covers this idea, so we do not claim it is new.

### 4.4 A trade-off we expect to see

Compression reduces bytes, so the memory system delivers each token faster, which means the calculators must work faster to keep up. For example, on a memory port delivering 16 bytes per clock, an uncompressed token (256 bytes) takes 16 cycles to arrive, while an INT4 token (80 bytes) takes 5. The compute units then need roughly three times as many multiply-accumulates per cycle to stay ahead. **The design question is whether the datapath stays memory-bound or becomes compute-bound.** That is what our cycle counters will show. (Numbers here are illustrative; real ones depend on the board.)

---

## 5. What already exists

We have a working software pipeline (folder `kv_step1/`):

| File | Purpose |
|---|---|
| `dump_kv.py` | Runs a small language model (Qwen2.5-0.5B) on 2048 tokens of WikiText-2 and saves the real Q, K, V tensors. |
| `kvgold.py` | The **golden model**: a trusted software implementation of quantization, packing, attention, and eviction. The hardware must eventually match it. |
| `sweep.py` | Runs experiments (`check`, `quant`, `evict`) and reports storage and accuracy. |

**Status**
- ✅ Trace validated: recomputed attention matches the model's own output to about 1e-5.
- ⏳ Quantization and eviction result tables: to be shared and reviewed.
- ⏳ Everything else in Section 6.

**Important limitation.** Our current accuracy numbers measure the error introduced at each layer using saved data. They understate how errors can accumulate through a real model. We plan a perplexity evaluation to complement them.

---

## 6. The plan

The schedule assumes about **20 weeks [TBD: confirm the real deadline]**. Weeks are relative to project start.

| Phase | Weeks | What happens | Output |
|---|---|---|---|
| **0. Trust the baseline** | 1 | Run all sweeps; add a storage model for Qwen and Llama-3-8B shapes; diagram how KIVI works; read key papers | Storage table, diagrams, literature notes |
| **1. Cost model** | 2 | Count operations per token for A/B/C; model memory vs compute limits for our board | `hardware_cost.csv`, roofline plot |
| **2. Golden models** | 3–4 | Add the folded computation; build a fixed-point model (the exact integer arithmetic the hardware will use); decide precision; export test vectors | Fixed-point model, precision report |
| **3. RTL** | 5–10 | Write and verify hardware: baseline first, then unpack + naive, then fused. Early board smoke test around week 6 | Verified RTL for A, B, C |
| **4. Board** | 11–14 | Get designs running on the FPGA; read hardware counters | Measured cycles and DDR bytes |
| **5. Sweeps and PPA** | 15–17 | Vary bit-width, group size, layout, number of lanes and heads; measure area, timing, power | Design-space plots |
| **6. Extension and writing** | 18–20 | Eviction if time permits; thesis, demo, presentation | Final report |

**Decision gates (planned exits from risky paths)**
- **End of week 4:** if the fused version costs too much accuracy or saves too little in the model, we drop C and report A versus B. The study remains valid.
- **End of week 12:** if the board is not working, we fall back to simulation with a memory model and state the limitation clearly.

**Rule:** get a trivial design running on the actual board by about week 6, long before our real RTL is ready. Board bring-up is where projects usually lose weeks.

---

## 7. What "done" looks like

| Level | What we will have |
|---|---|
| **Minimum** | Verified RTL for A and B, running on the FPGA, with measured cycle breakdown and DDR traffic, plus the accuracy study |
| **Target** | The above plus the fused version C, a design-space sweep, and area/power results |
| **Stretch** | Eviction hardware (sink+window, then H2O), or a KIVI-style 2-bit path with residual-buffer logic |

**The three main figures**
- **Plot A:** accuracy versus compression.
- **Plot B:** theoretical versus measured speedup for A, B, C.
- **Plot C:** where the clock cycles go (memory wait, unpack, dequantize, multiply-accumulate, softmax, metadata). We consider this our most valuable result.

---

## 8. How we work together

**Workstreams** (a proposal; people can hold more than one) **[TBD: assign names]**

| Workstream | What it covers | Helpful background |
|---|---|---|
| Software and accuracy | Traces, golden models, sweeps, perplexity | Python, NumPy; basic ML helps |
| Architecture modelling | Cost model, roofline, cycle-level Python model | Computer architecture |
| RTL design | Unpacker, MAC lanes, softmax, memory interface | SystemVerilog |
| Verification | Testbenches, test vectors, bit-exact comparison | SystemVerilog, SVA |
| FPGA and host | Board bring-up, AXI/DMA, host software, counters, power | Vivado, embedded software |
| Analysis and writing | Plots, literature, documentation, presentations | Clear writing |

**Ground rules**
1. **Golden model first.** No RTL block is considered correct until it matches the golden model bit for bit on the same inputs.
2. **Frozen conventions.** Rounding (round-half-up), scale/zero-point format (fp16), bit packing order (first element in the low bits), how leftover tokens are handled, and tie-breaking rules are written in `docs/spec.md`. **Nobody changes them without agreement**, because the software, hardware, and testbenches all depend on them.
3. **Fair comparison.** A, B, and C share interfaces, memory system, and clock target.
4. **Report null results.** If the design turns out purely memory-bound, we say so.
5. **Reproducibility.** Every result file states which trace, model, and commit produced it.
6. **Everyone can explain their part.** AI tools are fine for scripts, plots, and reviews, but the RTL and the reasoning behind design decisions must be understood by the person who owns them. Please also check your institution's policy on AI assistance.
7. **Decisions get written down** in `docs/decisions.md` (date, decision, reason).

**Target repository layout**
```
kv-hw-codesign/
  src/            dump_kv.py, kvgold.py, sweep.py, fixedpoint.py, export_vectors.py
  experiments/    hw_model.py, cycle_model.py, ppl_eval.py
  rtl/            hardware modules
  tb/             testbenches and test vectors
  fpga/           constraints, block design, host software
  results/        CSV files and plots
  docs/           spec.md, decisions.md, literature.md, this overview
```

---

## 9. Metrics we will report

- **Compression:** bytes per token, including metadata.
- **Accuracy:** similarity between compressed and full-precision attention output (cosine, relative error), later perplexity.
- **Memory:** DDR bytes moved per decode step; achieved bandwidth versus the board's peak.
- **Speed:** cycles per token and per step; whether the design is memory- or compute-bound.
- **Hardware cost:** LUTs, flip-flops, block RAM, DSP slices, maximum clock (Fmax).
- **Power and energy:** estimated, and measured if the board allows.

---

## 10. Reading list

Read in this order. Items marked ★ we have confirmed exist; for the others, please confirm details yourself before citing them.

1. **KIVI** (2024), the algorithm we build on: keys per-channel, values per-token, residual buffer.
2. **BitDecoding** (HPCA 2026) ★: GPU study of why low-bit KV decoding underdelivers, and how to fix it. Check its "query transformation" idea against ours.
3. **Omni-LUT** (ISCA 2026) ★: an accelerator paired with hardware-aware KV quantization.
4. **PagedAttention / vLLM**: how KV memory is organized into blocks.
5. **StreamingLLM** and **H2O**: eviction policies for the later extension.
6. **VEDA (DAC 2025)** and **HiKV (2026)**: hardware for KV eviction and selection.
7. **DeepSeek V4.1-Flash KV-compression report** (Sept 2026): context on recent industry direction. Our knowledge of it comes from secondary summaries, so **read the original**.
8. Also relevant: Oaken, Titanus, InnerQ, TurboQuant (arXiv 2504.19874), FlightLLM.

For each paper, write five lines: the idea, the hardware it implies, the cost, the metric they report, and what could go wrong when building it.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large language model, the kind of AI behind chatbots |
| **Token** | A word-piece; the unit an LLM reads and writes |
| **Decode** | Generating output tokens one at a time |
| **Attention** | The step where a new token "looks back" at earlier tokens |
| **Query (Q), Key (K), Value (V)** | The three vectors used in attention. Keys and Values of past tokens are cached |
| **KV cache** | Stored Keys and Values of all previous tokens |
| **Head** | One independent attention unit; a model has many |
| **head_dim (D)** | Length of each vector per head (64 in our first trace) |
| **GQA** | Grouped-query attention: several query heads share one K/V head |
| **Quantization** | Storing numbers with fewer bits |
| **Scale, zero-point** | Extra numbers needed to convert a stored code back into a real value |
| **Metadata** | The scales and zero-points; they cost extra storage |
| **Group size** | How many values share one scale/zero-point |
| **Per-token / per-channel** | Which values share scale/zero-point: within one token, or across tokens for one channel |
| **Residual buffer** | Recent tokens kept in full precision until enough exist to quantize |
| **Packing / unpacking** | Fitting several small codes into one byte / separating them |
| **Dequantize** | Converting stored codes back into real numbers |
| **Golden model** | Trusted software reference that hardware must match |
| **Bit-exact** | Identical outputs, bit for bit |
| **Fixed-point** | Arithmetic on integers with an implied decimal point; what hardware usually uses |
| **RTL** | Register-transfer level hardware description (SystemVerilog) |
| **FPGA** | Reprogrammable chip on which we run our design |
| **DDR / DRAM** | Main memory attached to the board; where the KV cache lives |
| **AXI** | Standard on-chip bus for moving data between memory and logic |
| **Memory-bound / compute-bound** | Limited by data movement / by calculation |
| **Roofline** | A chart showing which of those two limits applies |
| **Fmax** | Maximum clock frequency a design can run at |
| **PPA** | Power, performance, area |
| **Eviction** | Dropping less important cache entries to save memory |
| **Perplexity** | A standard accuracy score for language models; lower is better |

---

## 12. Open questions **[TBD]**

1. Final deadline and interim review dates.
2. Which FPGA board (this decides memory bandwidth and how ambitious we can be).
3. Tools: Vivado version and simulator.
4. Who owns which workstream.
5. Whether a GPU is available for optional reference measurements (nothing depends on it).
6. Supervisor and institution requirements on deliverables and on AI-tool use.
7. Results of the first `quant` and `evict` sweeps, which will fix our first design points.

---

## 13. FAQ

**I don't know anything about LLMs. Can I still help?**
Yes. Read Section 2, then pick a workstream in Section 8. For RTL and verification you only need to understand the three-step data flow (read packed data → unpack/convert → multiply-accumulate).

**Isn't this already solved by existing papers?**
Parts of it are, mainly on GPUs and in a few accelerators. We are not trying to out-perform them. We are measuring and explaining the gap on an FPGA with a fair baseline, and building a parameterized, verified design. Section 3 explains this in detail.

**Why FPGA rather than a chip?**
It is buildable and measurable within our time and budget, with real memory and real power numbers.

**Why not 2-bit everywhere?**
Accuracy collapses quickly at very low bit-widths unless the scheme is careful (that is why KIVI treats keys and values differently). Our sweeps will show where the trade-off lies.

**What if the fused version shows no benefit?**
Then that is our finding: "the extra logic does not pay off on this platform because X." That is a legitimate result as long as it is measured and explained.

**What if the design is entirely limited by memory bandwidth?**
Then the compression itself is the main gain, and our cycle breakdown will show that the compute side is not the problem. We report it.

**Where do I start today?**
1. Read Sections 1–4.
2. Skim the KIVI paper and BitDecoding.
3. Run `python sweep.py check`, and if it passes, `python sweep.py quant`.
4. Tell the group which workstream you want.
