# Literature Audit for the KV-Cache Hardware Project

*Prepared 30 September 2026 from the local `docs/Papers` link list and primary/official sources where available. This is a screening map, not a substitute for reading the full papers before writing related work. The local list contains two IEEE links that could not be identified reliably; do not cite them until corrected.*

## 1. Core cache policy and hardware papers

| Work | Main idea / evidence | Relevance and implication for this project |
|---|---|---|
| **H2O: Heavy-Hitter Oracle** (arXiv:2306.14048) | Dynamically retains tokens with high accumulated attention contribution plus recent tokens. Software/inference evaluation. | Defines the dynamic-policy baseline. Hardware cost to measure: importance-score state/update, victim selection, slot mapping. Not a novel policy for this project. [Primary paper](https://arxiv.org/abs/2306.14048) |
| **StreamingLLM: Attention Sinks** (arXiv:2309.17453; ICLR 2024) | Keeps initial sink tokens and a rolling recent window, enabling bounded-cache streaming. | Defines the simple, regular-addressing baseline. [Primary paper](https://arxiv.org/abs/2309.17453) |
| **VEDA: Voting-Based Eviction + Flexible Accelerator** (arXiv:2507.00797; DAC 2025) | Proposes voting-based eviction and an accelerator/dataflow for LLM generation, with flexible PE mapping and serial nonlinear scheduling. | Very close prior art: a new custom eviction accelerator would overlap. Read implementation and evaluation closely; our candidate gap must be the controlled cost comparison of canonical policy managers and their memory organization, not “hardware eviction exists.” [Paper](https://arxiv.org/abs/2507.00797) |
| **HiKV: Hierarchical Importance-Aware KV Cache** (arXiv:2607.22389) | Combines token eviction and finer element selection; uses a reconfigurable importance sorter. | Closest current accelerator work. It already makes policy/importance selection a hardware design issue. Establish a precise non-overlapping scope with the supervisor. [Paper](https://arxiv.org/abs/2607.22389) |
| **IIT Jodhpur: Graph-Based Dynamic KV-Cache Compression** (ISCAS 2026, DOI 10.1109/ISCAS66217.2026.11562780) | Models cache entries as nodes in a similarity graph and merges/compresses similar entries; published by a team including Prof. Binod Kumar. Public abstracts/reporting describe GPT-2/TinyLlama evaluation and a memory/quality trade-off. | Direct local-IIT precedent for algorithmic cache compression. It is not the same as an FPGA policy-controller comparison, but it makes a generic “dynamic compression” novelty claim untenable. Read the final paper and verify whether its hardware measurements are synthesized, simulated, or on-board before positioning against it. [IIT Jodhpur faculty page](https://binodkumar23.github.io/) · [ISCAS program](https://2026.ieee-iscas.org/assets/ISCAS_2026_Program.pdf) |
| **KIVI: Tuning-Free Asymmetric 2-bit Quantization** (arXiv:2402.02750) | Quantizes keys and values differently to reduce cache precision/storage. | Relevant if quantization returns as an extension; not part of the policy-only core because it confounds policy costs. [Paper](https://arxiv.org/abs/2402.02750) |
| **FlightLLM** (arXiv:2401.03868; FPGA 2024) | Full FPGA LLM mapping flow with FPGA-aware compute, memory hierarchy, and decode strategy. | Demonstrates what a broad FPGA inference contribution entails. Current project is intentionally a smaller cache-management subsystem, not a competing full LLM accelerator. [Paper](https://arxiv.org/abs/2401.03868) |

## 2. Other KV-cache policy, compression, and memory systems in the link list

| Work/link | What it studies | Classification |
|---|---|---|
| **CacheGen** (arXiv:2310.07240; SIGCOMM 2024) | Compresses and streams KV state for context loading/prefix reuse over limited bandwidth; adaptive compression level. | Relevant to data movement and transfer; not a token-retention controller. [Paper](https://arxiv.org/abs/2310.07240) |
| **RocketKV** (OpenReview `RyOpooIxDF`; arXiv:2502.14051) | Two-stage compression: coarse token eviction, then fine-grained query-aware sparse attention. | Important competing software+algorithm/hardware-aware policy; its irregular/top-k accesses matter to hardware comparison. Not a simple eviction-only baseline. [Paper](https://arxiv.org/abs/2502.14051) |
| **ShadowKV** (OpenReview `oa7MYAO6h6`; ICML 2025) | Uses CPU/host-side low-rank representations and selective GPU retrieval to expand effective KV capacity and throughput. | Memory-tier and sparse retrieval work; not a classical eviction policy. Shows why one must distinguish saved capacity from actual transferred bytes and lookup overhead. [Project/paper code](https://github.com/ByteDance-Seed/ShadowKV) |
| **Cake: Compute or Load KV Cache? Why Not Both?** (OpenReview `WOyOtaO6lQ`; ICML 2025) | Overlaps recomputation and KV-cache I/O for prefix-cache loading to reduce time-to-first-token. | Serving/storage scheduling, mostly prefill/cache-load path; background for the broader hardware/software framing, not a decode eviction baseline. [ICML poster](https://icml.cc/virtual/2025/poster/45020) |
| **PagedAttention / vLLM** (arXiv:2309.06180; SOSP 2023) | Block-based KV allocation and indirection to reduce fragmentation and support sharing/serving. | Essential memory-layout context. The proposed FPGA cache manager is single request and bounded; it does not reproduce paged serving. [Paper](https://arxiv.org/abs/2309.06180) |
| **FlexGen** (arXiv:2303.06865) | LP-based placement/scheduling across GPU, CPU, disk; compression/offloading for throughput with limited GPU memory. | General heterogeneous-memory inference; context for hardware-aware software co-design, not a retention policy. [Paper](https://arxiv.org/abs/2303.06865) |
| **Cache-Resident LLM Inference in GB-Scale LLCs** (arXiv:2606.25353) | Decouples weight-centric work from attention/KV management on clustered CPU systems with large last-level caches. | Memory hierarchy/system organization, not an FPGA policy algorithm. [Paper](https://arxiv.org/abs/2606.25353) |
| **DistServe** (arXiv:2401.09670) | Disaggregates prefill and decode resources for serving goodput/QoS. | System-level serving context, not cache replacement. [Paper](https://arxiv.org/abs/2401.09670) |
| **Sarathi-Serve** (arXiv:2308.16369) | Chunked prefill and decode scheduling to improve serving throughput. | Serving scheduler context; not cache policy hardware. [Paper](https://arxiv.org/abs/2308.16369) |
| **KV Cache Optimization Strategies for Scalable Inference** (arXiv:2603.20397) | Recent survey/taxonomy of eviction, compression, hybrid memory, attention mechanisms and combinations. | Useful for map and terminology; primary sources must support technical claims. [Survey](https://arxiv.org/abs/2603.20397) |
| **LLM Acceleration Based on KV Cache Management** (arXiv:2412.19442) | Survey of the field. | Background taxonomy; check overlap with the newer survey and do not count surveys as implementation evidence. [Survey](https://arxiv.org/abs/2412.19442) |
| **Large Language Diffusion Models** (arXiv:2502.09992) | Diffusion-style text generation rather than standard autoregressive decoding. | Background only; not directly applicable to this cache-manager design. [Paper](https://arxiv.org/abs/2502.09992) |
| **The Llama 3 Herd of Models** (arXiv:2407.21783) | Model-family report, architecture, training and evaluation. | Model/attention background; not KV policy or hardware design. [Report](https://arxiv.org/abs/2407.21783) |
| **Attention Is All You Need** (arXiv:1706.03762) | Transformer/attention foundations. | Introductory background, not related work for the proposed contribution. [Paper](https://arxiv.org/abs/1706.03762) |

## 3. Local list links that need correction or de-prioritization

- `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11206428&tag=1`: could not verify the title/authors/abstract from the identifier and stamp URL. Resolve via IEEE metadata or replace with DOI/title before use.
- `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11534083&tag=1`: could not verify as a KV-cache paper; the numeric identifier may be mistyped or point to unrelated content. Do not cite as currently written.
- `https://pub.towardsai.net/llm-inference-handbook-2026-135c266b86e7`: secondary tutorial/blog; use only for orientation, never as evidence for research claims.
- NVIDIA inference optimization blog and Jay Alammar's Illustrated Transformer are useful explanations, not primary research evidence.
- OpenReview IDs are discoverable by their ID, but cite the final paper/venue version. RocketKV and ShadowKV are resolved above; Cake is a cache-load scheduler rather than an eviction policy.

## 4. IIT research context relevant to project positioning

- **IIT Jodhpur:** Prof. Binod Kumar's group lists an ISCAS 2026 paper on graph-based dynamic KV-cache compression. This is the closest identified IIT faculty-led project and overlaps with algorithmic KV selection/compression. It is a reason to narrow the proposal to measurable FPGA policy-manager cost, not to claim an unoccupied topic. [Faculty/publications](https://binodkumar23.github.io/) · [ISCAS 2026 program](https://2026.ieee-iscas.org/assets/ISCAS_2026_Program.pdf)
- **IIT Delhi:** LCS2, led by Prof. Tanmoy Chakraborty, lists KV-cache and context compression within efficient/controllable LLM research. This is adjacent algorithm/system work, not evidence of a matching FPGA implementation. [Lab](https://www.lcs2.in/)
- **IIT Guwahati:** its CSE Computer Architecture and Embedded Systems area explicitly lists near-memory processing/ML accelerators and memory management/emerging memories. This provides a relevant architecture-and-memory research context, but the public group page does not identify a specific LLM KV-cache project. [Research groups](https://iitg.ac.in/cse/public_html_31012025/cseresearchgroups)
- **IIT Patna:** the EE department lists VLSI architectural design, digital VLSI, FPGA-based system design among local expertise areas. This supports platform/supervision fit, not novelty. [Department/annual report](https://www.iitp.ac.in/departments/electrical-engineering)

## 5. What the literature says about a hardware project

The field is not split into “software policies” versus “hardware.” Strong systems papers co-design them: VEDA and HiKV alter retention/selection to suit dedicated hardware; PagedAttention adapts memory layout/runtime; FlightLLM redesigns dataflow and memory organization for FPGA. Therefore an FPGA project is sensible when the hardware question is explicit and the chosen policy materially changes update work, locality, controller state, or bandwidth.

For this project, measure the *whole path*: policy-update work plus attention, and count actual traffic. At equal retained-token count, both SW and H2O may read roughly the same number of K/V vectors in dense attention; any quality advantage of H2O must be weighed against score update and selector costs. A software quality comparison alone is not a hardware contribution; an RTL demo alone is not enough to establish utility.

