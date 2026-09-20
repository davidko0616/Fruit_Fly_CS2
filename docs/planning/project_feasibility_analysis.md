# FlyWire Connectome CS2 Agent — Feasibility Analysis

## Executive Summary

The project specification is **exceptionally well-structured** — it reads as the work of someone who has thought carefully about experimental design, incremental development, and scientific rigor. The phased approach, baseline comparisons, and explicit avoidance of premature scaling are all sound.

Research into the three critical technology pillars reveals that the project is **more feasible than it might initially appear**, though some areas need enrichment and the implementation path benefits from adjustments.

| Pillar | Feasibility | Risk Level |
|---|---|---|
| FlyWire Connectome Data | ✅ Excellent | 🟢 Low |
| AMD ROCm + PyTorch | ✅ Viable (with caveats) | 🟡 Medium |
| CS2 Environment Interface | ⚠️ Challenging but buildable | 🟡 Medium |
| Connectome → Neural Net (core idea) | ✅ Sound | 🟡 Medium |
| Experimental Design | ✅ Strong | 🟢 Low |


---

## 1. FlyWire Connectome Data — 🟢 FEASIBLE

### Findings

The FlyWire connectome is **well-documented, publicly accessible, and has mature Python tooling**.

- **~139,255 neurons**, **~54.5 million individual synapse annotations**
- When aggregated to neuron-to-neuron edges: **~5–10 million unique directed edges**
- Available via `caveclient` (requires free CAVE token) or bulk Parquet/CSV downloads
- Licensed under **CC-BY 4.0**

### Schema

```
pre_pt_root_id → post_pt_root_id, syn_count
```

### Rich metadata available

| Field | Use for project |
|---|---|
| `cell_type` | Subgraph selection (e.g., visual pathway neurons) |
| `nt_type` | Excitatory vs inhibitory sign of connections |
| `super_class` | Sensory / inter / motor classification |
| `neuropil` | Brain region localization |
| `flow` | Information flow direction (sensory→inter→motor) |

### Assessment

The spec's treatment of FlyWire data is accurate. The `nt_type` field (neurotransmitter: acetylcholine, GABA, glutamate) is a **valuable addition the spec doesn't mention** — it determines connection sign (excitatory/inhibitory), which is biologically critical and computationally useful.

> [!TIP]
> **Recommendation:** Use neurotransmitter type to constrain weight signs. GABA connections should have negative weights; acetylcholine connections should have positive weights. This is a low-cost addition that adds biological fidelity.

> [!TIP]
> **Recommendation:** Use the `flow` metadata (sensory → interneuron → motor) to create a biologically-motivated input→hidden→output mapping, rather than random assignment of input/output neurons.

---

## 2. AMD ROCm + PyTorch — 🟡 VIABLE (with caveats)

Good news: this is **significantly less risky than initially expected**.

### Findings

| Issue | Status |
|---|---|
| RX 7800 XT official ROCm support | ✅ **Officially supported since ROCm 6.4.1** (May 2025). `gfx1101` is on the official compatibility matrix. |
| ROCm on Linux | ✅ Ubuntu 22.04/24.04, RHEL 9.x. Official PyTorch ROCm wheels include `gfx1101`. |
| ROCm on Windows | ✅ Official **preview** PyTorch wheels from AMD (`repo.radeon.com`). Requires Windows 11 + Python 3.12. |
| `torch.sparse` on ROCm | ✅ COO/CSR via `hipSPARSE`/`rocSPARSE`. Works for standard `sparse.mm()` and `addmm()`. |
| `torch.sparse` gaps | ⚠️ No 2:4 structured sparsity (NVIDIA hardware only). Semi-structured sparsity unsupported. Autograd for BSR/BSC may have edge-case bugs. |
| Legacy workaround | `HSA_OVERRIDE_GFX_VERSION=11.0.0` — only needed for ROCm < 6.4.1. Not needed on current versions. |

### Platform Options

| Platform | Viability | Notes |
|---|---|---|
| **Linux (Ubuntu 22.04/24.04) + ROCm 6.4.1+** | ✅ **Best option** | Official support, Docker available (`rocm/pytorch`), most stable |
| **Windows 11 + AMD preview wheels** | ✅ Viable | Preview quality; install from `repo.radeon.com`; Python 3.12 required |
| **WSL2 + ROCm** | ⚠️ Moderate | Functional but GPU passthrough can be fragile |
| `torch-directml` | ❌ Avoid | Too slow for training, limited sparse ops |
| ZLUDA | ❌ Avoid | Abandoned by AMD, crashes with PyTorch autograd |

### Key Practical Warnings

> [!WARNING]
> **MIOpen Cold Start:** First execution of convolutions/matmuls compiles and benchmarks kernels for `gfx1101`. PyTorch will appear frozen for **2–10 minutes** on first run. Set `MIOPEN_LOG_LEVEL=3` to see progress. Kernels are cached in `~/.config/miopen` afterwards.

> [!WARNING]
> **BIOS Settings Required (Linux):** Enable **"Above 4G Decoding"** and **"Re-Size BAR Support"** (Smart Access Memory) in motherboard BIOS. Without this, PyTorch crashes immediately on GPU initialization.

> [!NOTE]
> **No hardware FP8 or structured sparsity:** RDNA 3 supports FP16/BF16/INT8/INT4 via WMMA (AI matrix accelerators), but lacks NVIDIA's FP8 Tensor Cores and 2:4 structured sparsity. This doesn't affect this project (we use unstructured biological sparsity), but is worth noting.

### Sparse Ops Assessment for This Project

The project needs `torch.sparse` COO/CSR matrix–vector and matrix–matrix products. These map to `rocSPARSE` on AMD and are **functional out of the box**. This is adequate for the connectome-constrained sparse linear layers.

For PyTorch Geometric (if used for GNN approaches): Modern PyG 2.4+ provides pure-PyTorch fallbacks that avoid the problematic `torch-sparse` C++ extension. Use these.

### Recommendation

**Primary:** Linux (Ubuntu 22.04) + ROCm 6.4.1+ with official PyTorch ROCm wheels. This is the supported path.

**Secondary/Development:** Windows 11 with AMD preview wheels for development and debugging. CPU-only is also fine for small-scale experiments.

**For scaling to full 139K neurons:** The 16 GB VRAM on the RX 7800 XT is actually a strength — more than most consumer NVIDIA cards at this price point (RTX 4070 has 12 GB). Whether 16 GB is sufficient for the full connectome depends on implementation efficiency, but it's competitive.

---

## 3. CS2 Environment Interface — 🟡 CHALLENGING BUT VIABLE

The CS2 interface situation is **better than initially feared**, though still the project's most engineering-heavy component.

### GSI — Corrected Findings

> [!IMPORTANT]
> **Key correction:** GSI in player mode **DOES expose player position `(x,y,z)` and view direction vector**. Earlier concerns about GSI lacking spatial data were partially incorrect.

| GSI Mode | Data Available |
|---|---|
| **Player mode (self)** | ✅ Own position `(x,y,z)`, view vector `(x,y,z)`, health, armor, ammo, weapon state, inventory, match stats, status flags (flashed/smoked/burning 0–255) |
| **Spectator / GOTV mode** | ✅ **All 10 players:** position, view vectors, health, weapons, match stats. **Plus:** all active grenades (position, velocity, type), bomb position + carrier + timer |
| **Enemy data in player mode** | ❌ Suppressed (anti-wallhack). Only scoreboard-level data (kills/deaths/score) |

**Update rate:** Configurable, but HTTP push with ~10–20 Hz practical maximum. Not tick-rate (64 Hz sub-tick).

### Server Plugin Ecosystem — Much Richer Than Expected

| Tool | What it Does | Relevance |
|---|---|---|
| **CounterStrikeSharp** (C#) | Full server plugin framework (successor to SourceMod) | Core plugin platform for observation/action bridge |
| **Metamod:Source v2** | C++ plugin loading for Source 2 | Foundation for CounterStrikeSharp |
| **CS2-Bot-Controller** (`XBribo`) | Tick-level bot `usercmd` injection: buttons, view angles, movement vectors | **Direct action interface for bots** |
| **CS2-Smarter-Bot** | Enhanced bot AI with raytraced visibility, better pathing | Reference for bot behavior modification |
| **awpy** (Python/Rust) | CS2 demo parsing → Polars DataFrames (positions, visibility, damage, kills) | Offline training data / analysis |
| **demoparser2** (Python/Rust) | High-speed Source 2 demo parsing → Pandas DataFrames | Alternative offline data source |

### Viable Architecture for This Project

The most promising approach uses **a local dedicated server + CounterStrikeSharp + CS2-Bot-Controller**:

```text
CS2 Dedicated Server (headless, -insecure)
    │
    ├── Metamod:Source v2
    │       └── CounterStrikeSharp
    │               └── Custom Plugin
    │                       │
    │                       ├── READS: All entity positions, health,
    │                       │          velocities, view angles, visibility
    │                       │          (ground truth at tick rate)
    │                       │
    │                       └── WRITES: Bot usercmd injection
    │                                   (buttons, aim angles, movement)
    │
    └── IPC Bridge (WebSocket / shared memory)
            │
            ▼
        Python RL Agent
```

This gives **tick-level observation AND action** without screen capture, without VAC concerns, and without needing a visible game client.

### What the Spec's Observation Space Can Actually Use

| Spec Observation | Source | Feasibility |
|---|---|---|
| Player position | GSI (player mode) or server plugin | ✅ |
| Player velocity | Server plugin (entity state) | ✅ |
| View yaw/pitch | GSI or server plugin | ✅ |
| Health/armor | GSI | ✅ |
| Ammo/weapon/state | GSI | ✅ |
| **Enemy position** | **Server plugin only** (suppressed in GSI player mode) | ✅ via plugin |
| Enemy distance/direction | Computed from positions | ✅ |
| Enemy visibility | Server plugin (raycast) | ✅ |
| Enemy health | Server plugin | ✅ |
| Bomb state | GSI | ✅ |

### Critical Limitation: No Fast-Forward

> [!CAUTION]
> **CS2 runs in real time.** Unlike OpenAI Gym environments, there is no `env.step()`. The game cannot be reliably accelerated beyond ~2–4x (`host_timescale`) without breaking physics. This means RL training requiring millions of interactions will take **days to weeks of wall-clock time** per experiment. This is the single biggest practical constraint.

### Mitigation Strategies for Training Speed

1. **Toy 2D environment first** (the spec already proposes this — essential)
2. **Offline imitation learning** from demo files (`awpy`/`demoparser2`) before online RL
3. **Curriculum learning** — start with extremely simplified scenarios (stationary target, small arena)
4. **Headless server** — no rendering overhead, can run multiple instances in parallel
5. **Transfer learning** — pre-train in toy environment, fine-tune in CS2

### Prior Academic Work

Notably, the project is not the first to attempt CS2 RL:
- **KTH Master's Thesis (2026):** "Evaluating Neural Network Architectures in Real-Time Deep Reinforcement Learning with PPO in CS2" — used NatureCNN, IMPALA, and LSTM variants with visual input and 4-stage curriculum
- **SiDeGame:** A 2D top-down defusal simulation created specifically because full CS2 RL was too resource-intensive
- **RekaAI/CS2-10k:** 600+ hours of CS2 egocentric video for world models and imitation learning

### Assessment

The CS2 interface is **buildable** with the CounterStrikeSharp + CS2-Bot-Controller stack, but requires:
1. Setting up a local CS2 dedicated server
2. Installing Metamod:Source v2 + CounterStrikeSharp
3. Writing a custom C# plugin that bridges game state to Python
4. This is a significant engineering sub-project (~1–2 weeks of focused work)

The spec should treat this as an explicit engineering milestone, not an assumed capability.

---

## 4. Connectome → Neural Network Conversion — 🟡 MEDIUM RISK

### What's Sound

- The sparse graph representation approach is correct
- The `W_ij = g(S_ij)` formulation is appropriate
- The Configuration A/B/C experimental design is well-structured
- Progressive scaling (100 → 139K neurons) is essential

### Concerns

**4a. Sparse matmul performance on AMD**

PyTorch's sparse operations are less optimized than dense operations, especially on non-NVIDIA hardware. For a 139K-node sparse graph with ~5M edges, sparse matrix-vector products during forward/backward passes need to be efficient.

**4b. Subgraph selection strategy is underspecified**

The spec says to scale from 100 to 139K neurons, but doesn't specify **which** neurons to select at each scale. This matters enormously:

- Random 100 neurons from 139K will likely form a disconnected graph
- Biologically meaningful subsets (e.g., a specific neuropil, a sensory-to-motor pathway) will have much better connectivity

> [!TIP]
> **Recommendation:** Use FlyWire's `neuropil`, `flow`, and `cell_type` metadata to select **biologically meaningful subgraphs** at each scale. For example:
> - 100 neurons: A specific small circuit (e.g., a known visual processing motif)
> - 1,000 neurons: A complete neuropil region
> - 10,000 neurons: Multiple connected neuropils forming a sensory→motor pathway

**4c. Graph is directed and may have cycles**

The biological connectome has extensive recurrent connectivity. This means the "forward pass" of the network is not a simple feedforward computation. The spec mentions this in Level 2/3 but doesn't address how Level 1 handles recurrence.

Options:
1. **Topological sort + layer assignment:** Assign neurons to layers based on longest path from input neurons. Breaks cycles by treating back-edges as "recurrent" connections computed from the previous timestep.
2. **Message-passing GNN approach:** Run K rounds of message passing. Each round updates all neurons simultaneously.
3. **Ignore cycles initially:** Perform a topological sort, remove back-edges, treat it as a DAG. Simplest but loses biological structure.

> [!IMPORTANT]
> **Recommendation:** Start with option 1 (topological sort with recurrent back-edges). This preserves cycles as temporal recurrence while allowing standard backpropagation through time (BPTT) or truncated BPTT.

**4d. Input/output neuron assignment**

The spec doesn't specify how to map game observations to input neurons or output neurons to actions. The FlyWire data has `flow` metadata (sensory/inter/motor) that provides a natural mapping:
- Sensory neurons → input layer
- Motor neurons → output layer
- Interneurons → hidden processing

---

## 5. Experimental Design — 🟢 SOUND

The baseline comparison framework (FlyWire vs. Random Sparse vs. MLP) is **exactly right** for a research project. The six experimental questions in Section 15 are well-formulated.

### Minor Suggestions

**5a. Add a "matched dense" baseline**

In addition to the MLP baseline, add a dense network with the **same number of parameters** as the FlyWire model. This controls for parameter count rather than architecture.

**5b. Statistical rigor**

Each experiment should be run with **multiple random seeds** (≥5) and results reported with confidence intervals. The spec mentions recording random seeds but doesn't explicitly call for multiple runs.

**5c. Learning curves, not just final performance**

Compare **learning curves** (performance vs. training steps), not just final performance. Biological topology might enable faster learning even if final performance is similar.

---

## 6. Reward Function — 🟢 SOUND

The simple initial reward function is appropriate. The spec correctly warns against premature complexity.

### One Addition

For Task 1 (aiming), a **shaping reward** based on angle-to-target would dramatically accelerate learning:

$$R_{\text{aim}} = -\alpha \cdot \theta_{\text{error}} + w_h \cdot \text{hit}$$

where $\theta_{\text{error}}$ is the angular distance to the target.

Without shaping, the agent must discover through random exploration that firing when aimed at an enemy produces reward — which is extremely unlikely in a large action space.

---

## 7. Development Order — Reconstructed Recommendation

Given the feasibility findings, here is my recommended reconstruction of the development phases:

### Phase 0 — Platform Setup *(updated — lower risk than expected)*
- [ ] **If on Windows 11:** Install AMD preview PyTorch wheels from `repo.radeon.com` (Python 3.12)
- [ ] **If on Linux (recommended for training):** Install ROCm 6.4.1+, PyTorch ROCm wheels
- [ ] **BIOS check (Linux):** Enable "Above 4G Decoding" and "Re-Size BAR Support"
- [ ] **Validate sparse ops:** Test `torch.sparse.mm()` with a ~10K×10K CSR matrix with ~100K nonzeros on GPU
- [ ] **Expect MIOpen cold start:** First GPU matmul will take 2–10 minutes (kernel compilation). Set `MIOPEN_LOG_LEVEL=3`.

### Phase 1 — FlyWire Data Acquisition *(unchanged, low risk)*
- [ ] Register for CAVE token at codex.flywire.ai
- [ ] Download connectivity data via `caveclient` or bulk Parquet download
- [ ] Inspect schema (`pre_pt_root_id`, `post_pt_root_id`, `syn_count`, etc.)
- [ ] Validate neuron counts (~139K) and synapse counts (~54.5M)
- [ ] Download neuron metadata (cell_type, nt_type, flow, neuropil, super_class)
- [ ] Document dataset version (e.g., materialization v783) and download date

### Phase 2 — Sparse Graph Construction *(enriched with metadata)*
- [ ] Build sparse adjacency (COO or CSR) from connectivity table
- [ ] Filter by `cleft_score` threshold (synapse detection confidence)
- [ ] Aggregate to neuron-to-neuron synapse counts
- [ ] Analyze graph statistics (degree distribution, connected components, diameter)
- [ ] Implement subgraph extraction by neuropil, cell_type, and flow
- [ ] Map `nt_type` to excitatory/inhibitory sign per neuron
- [ ] Verify: graph is sparse, directed, has cycles (recurrent)

### Phase 3 — Small FlyWire PyTorch Model *(enriched)*
- [ ] Select a biologically meaningful ~100-neuron subgraph using metadata
- [ ] Assign input/output neurons using `flow` metadata (sensory→motor)
- [ ] Implement sparse linear layer with connectome-constrained mask
- [ ] Incorporate neurotransmitter sign constraints (GABA→negative, ACh→positive)
- [ ] Handle recurrent connections (topological sort + back-edges as temporal recurrence)
- [ ] Verify forward pass produces correct output shapes
- [ ] Verify mask is maintained during gradient updates

### Phase 4 — Synthetic Task Training *(unchanged)*
- [ ] Simple classification task (e.g., XOR, spiral, MNIST subset)
- [ ] Verify gradients flow correctly through sparse architecture
- [ ] Verify that masking is maintained (no weight "regrowth" in masked positions)
- [ ] Benchmark GPU memory and speed at this scale

### Phase 5 — Baselines *(enriched)*
- [ ] MLP with same parameter count
- [ ] Random sparse network with same neuron/connection count
- [ ] Dense network with same parameter count *(new baseline)*
- [ ] Compare on synthetic task with **≥5 random seeds each**
- [ ] Report learning curves, not just final performance

### Phase 6 — Toy 2D Combat Environment *(elevated to primary platform)*
- [ ] Implement 2D grid environment with agent, enemy, obstacles
- [ ] Structured observation space (position, enemy direction, distance, visibility)
- [ ] Discrete action space (move 4 directions, turn, fire)
- [ ] Simple reward function (hit = +1, get hit = −1, time penalty)
- [ ] OpenAI Gym-compatible `env.step()` interface
- [ ] **This is the primary experimental platform for the core research question**

### Phase 7 — RL Training in Toy Environment
- [ ] PPO or similar on-policy algorithm
- [ ] Train FlyWire, random sparse, MLP, and dense baselines
- [ ] Compare learning curves and final performance
- [ ] Scale from 100 → 1,000 → 5,000 → 10,000 neurons
- [ ] Answer experimental questions 1–5 from Section 15

### Phase 8 — CS2 Server Infrastructure *(new explicit phase)*
- [ ] Install CS2 + set up local dedicated server (`-insecure`)
- [ ] Install Metamod:Source v2 + CounterStrikeSharp
- [ ] Evaluate CS2-Bot-Controller for usercmd injection
- [ ] Build custom C# plugin: expose game state via WebSocket/HTTP to Python
- [ ] Minimum: player position + enemy positions + health + view angles
- [ ] Build Python client that receives observations and sends actions
- [ ] **Decision point:** Is the plugin bridge reliable enough for training?

### Phase 9 — CS2 Integration
- [ ] Map toy environment observation/action interface to CS2 plugin bridge
- [ ] Task 1: Aim at stationary bot (bot_stop 1)
- [ ] Task 2: Navigate between waypoints
- [ ] Validate reward signal (kills, damage via plugin events)

### Phase 10+ — CS2 Training & Scaling
- [ ] Train FlyWire agent on simple CS2 tasks
- [ ] Compare with baselines under identical CS2 conditions
- [ ] Progressively increase task complexity
- [ ] Scale network size (with GPU memory profiling at each step)

---

## 8. What Doesn't Need Reconstruction

| Section | Verdict |
|---|---|
| §1 Project Overview | ✅ Excellent framing |
| §2 Research Question | ✅ Well-formulated |
| §3 Motivation | ✅ Compelling |
| §4 FlyWire Connectome | ✅ Accurate |
| §5 Biological Accuracy Levels | ✅ Smart tiering |
| §6 Architecture | ✅ Sound |
| §9 Action Space | ✅ Good incremental approach |
| §10 Learning Tasks | ✅ Good progression |
| §11 Reward Function | ✅ Appropriately simple |
| §13 Trainable vs Fixed | ✅ Important experimental variable |
| §14 Baselines | ✅ Essential |
| §15 Experimental Questions | ✅ Well-formulated |
| §16 Scaling Strategy | ✅ Critical and correct |
| §19 Dev Philosophy | ✅ Exactly right |
| §21 Scientific Constraints | ✅ Important honesty |
| §22 Data Provenance | ✅ Good practice |
| §23 Reproducibility | ✅ Essential |

---

## 9. Summary of Required Changes to the Spec

### 🟡 Should Fix Before Phase 3

1. **GPU Platform Details:** The spec's `PyTorch → ROCm → AMD RX 7800 XT` stack is correct in principle, but should note that ROCm 6.4.1+ is required, MIOpen cold-start is expected, and Linux is preferred over Windows for training. The Windows preview wheels work but are less tested.
2. **Subgraph selection strategy:** Add a concrete plan for which neurons to select at each scale, using biological metadata (`neuropil`, `cell_type`, `flow`). Random selection from 139K neurons will produce disconnected graphs.
3. **Recurrence handling:** Specify how cycles in the biological graph are handled computationally (recommend: topological sort + back-edges as temporal recurrence with BPTT).
4. **Input/output neuron mapping:** Use `flow` metadata (sensory/motor) rather than arbitrary assignment.
5. **Neurotransmitter sign constraints:** Use `nt_type` metadata to constrain weight signs (GABA → inhibitory/negative, acetylcholine → excitatory/positive). Low-cost, high biological value.

### 🟡 Should Fix Before Phase 8 (CS2)

6. **CS2 observation space:** The spec's observation space is achievable, but requires a custom CounterStrikeSharp server plugin — not just GSI. Enemy spatial data is suppressed in GSI player mode. This is an engineering sub-project that should be treated as an explicit milestone.
7. **CS2 training speed:** Add explicit acknowledgment that CS2 runs in real time with no fast-forward. Each RL experiment will take days-to-weeks of wall-clock time. Plan accordingly (curriculum learning, offline pre-training from demos, parallel headless servers).

### 🟢 Nice to Add

8. **Matched-parameter dense baseline** in addition to MLP and random sparse.
9. **Multiple seeds per experiment:** Explicitly require ≥5 random seeds per condition.
10. **Learning curve comparison** in addition to final performance.
11. **Reward shaping** for aiming tasks (angular distance to target).
12. **Offline imitation learning** from CS2 demo files (`awpy`/`demoparser2`) as pre-training before online RL.

---

## 10. Overall Verdict

**The project is scientifically sound, the spec is exceptionally well-written, and it is more feasible than it might initially appear.**

### What's Strong

- The core research question is genuinely interesting and testable
- The experimental design with baselines and progressive scaling is rigorous
- The incremental development philosophy (toy task → toy environment → CS2) is exactly right
- The FlyWire data is well-suited to this purpose and has good tooling
- The RX 7800 XT + ROCm stack is now officially supported (as of ROCm 6.4.1)

### What Needs Work

The two areas requiring attention are **practical engineering problems, not conceptual ones:**

1. **CS2 integration** requires building a CounterStrikeSharp plugin bridge — a significant but well-defined engineering task with existing community tools (`CS2-Bot-Controller`) to build on. The real constraint is training speed (real-time execution), which makes the toy 2D environment the essential primary experimental platform.

2. **Biological metadata** (neurotransmitter types, flow classification, neuropil regions) should be incorporated from the start — the FlyWire data provides this for free and it meaningfully improves both the scientific rigor and computational design of the network.

### Recommendation

**Proceed with implementation.** Start with Phase 0 (platform setup) and Phase 1 (FlyWire data acquisition) in parallel. The first real scientific milestone — training a small connectome-constrained network on a synthetic task — is achievable within 1–2 weeks.

**Bottom line: The science is ready. The infrastructure is viable. The biggest risk is CS2 training speed, which the spec already mitigates with its toy environment strategy.**
