# Implementation Plan — Phases 0–4: From Zero to Trainable Connectome Network

## Goal

Set up the project from scratch, download FlyWire connectome data, build a sparse graph representation, construct a small connectome-constrained PyTorch network, and successfully train it on a synthetic classification task — all on Windows with CPU PyTorch.

This covers the **first milestone** from the project spec:

> Download a subset of the FlyWire connectome, convert it into a sparse graph, construct a trainable PyTorch network using the graph topology, and successfully train it on a simple synthetic classification task.

---

## Open Questions

> [!IMPORTANT]
> **CAVE Token:** To use the `caveclient` Python API, you need to register for a free CAVE token at [https://global.daf-apis.com/auth/api/v1/user/token](https://global.daf-apis.com/auth/api/v1/user/token) (requires a Google account). However, for Phase 1 we can **skip this entirely** by downloading the bulk data files directly from Zenodo (no authentication required). Do you have a preference?

> [!NOTE]
> **Subgraph selection for the first ~100-neuron experiment:** The FlyWire data has rich metadata (cell types, neuropil regions, neurotransmitter types). For the first small experiment, I plan to select a biologically meaningful subgraph — for example, neurons from the **Antennal Lobe (AL)** olfactory processing circuit, which has well-characterized sensory→interneuron→projection neuron connectivity. This gives us a natural input→hidden→output structure. Is this approach acceptable, or do you have a preferred brain region?

---

## Proposed Changes

### Phase 0 — Project Setup & Environment

#### [NEW] [requirements.txt](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/requirements.txt)

Python dependencies for the project:
```
torch>=2.0
numpy
scipy
pandas
pyarrow
matplotlib
networkx
tqdm
pyyaml
```

No ROCm or GPU dependencies yet — CPU-only for initial development.

#### [NEW] [setup.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/setup.py)

Minimal package setup for importable project modules.

#### [MODIFY] [README.md](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/README.md)

Update with project description, setup instructions, and development status.

---

### Phase 1 — FlyWire Data Acquisition

#### [NEW] [data/raw/](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/data/raw/) (directory)

Downloaded FlyWire v783 data files from Zenodo (no auth required):

| File | Source | Size | Purpose |
|---|---|---|---|
| `proofread_connections_783.feather` | [Zenodo 10676866](https://doi.org/10.5281/zenodo.10676866) | ~852 MB | Neuron-to-neuron connectivity with synapse counts |
| `proofread_root_ids_783.npy` | Same Zenodo | ~1.1 MB | Array of all 139,255 proofread neuron IDs |

#### [NEW] [data/metadata/](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/data/metadata/) (directory)

| File | Source | Size | Purpose |
|---|---|---|---|
| `neuron_annotations.tsv` | [GitHub flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | ~35 MB | Cell types, neurotransmitter types, flow, neuropil, super_class |

#### [NEW] [connectome/download.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/connectome/download.py)

Script to download the above files from Zenodo and GitHub. Records:
- Dataset version (v783)
- Download date
- File checksums
- Source URLs

Writes a `data/manifest.json` with provenance metadata.

#### [NEW] [connectome/inspect.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/connectome/inspect.py)

Script to load and inspect the downloaded data:
- Print schema and column names
- Print row counts and basic statistics
- Validate neuron counts match expected ~139,255
- Print sample rows
- Print distribution of neurotransmitter types, flow categories, neuropil regions

---

### Phase 2 — Sparse Graph Construction

#### [NEW] [connectome/graph.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/connectome/graph.py)

Core graph module. Implements:

1. **`ConnectomeGraph` class** — loads connectivity + annotations and builds:
   - Sparse adjacency matrix (scipy CSR format)
   - Neuron metadata lookup (cell_type, nt_type, flow, neuropil)
   - Neurotransmitter sign mapping (GABA/GLUT → inhibitory, ACH → excitatory)

2. **`extract_subgraph()`** — extract a connected subgraph by:
   - Neuropil region (e.g., `AL_R` for right Antennal Lobe)
   - Flow type (e.g., sensory, inter, motor)
   - Cell type prefix (e.g., all neurons matching `PN*`)
   - Neuron count limit (take top-N by degree)

3. **`assign_io_neurons()`** — using `flow` metadata:
   - `afferent` / sensory neurons → input layer
   - `efferent` / motor / descending neurons → output layer
   - `intrinsic` / central neurons → hidden processing

4. **Graph statistics:**
   - Node count, edge count, density
   - In-degree / out-degree distributions
   - Connected components (strong/weak)
   - Fraction excitatory vs inhibitory

#### [NEW] [connectome/analysis.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/connectome/analysis.py)

Visualization and analysis utilities:
- Degree distribution plots
- Adjacency matrix spy plot (sparsity pattern)
- Neurotransmitter type distribution
- Subgraph connectivity summary

---

### Phase 3 — FlyWire-Derived PyTorch Model

#### [NEW] [models/sparse_layer.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/models/sparse_layer.py)

**`ConnectomeSparseLinear`** — a PyTorch `nn.Module` implementing a sparse linear layer constrained by connectome topology:

```python
# Conceptual design:
class ConnectomeSparseLinear(nn.Module):
    def __init__(self, adjacency_coo, synapse_counts, nt_signs):
        # adjacency_coo: (row_indices, col_indices) from connectome
        # synapse_counts: initial weight magnitudes from synapse count
        # nt_signs: +1 (excitatory) or -1 (inhibitory) per source neuron
        
        # Learnable weight magnitudes (only where connections exist)
        self.weight_magnitudes = nn.Parameter(...)
        
        # Fixed binary mask (non-learnable) — defines topology
        self.register_buffer('mask', ...)
        
        # Fixed neurotransmitter signs (non-learnable)
        self.register_buffer('nt_signs', ...)
    
    def forward(self, x):
        # Effective weight = magnitude * sign * mask
        # Sparse matrix-vector product
        ...
```

Key properties:
- **Fixed topology:** The mask is a buffer, not a parameter. Gradient updates cannot create new connections.
- **Neurotransmitter constraints:** GABA neurons have negative weights, ACH neurons have positive weights.
- **Weight initialization options:** Random, synapse-count-based, or normalized synapse count.
- Uses `torch.sparse` CSR tensors for the actual forward computation.

#### [NEW] [models/flywire_network.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/models/flywire_network.py)

**`FlyWireNetwork`** — the full connectome-constrained network:

```python
class FlyWireNetwork(nn.Module):
    def __init__(self, graph, input_dim, output_dim, num_steps=3):
        # graph: ConnectomeGraph subgraph
        
        # Input projection: input_dim → sensory neurons
        self.input_proj = nn.Linear(input_dim, num_sensory_neurons)
        
        # Connectome processing: K steps of sparse message passing
        # (handles recurrent connections via unrolled steps)
        self.connectome_layer = ConnectomeSparseLinear(...)
        
        # Output projection: motor neurons → output_dim
        self.output_proj = nn.Linear(num_motor_neurons, output_dim)
    
    def forward(self, x):
        # Project input to sensory neurons
        h = self.input_proj(x)  # → sensory neuron activations
        
        # Pad to full neuron vector
        state = zeros(num_neurons)
        state[sensory_indices] = h
        
        # K steps of recurrent processing
        for step in range(self.num_steps):
            state = activation(self.connectome_layer(state))
        
        # Read out from motor neurons
        output = self.output_proj(state[motor_indices])
        return output
```

The `num_steps` parameter controls how many rounds of message-passing occur, which handles graph cycles as temporal recurrence.

#### [NEW] [models/mlp_baseline.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/models/mlp_baseline.py)

Standard MLP baseline with configurable hidden sizes. Will be used for comparison in Phase 5.

#### [NEW] [models/random_sparse.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/models/random_sparse.py)

Random sparse network baseline — same number of neurons and connections as the FlyWire model, but with randomly generated topology. Uses the same `ConnectomeSparseLinear` layer with a random mask.

---

### Phase 4 — Synthetic Task Training

#### [NEW] [training/train_synthetic.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/training/train_synthetic.py)

Training script for the synthetic classification task:
- **Task:** Spiral dataset (2D input, N classes) — nonlinear, requires hidden representations
- **Models:** FlyWire network (and later baselines)
- **Training:** Standard cross-entropy loss, Adam optimizer
- **Logging:** Loss curves, accuracy, parameter count, connection count
- **Output:** Saved model, training plots, metrics JSON

#### [NEW] [training/configs/synthetic_100.yaml](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/training/configs/synthetic_100.yaml)

Configuration file for the ~100-neuron experiment:
```yaml
experiment:
  name: synthetic_spiral_100
  seed: 42

data:
  task: spiral
  n_classes: 3
  n_samples: 1000

connectome:
  version: 783
  subgraph:
    method: neuropil   # or: cell_type, flow, custom
    region: AL_R       # right Antennal Lobe
    max_neurons: 100
  weight_init: normalized_synapse_count
  
model:
  num_steps: 3          # recurrent processing steps
  activation: relu

training:
  epochs: 200
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
```

#### [NEW] [tests/test_graph.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/tests/test_graph.py)

Unit tests for the graph module:
- Loading connectivity data
- Subgraph extraction produces connected graph
- Neurotransmitter sign mapping is correct
- IO neuron assignment uses flow metadata
- Sparse adjacency has correct shape and nnz

#### [NEW] [tests/test_model.py](file:///c:/Users/chung/Documents/Fruit_Fly_CS2/tests/test_model.py)

Unit tests for the model:
- Forward pass produces correct output shape
- Mask is maintained after gradient update (no weight regrowth)
- Neurotransmitter sign constraints are enforced
- Model parameter count matches expected

---

## Repository Structure After Implementation

```
Fruit_Fly_CS2/
├── connectome/
│   ├── __init__.py
│   ├── download.py          # Phase 1: Data download
│   ├── inspect.py           # Phase 1: Data inspection
│   ├── graph.py             # Phase 2: Sparse graph
│   └── analysis.py          # Phase 2: Graph analysis
├── models/
│   ├── __init__.py
│   ├── sparse_layer.py      # Phase 3: Connectome sparse layer
│   ├── flywire_network.py   # Phase 3: FlyWire network
│   ├── mlp_baseline.py      # Phase 3: MLP baseline
│   └── random_sparse.py     # Phase 3: Random sparse baseline
├── training/
│   ├── __init__.py
│   ├── train_synthetic.py   # Phase 4: Synthetic training
│   └── configs/
│       └── synthetic_100.yaml
├── tests/
│   ├── test_graph.py
│   └── test_model.py
├── data/
│   ├── raw/                 # Downloaded FlyWire files
│   └── metadata/            # Neuron annotations
├── requirements.txt
├── setup.py
└── README.md
```

---

## Verification Plan

### Automated Tests
```bash
# Unit tests for graph and model modules
python -m pytest tests/ -v

# Verify synthetic training runs end-to-end
python training/train_synthetic.py --config training/configs/synthetic_100.yaml
```

### Manual Verification

After each phase:

| Phase | Verification |
|---|---|
| Phase 0 | `python -c "import torch; print(torch.__version__)"` succeeds |
| Phase 1 | `data/raw/proofread_connections_783.feather` exists, `connectome/inspect.py` prints correct schema and ~139K neurons |
| Phase 2 | `connectome/graph.py` produces a ~100-node subgraph with known statistics, spy plot shows sparse structure |
| Phase 3 | Forward pass through FlyWire network produces correct output shape, mask is maintained after `.backward()` |
| Phase 4 | Training loss decreases, accuracy > random baseline (33% for 3-class spiral), training curve plot saved |

### Key Invariants to Check
- Sparse mask is **never modified** by gradient updates
- Neurotransmitter sign constraints are **enforced after every optimizer step**
- Subgraph is **connected** (at least weakly)
- Parameter count matches expected (only weights where biological connections exist)
