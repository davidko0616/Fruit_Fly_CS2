# FlyWire Connectome-Based CS2 Learning Agent

This project aims to investigate whether the neural connectivity structure of the Drosophila melanogaster (fruit fly) brain, reconstructed through the FlyWire connectome, can be used as the architecture of a trainable artificial agent capable of learning to play Counter-Strike 2 (CS2).

## Setup
```bash
pip install -r requirements.txt
pip install -e .
```

## Downloading Data
```bash
python connectome/download.py
```

## ZLUDA experiment

An isolated Windows ZLUDA environment was tested on the RX 7800 XT. CPU model
validation passes, but GPU sparse execution is blocked by an unsupported
`cusparseSetStream` call. See [setup, results, and reproduction instructions](tools/ZLUDA.md).

## CPU training milestone

The 100-neuron AL_R network trained on a three-class spiral dataset and achieved
**99.3% held-out test accuracy** for seed 42 (148 test examples; chance is 33.3%).
The model has 7,615 fixed directed connections and 7,778 trainable parameters.
This demonstrates learning on a synthetic task; comparisons against baselines
and additional seeds remain pending.

```bash
python -m unittest discover -s tests -v
python -m training.train_synthetic --config training/configs/synthetic_100.yaml
```

Training writes to `artifacts/synthetic/` and refuses to overwrite an existing
run directory. Change the seed or the configured experiment name for a new run.
The [saved experiment](experiments/cpu_spiral_100/README.md) includes the
checkpoint, graph, synthetic dataset, exact configuration, metrics, provenance,
and training plot.

See [project status](docs/STATUS.md) and the [original planning documents](docs/planning/README.md).
