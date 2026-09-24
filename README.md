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

## Record and replay neuron activity

The recorded runner uses the saved 100-neuron graph and dataset, so it does not
need the large data download above. CPU is the default; add `--device cuda` for
an NVIDIA GPU. Each run saves every sample's internal activity and prediction,
plus exact parameter versions and learning diagnostics.

```bash
python -m training.train_recorded --output artifacts/synthetic/my_recorded_run --epochs 200
python tools/verify_recording.py --run artifacts/synthetic/my_recorded_run
python -m visualization.serve --run artifacts/synthetic/my_recorded_run
```

Open **http://127.0.0.1:8765** to replay the recording. The viewer includes neuron
selection, processing-step controls, activity heatmaps and prediction inspection.
The verified 200-epoch CPU run recorded 312,904 sample-forwards and retained 99.3%
test accuracy. Full traces use about 712 MiB and remain local.
See [recording format, recovery behavior and usage](docs/RECORDING.md).

## NVIDIA laptop validation

The MX570 A (4 GB) successfully trained the existing sparse model using CUDA.
The laptop CPU was faster for the small unrecorded experiment. See
[CUDA setup and results](tools/CUDA.md).

## ZLUDA experiment

An isolated Windows ZLUDA environment was tested on the RX 7800 XT. CPU model
validation passes, but GPU sparse execution is blocked by an unsupported
`cusparseSetStream` call. See [setup, results, and reproduction instructions](tools/ZLUDA.md).

## CPU training milestone

The 100-neuron AL_R network trained on a three-class spiral dataset and achieved
**99.3% held-out test accuracy** for seed 42 (148 test examples; chance is 33.3%).
The model has 7,615 fixed directed connections and 7,778 trainable parameters.
This demonstrates learning on a synthetic task. The
[five-seed matched comparison](experiments/baseline_comparison_100/README.md)
is complete: FlyWire averages 99.46% test accuracy, versus 99.32% for matched
random and MLP baselines. The MLP reaches 95% validation accuracy sooner in all
five paired seeds. These near-ceiling results do not establish a biological
advantage. All 15 runs retain full activity traces and passed recording audits.

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

## Toy combat aiming milestone

The first recorded PPO run in the deterministic toy-combat environment improves
held-out stochastic hit rate from 35.2% to 95.3% on CPU. Every rollout decision,
reward component, outcome, and internal connectome activation is replayable from
the saved policy version. See the [protocol and results](experiments/toy_combat_aiming/README.md).

The subsequent [five-seed matched comparison](experiments/toy_combat_baseline_comparison/README.md)
uses exactly 7,913 actor parameters for FlyWire, randomized wiring, and an MLP.
Final stochastic hit rates are 90.7%, 96.1%, and 100.0%, respectively. FlyWire
trails both controls on hit rate, return, and learning-curve area in every paired
seed, so this task provides no evidence of a connectome-wiring advantage.

## Toy combat navigation milestone

The next recorded FlyWire policy learned to move out from behind obstacles,
acquire line of sight, aim, and hit a stationary target. On 256 fixed held-out
layouts, stochastic hit rate rose from 48.4% to **97.3%**, with 100% line-of-sight
acquisition and firing alignment at the final checkpoint. All 51,200 recorded CPU
decisions passed independent policy and environment replay. See the
[fixed protocol and results](experiments/toy_combat_navigation/README.md).

The following [integrated-navigation milestone](experiments/toy_combat_integrated_navigation/README.md)
removes the phase scaffold so movement, turning, waiting, and firing coexist.
Curriculum transfer and fine-tuning raise held-out stochastic hit rate from 80.9%
to **99.6%**. The final controller acquires line of sight, reaches firing
alignment, and fires in all 256 held-out layouts; all 51,200 recorded decisions
again pass replay.

## Moving-target milestone

The [moving-target curriculum](experiments/toy_combat_moving_target/README.md)
moves the target one cell after every agent decision. After a transparently
reported failed first confirmation and a predeclared zero-entropy replacement,
the independent seed-69 controller improves held-out stochastic hit rate from
79.3% to **92.2%**. Target motion occurs in every evaluation episode, averaging
33.5 moves for the final controller, and all 51,200 decisions pass replay.

## Partial-observability milestone

The [partial-observability curriculum](experiments/toy_combat_partial_observability/README.md)
hides live target coordinates behind 16 obstacles and supplies only the last
visible target position during later occlusion. The independent seed-77
controller reaches **82.8%** held-out stochastic hit rate while spending 73.5% of
decisions occluded. With identical weights and evaluation seeds, zeroing that
remembered target input reduces hit rate to **0.0%**, demonstrating that the
controller depends on remembered state. All 51,200 recorded decisions pass
independent replay.
