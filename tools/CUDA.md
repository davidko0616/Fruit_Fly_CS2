# NVIDIA laptop validation

The September 21, 2026 laptop reports an NVIDIA GeForce MX570 A, 4096 MiB
VRAM, driver 560.94, and CUDA driver capability 12.6.

## Results

Both runs completed successfully on this laptop using the unchanged model:

| Measure | CUDA / MX570 A | CPU / 4 threads |
|---|---:|---:|
| Training time, 200 epochs | 63.73 s | 39.65 s |
| Test accuracy | 99.324% (147/148) | 99.324% (147/148) |
| Validation accuracy | 100% | 100% |
| Neurons / directed connections | 100 / 7,615 | 100 / 7,615 |
| Trainable parameters | 7,778 | 7,778 |
| Peak PyTorch allocated GPU memory | 66.24 MiB | N/A |
| Peak PyTorch reserved GPU memory | 68.00 MiB | N/A |

CUDA sparse forward, backward, and optimizer updates all worked. Topology and
sign checks passed throughout, and the connectome weights received nonzero
gradients. A separate process reloaded the CUDA checkpoint and reproduced its
test loss and accuracy exactly. All nine existing unit tests passed.
An `nvidia-smi` snapshot during training reported 211 MiB total GPU memory use;
this snapshot is not a measured peak.

The CPU was approximately 1.6 times faster for this small checked workload.
These are single-run measurements, not repeated performance benchmarks. Larger
circuits and full-connectome training have not been tested on this GPU.

Local metrics, learning histories, and checkpoints are in
`artifacts/synthetic/mx570a_cuda/` and `artifacts/synthetic/mx570a_cpu/`.
The installed package versions are saved in `tools/requirements-cuda-lock.txt`.

## Environment

Use Python 3.12 and an isolated `.venv`. The test uses PyTorch's official
CUDA 12.6 wheel (see https://docs.pytorch.org/get-started/previous-versions/):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools/validate_cuda.py --device cuda --output artifacts/synthetic/mx570a_cuda
.\.venv\Scripts\python.exe tools/validate_cuda.py --device cpu --output artifacts/synthetic/mx570a_cpu
.\.venv\Scripts\python.exe tools/validate_cuda.py --device cuda --evaluate artifacts/synthetic/mx570a_cuda/model.pt --output artifacts/synthetic/mx570a_cuda/reload.json
```

Choose a new output directory when repeating training; existing runs are not
overwritten. The runner uses the archived graph and dataset directly, so the
large raw connectome files are unnecessary. It runs 200 epochs, seed 42,
batch size 64, Adam at 0.001, three message-passing steps, and four CPU threads.
Checkpoint selection uses validation loss; test data is evaluated afterward.
Topology, weight signs, and finite gradients are checked throughout training.
CUDA timing is synchronized. Peak memory values cover PyTorch allocations,
not the entire driver/context or other applications.

This is a hardware capability test of the existing 100-neuron circuit, not
evidence of full-connectome capacity or a biological advantage. Per-batch
validation checks cause GPU synchronization, so timings describe this checked
training loop, not optimized GPU throughput. The original synthetic trainer
remains CPU-only; use this separate runner for the CUDA experiment.
