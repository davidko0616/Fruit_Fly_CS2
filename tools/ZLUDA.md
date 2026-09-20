# Isolated ZLUDA validation

**Result (2026-09-17): environment installed; GPU sparse validation failed.**
CPU forward/backward/Adam/training checks pass. The RX 7800 XT is detected and
dense GPU math works, but the existing sparse layer fails at `torch.sparse.mm()`:

```text
CUSPARSE_STATUS_NOT_SUPPORTED when calling
cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream())
```

This is a missing ZLUDA API implementation in the supplied build, not evidence
that the connectome architecture cannot learn. GPU backward and training were
not reached. No production model code was changed to bypass the failure.

This experiment runs the existing `ConnectomeSparseLinear` and `FlyWireNetwork`
without changing their implementations. It is a compatibility check, not the
planned spiral classification benchmark or evidence of biological advantage.

## Environment

- Python 3.13.0, in `.venv-zluda` (no system-site-packages).
- PyTorch 2.7.1+cu118; exact dependencies: `requirements-zluda-lock.txt`.
- ZLUDA: clean subset of the user-supplied build from
  `C:\Users\chung\Documents\zluda`. The directory also contains legacy 2024
  libraries; those unversioned DLLs are excluded from the staged copy.
- AMD TheRock SDK: `gfx110X-all-7.14.0a20260612`, supporting the gfx1101 target.
- GPU under test: Radeon RX 7800 XT.

The runtime and downloads live in `.local-zluda`; results live in
`artifacts/zluda`. These directories and the virtual environment are Git-ignored.
The launcher sets `HIP_PATH`, `PATH`, and `HIP_VISIBLE_DEVICES` for its process and
restores them afterward. On this machine HIP device 0 is integrated gfx1036
graphics and device 1 is the RX 7800 XT. Only device 1 is exposed to the test.
It does not modify the graphics driver, global Python packages, the source ZLUDA
directory, or persistent environment variables. ZLUDA may use its normal
per-user compilation cache in AppData.

`setup_zluda_runtime.py` records source DLL SHA-256 fingerprints, the exact SDK
URL, and the downloaded archive SHA-256 in `.local-zluda/runtime.json`. The SDK
length is checked against the publisher's S3 listing; the calculated SHA-256 is
a reproducibility fingerprint, not an independently verified publisher signature.

## Reproduce

Run from the repository root in PowerShell with access to the installed Python:

```powershell
py -3.13 -m venv .venv-zluda
.\.venv-zluda\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv-zluda\Scripts\python.exe -m pip install -r tools/requirements-zluda-lock.txt
.\.venv-zluda\Scripts\python.exe tools/setup_zluda_runtime.py
.\.venv-zluda\Scripts\python.exe tools/validate_zluda.py --prepare
.\.venv-zluda\Scripts\python.exe tools/validate_zluda.py --device cpu
powershell -NoProfile -File tools/run_zluda.ps1 -CheckLibraries
powershell -NoProfile -File tools/run_zluda.ps1 -ProbeOnly
powershell -NoProfile -File tools/run_zluda.ps1 -SparseApiOnly
powershell -NoProfile -File tools/run_zluda.ps1
```

The SDK download is about 3 GB compressed and uses resumable range downloads.
The PyTorch CUDA wheel is about 2.9 GB. Extraction requires additional disk space.
First GPU execution may compile kernels and take considerably longer than later runs.

## Validation coverage

1. Discover and explicitly select the RX 7800 XT rather than integrated graphics.
2. Compare a GPU matrix product against CPU.
3. Compare the unchanged sparse layer's forward result, input gradients, edge
   magnitude gradients, and bias gradients against independent dense CPU math
   using a small directed, mixed-sign graph.
4. Extract the same actual 100-neuron AL_R subgraph using the existing graph
   selection method. Read only needed connection columns to reduce memory use.
5. Compare the full network's forward output and every parameter gradient to CPU.
6. Compare an Adam update to CPU, then run 60 additional updates. Require finite
   gradients, nonzero core gradients, changed core weights, and at least a 20%
   reduction in regression loss. Compare learning curves within stated tolerances.
7. Check topology and sign constraints after every update.

The actual graph has 7,615 directed edges, with all 100 neurons annotated intrinsic.
The current fallback selects indices 0–9 as inputs and 90–99 as outputs. This is
preserved for validation, and is not a claim of biological input/output mapping.

JSON reports are saved before every stage, so a native crash leaves a report
with `status: running` and the last attempted stage. That status does **not** mean
the validation passed. Only `status: passed` indicates successful completion.

## Initial compatibility findings

- Leaving both AMD GPUs visible caused sparse library initialization errors and
  an access violation in the GPU probe. Restricting the process to HIP device 1
  resolved those failures; all entries in `cuda_check` then passed.
- PyTorch 2.9.1+cu128 passed the dense matrix probe but failed at `torch.abs()`.
  The ZLUDA trace showed seven NVIDIA ELF kernel images and no PTX for that
  module, followed by `cuModuleGetFunction(AbsFunctor...) -> CUDA_ERROR_NOT_FOUND`.
  Results are preserved as `cpu-cu128.json`, `cuda-cu128.json`, and
  `cuda-probe-cu128.json`; the trace is in `artifacts/zluda/trace/python.exe`.
- Switched the isolated environment to the official PyTorch 2.7.1+cu118 build
  to test the older CUDA packaging. This got past `torch.abs()` after first-use
  compilation, but the sparse forward pass failed at `cusparseSetStream`.
- The direct API probe independently returned `CUSPARSE_STATUS_NOT_SUPPORTED`
  (10) for both `cusparseSetStream` and `cusparseSetPointerMode`, while creating
  and destroying the sparse handle succeeded. Passing `cuda_check` alone is
  therefore insufficient to establish PyTorch sparse support.
- [PyTorch 2.7.1's sparse handle implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CuSparseHandlePool.cpp#L37)
  calls `cusparseSetStream` when obtaining the handle; this is required before
  the sparse multiplication can run.

Reports: `artifacts/zluda/cpu.json` (passed), `cuda.json` (failed),
`sparse-api.json` (exact return codes), and `graph.json` (graph selection).
For the current environment, the full GPU command and API probe are expected
to exit nonzero. Recheck after a ZLUDA build implements the required sparse APIs,
or use the same CPU/GPU validation harness when moving to native ROCm.

## Sources

- [ZLUDA Windows setup](https://zluda.readthedocs.io/latest/)
- [ZLUDA's required HIP SDK](https://zluda.readthedocs.io/latest/hip_sdk.html)
- [Official PyTorch version installation commands](https://pytorch.org/get-started/previous-versions/)
- [AMD TheRock SDK archive](https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-windows-gfx110X-all-7.14.0a20260612.tar.gz)
