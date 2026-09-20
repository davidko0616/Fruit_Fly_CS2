"""Validate the unchanged sparse layer and FlyWire model on CPU or ZLUDA.

Run --prepare once, then --device cpu, then --device cuda through run_zluda.ps1.
Reports are written before each operation so a native crash leaves its last stage.
This is a numerical compatibility test, not a scientific learning benchmark.
"""
import argparse
import copy
import json
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "zluda"


def prepare():
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    import scipy.sparse as sp
    from connectome.graph import ConnectomeGraph

    OUT.mkdir(parents=True, exist_ok=True)
    # Read only the four fields the existing extraction method uses. This avoids
    # materializing six unused neurotransmitter probability columns for 16M rows.
    print("Reading AL_R connectivity columns...", flush=True)
    table = feather.read_table(ROOT / "data/raw/proofread_connections_783.feather",
                               columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"],
                               memory_map=True)
    region = table.filter(pc.equal(table["neuropil"], "AL_R"))
    del table
    graph = ConnectomeGraph.__new__(ConnectomeGraph)
    graph.df_conn = region.to_pandas()
    del region
    graph.df_meta = pd.read_csv(ROOT / "data/metadata/neuron_annotations.tsv", sep="\t", low_memory=False)
    graph.df_meta = graph.df_meta.drop_duplicates("root_id").set_index("root_id")
    selected = graph.extract_subgraph_by_neuropil("AL_R", max_neurons=100)
    inputs, outputs, hidden = graph.assign_io_neurons(selected["metadata"])
    sp.save_npz(OUT / "adjacency.npz", selected["adjacency"])
    np.savez(OUT / "graph.npz", signs=selected["nt_signs"], inputs=inputs,
             outputs=outputs, root_ids=np.asarray(selected["root_ids"], dtype=np.int64))
    description = {"region": "AL_R", "neurons": len(selected["root_ids"]),
                   "edges": selected["adjacency"].nnz, "inputs": inputs.tolist(),
                   "outputs": outputs.tolist(), "hidden_count": len(hidden),
                   "flow_counts": selected["metadata"]["flow"].value_counts().to_dict(),
                   "note": "Original extraction and fallback IO used unchanged; validation only."}
    (OUT / "graph.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
    print(json.dumps(description), flush=True)


def validate(device, probe_only):
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / (device + ("_probe" if probe_only else "") + ".json")
    report = {"device_requested": device, "python": platform.python_version(),
              "status": "running", "stage": "import_torch", "checks": []}

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def stage(name):
        report["stage"] = name
        save()
        print(name, flush=True)

    def passed(name):
        report["checks"].append(name)
        save()

    save()
    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from models.sparse_layer import ConnectomeSparseLinear
        from models.flywire_network import FlyWireNetwork

        torch.set_num_threads(4)
        torch.manual_seed(42)
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      hip_build=torch.version.hip)
        stage("device_discovery")
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA unavailable through ZLUDA"
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            report["devices"] = devices
            matching = [i for i, name in enumerate(devices) if "7800" in name]
            assert matching, f"RX 7800 XT not detected: {devices}"
            torch.cuda.set_device(matching[0])
            report["selected_gpu"] = devices[matching[0]]
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        stage("dense_gpu_probe" if device == "cuda" else "dense_cpu_probe")
        a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        got = a.to(device) @ a.to(device).T
        torch.testing.assert_close(got.cpu(), a @ a.T)
        passed("dense_matmul")
        if probe_only:
            report["status"] = "passed"
            save()
            return

        # A directed fixture with mixed signs independently checks orientation,
        # sparse weight gradients, input gradients, and bias against dense math.
        stage("sparse_layer_forward")
        rows = np.array([0, 0, 1, 2, 3, 4, 4, 5])
        cols = np.array([1, 2, 2, 3, 4, 0, 5, 1])
        adj = sp.csr_matrix((np.ones(8), (rows, cols)), shape=(6, 6))
        signs = np.array([1, -1, 1, -1, 1, 1], dtype=np.float32)
        layer_cpu = ConnectomeSparseLinear(adj, signs)
        layer = copy.deepcopy(layer_cpu).to(device)
        x_cpu = torch.randn(5, 6, requires_grad=True)
        x = x_cpu.detach().clone().to(device).requires_grad_(True)
        coo = adj.tocoo()
        dense_w = torch.zeros(6, 6)
        dense_w[coo.col, coo.row] = layer_cpu.weight_magnitudes.abs() * layer_cpu.edge_signs
        expected = x_cpu @ dense_w.T + layer_cpu.bias
        actual = layer(x)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
        passed("sparse_layer_forward_matches_dense_cpu")
        stage("sparse_layer_backward")
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-5)
        for name, param in layer.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all()
            torch.testing.assert_close(param.grad.cpu(), dict(layer_cpu.named_parameters())[name].grad,
                                       rtol=1e-4, atol=1e-5)
        passed("sparse_layer_input_and_parameter_gradients_match_dense_cpu")

        stage("flywire_graph_load")
        graph = np.load(OUT / "graph.npz")
        adjacency = sp.load_npz(OUT / "adjacency.npz")
        torch.manual_seed(123)
        reference = FlyWireNetwork(adjacency, graph["signs"], graph["inputs"],
                                   graph["outputs"], input_dim=2, output_dim=3, num_steps=3)
        candidate = copy.deepcopy(reference).to(device)
        observations = torch.randn(32, 2)
        targets = torch.stack((observations[:, 0], observations[:, 1],
                               observations[:, 0] - observations[:, 1]), dim=1)
        observations_gpu, targets_gpu = observations.to(device), targets.to(device)
        original_indices = candidate.connectome_layer.indices.clone()
        original_signs = candidate.connectome_layer.edge_signs.clone()
        initial_weights = candidate.connectome_layer.weight_magnitudes.detach().clone()
        cpu_optim = torch.optim.Adam(reference.parameters(), lr=0.003, foreach=False, fused=False)
        gpu_optim = torch.optim.Adam(candidate.parameters(), lr=0.003, foreach=False, fused=False)

        stage("flywire_forward")
        cpu_out, gpu_out = reference(observations), candidate(observations_gpu)
        torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=1e-4, atol=1e-5)
        report["forward_max_abs_error"] = float((gpu_out.cpu() - cpu_out).abs().max().detach())
        passed("actual_flywire_forward_matches_cpu")
        stage("flywire_backward")
        (cpu_out - targets).square().mean().backward()
        (gpu_out - targets_gpu).square().mean().backward()
        grad_errors = {}
        for name, param in candidate.named_parameters():
            expected_grad = dict(reference.named_parameters())[name].grad
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad.cpu(), expected_grad, rtol=1e-3, atol=1e-5)
            grad_errors[name] = float((param.grad.cpu() - expected_grad).abs().max())
        assert candidate.connectome_layer.weight_magnitudes.grad.abs().sum() > 0
        report["gradient_max_abs_errors"] = grad_errors
        passed("actual_flywire_all_parameter_gradients_match_cpu_and_core_gradient_nonzero")
        stage("flywire_adam_step")
        cpu_optim.step()
        gpu_optim.step()
        for name, param in candidate.named_parameters():
            torch.testing.assert_close(param.cpu(), dict(reference.named_parameters())[name],
                                       rtol=1e-3, atol=1e-4)
        passed("adam_update_matches_cpu")

        stage("flywire_training_60_steps")
        start = time.monotonic()
        losses, cpu_losses = [], []
        for step in range(60):
            for model, optim, obs, target, curve in (
                (reference, cpu_optim, observations, targets, cpu_losses),
                (candidate, gpu_optim, observations_gpu, targets_gpu, losses),
            ):
                optim.zero_grad(set_to_none=True)
                loss = (model(obs) - target).square().mean()
                assert torch.isfinite(loss), f"Nonfinite loss at step {step}"
                loss.backward()
                for param in model.parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                optim.step()
                curve.append(float(loss.detach().cpu()))
            core = candidate.connectome_layer
            assert torch.equal(core.indices, original_indices), "Topology changed"
            assert torch.equal(core.edge_signs, original_signs), "Sign assignments changed"
            weights = core.weight_magnitudes.abs() * core.edge_signs
            assert torch.all(weights * original_signs >= 0), "Weight sign violation"
            if step % 10 == 0:
                print(f"step={step} cpu_loss={cpu_losses[-1]:.6g} device_loss={losses[-1]:.6g}", flush=True)
        report.update(losses=losses, cpu_losses=cpu_losses,
                      paired_training_seconds=time.monotonic() - start)
        assert losses[-1] < 0.8 * losses[0], "Loss did not decrease by at least 20%"
        assert not torch.equal(initial_weights, candidate.connectome_layer.weight_magnitudes)
        np.testing.assert_allclose(losses, cpu_losses, rtol=0.02, atol=0.002)
        passed("60_step_training_reduces_loss_and_tracks_cpu")
        passed("fixed_topology_and_sign_constraints_preserved_after_every_update")
        if device == "cuda":
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["status"] = "passed"
        stage("complete")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        validate(args.device, args.probe_only)
