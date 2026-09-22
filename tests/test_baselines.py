import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.random_sparse import randomize_destinations, create_random_sparse_network
from models.mlp_baseline import MLPBaseline, matched_hidden_sizes
from training.mlp_recording import MLPActivityRecorder
from training.train_recorded import load_version


class BaselineTests(unittest.TestCase):
    def test_rewiring_matches_source_counts_signs_and_self_loops(self):
        adjacency = sp.csr_matrix(np.array([[2, 1, 3, 0], [0, 0, 4, 5], [7, 0, 0, 1], [0, 9, 0, 0]], dtype=float))
        signs = np.array([1, -1, -1, 1])
        result = randomize_destinations(adjacency, 42)
        self.assertEqual(result.nnz, adjacency.nnz)
        self.assertTrue(result.has_canonical_format)
        np.testing.assert_array_equal(np.diff(result.indptr), np.diff(adjacency.indptr))
        np.testing.assert_array_equal(result.diagonal(), adjacency.diagonal())
        for i in range(4):
            np.testing.assert_array_equal(np.sort(result.getrow(i).data), np.sort(adjacency.getrow(i).data))
        self.assertEqual((randomize_destinations(adjacency, 42) != result).nnz, 0)
        model = create_random_sparse_network(4, adjacency.nnz, [0], [3], 2, 3,
                                            nt_signs=signs, reference_adjacency=adjacency, seed=42)
        coo = result.tocoo()
        np.testing.assert_array_equal(model.connectome_layer.edge_signs.numpy(), signs[coo.row])
        expected = coo.data / np.bincount(coo.col, weights=coo.data, minlength=4)[coo.col]
        np.testing.assert_allclose(model.connectome_layer.weight_magnitudes.detach(), expected)

    def test_unique_sampling_capacity_and_global_rng_isolation(self):
        np.random.seed(19)
        before = np.random.get_state()
        model = create_random_sparse_network(5, 20, [0], [4], 2, 3, nt_signs=np.ones(5))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        indices = model.connectome_layer.indices.numpy()
        self.assertEqual(np.unique(indices, axis=1).shape[1], 20)
        self.assertFalse(np.any(indices[0] == indices[1]))
        with self.assertRaises(ValueError):
            create_random_sparse_network(5, 21, [0], [4], 2, 3, nt_signs=np.ones(5))

    def test_parameter_budget(self):
        sizes = matched_hidden_sizes(7778)
        self.assertEqual(sizes, (114, 63))
        self.assertEqual(sum(p.numel() for p in MLPBaseline(2, 3, sizes).parameters()), 7779)

    def test_mlp_capture_preserves_outputs_gradients_rng_and_layer_values(self):
        torch.manual_seed(17)
        model = MLPBaseline(2, 3, (5, 4))
        control = copy.deepcopy(model)
        x, y = torch.randn(3, 2), torch.tensor([0, 1, 2])
        expected = control(x)
        F.cross_entropy(expected, y).backward()
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            with MLPActivityRecorder(directory, model, [], {}, max_buffer_samples=3) as recorder:
                recorder.save_weights(0)
                actual = recorder.forward(x, y, [0, 1, 2], split='train', phase='initial', epoch=0, version=0)
                F.cross_entropy(actual, y).backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            for a, b in zip(model.parameters(), control.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            with np.load(directory / 'chunks/000000.npz') as data, torch.no_grad():
                pre1 = control.net[0](x); post1 = pre1.relu()
                pre2 = control.net[2](post1); post2 = pre2.relu()
                np.testing.assert_array_equal(data['hidden_preactivations'], torch.cat([pre1, pre2], 1).numpy())
                np.testing.assert_array_equal(data['hidden_activations'], torch.cat([post1, post2], 1).numpy())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            load_version(model, directory, 0)
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
            self.assertEqual(json.loads((directory / 'manifest.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
