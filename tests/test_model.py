import unittest

import numpy as np
import scipy.sparse as sp
import torch

from models.flywire_network import FlyWireNetwork
from models.sparse_layer import ConnectomeSparseLinear


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.adjacency = sp.csr_matrix(([3., 1., 2.], ([0, 1, 2], [2, 2, 0])), shape=(3, 3))
        self.signs = np.array([1, -1, 1])

    def test_normalized_counts_produce_expected_directed_signed_output(self):
        layer = ConnectomeSparseLinear(self.adjacency, self.signs, 'normalized_synapse_count')
        x = torch.tensor([[2., 4., 6.]], requires_grad=True)
        # 2 -> 0 has normalized weight 1. 0 -> 2 has +3/4;
        # 1 -> 2 has -1/4. No edges enter neuron 1.
        torch.testing.assert_close(layer(x), torch.tensor([[6., 0., 0.5]]))
        layer(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[0.75, -0.25, 1.]]))

    def test_updates_preserve_edges_and_signs(self):
        layer = ConnectomeSparseLinear(self.adjacency, self.signs)
        indices = layer.indices.clone()
        signs = layer.edge_signs.clone()
        initial = layer.weight_magnitudes.detach().clone()
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            (layer(torch.ones(2, 3)) - 2).square().mean().backward()
            optimizer.step()
            self.assertTrue(torch.equal(indices, layer.indices))
            self.assertTrue(torch.equal(signs, layer.edge_signs))
            self.assertTrue(torch.all((layer.weight_magnitudes.abs() * signs) * signs >= 0))
        self.assertFalse(torch.equal(initial, layer.weight_magnitudes))

    def test_forward_shape_and_initialization_validation(self):
        model = FlyWireNetwork(self.adjacency, self.signs, [0], [2], 2, 3)
        self.assertEqual(model(torch.randn(8, 2)).shape, (8, 3))
        with self.assertRaises(ValueError):
            ConnectomeSparseLinear(self.adjacency, self.signs, 'unknown')
        with self.assertRaises(ValueError):
            ConnectomeSparseLinear(self.adjacency, [0, 1, 1])
        with self.assertRaises(ValueError):
            FlyWireNetwork(self.adjacency, self.signs, [], [2], 2, 3)


if __name__ == '__main__':
    unittest.main()
