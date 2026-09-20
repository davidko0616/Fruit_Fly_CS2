import unittest
import warnings

import numpy as np
import pandas as pd

from connectome.graph import ConnectomeGraph


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = ConnectomeGraph.__new__(ConnectomeGraph)

    def test_fallback_partitions_intrinsic_neurons_without_overlap(self):
        metadata = pd.DataFrame({'flow': ['intrinsic'] * 100})
        with self.assertWarns(UserWarning):
            inputs, outputs, hidden = self.graph.assign_io_neurons(metadata)
        self.assertEqual((len(inputs), len(outputs), len(hidden)), (10, 10, 80))
        np.testing.assert_array_equal(np.sort(np.r_[inputs, outputs, hidden]), np.arange(100))

    def test_small_fallback_is_disjoint_and_strict_mode_rejects_missing_roles(self):
        metadata = pd.DataFrame({'flow': ['intrinsic', 'intrinsic']})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inputs, outputs, hidden = self.graph.assign_io_neurons(metadata)
        self.assertEqual((len(inputs), len(outputs), len(hidden)), (1, 1, 0))
        with self.assertRaises(ValueError):
            self.graph.assign_io_neurons(metadata, allow_fallback=False)
        with self.assertRaises(ValueError):
            self.graph.assign_io_neurons(metadata.iloc[:1])

    def test_metadata_roles_are_respected(self):
        metadata = pd.DataFrame({'flow': ['intrinsic', 'efferent', 'afferent', None]})
        inputs, outputs, hidden = self.graph.assign_io_neurons(metadata, allow_fallback=False)
        np.testing.assert_array_equal(inputs, [2])
        np.testing.assert_array_equal(outputs, [1])
        np.testing.assert_array_equal(hidden, [0, 3])

    def test_extract_preserves_direction_and_is_independent_of_record_order(self):
        self.graph.df_meta = pd.DataFrame({'root_id': [10, 20, 30],
                                           'top_nt': ['acetylcholine', 'gaba', 'glutamate']}).set_index('root_id')
        self.graph.df_conn = pd.DataFrame({
            'pre_pt_root_id': [30, 10, 20], 'post_pt_root_id': [10, 20, 30],
            'neuropil': ['TEST'] * 3, 'syn_count': [3, 5, 7]})
        a = self.graph.extract_subgraph_by_neuropil('TEST')
        self.assertEqual(a['root_ids'], [10, 20, 30])
        self.assertEqual(a['adjacency'][0, 1], 5)
        self.assertEqual(a['adjacency'][1, 0], 0)
        np.testing.assert_array_equal(a['nt_signs'], [1, -1, -1])
        limited = self.graph.extract_subgraph_by_neuropil('TEST', max_neurons=2)
        self.graph.df_conn = self.graph.df_conn.iloc[::-1]
        reordered = self.graph.extract_subgraph_by_neuropil('TEST', max_neurons=2)
        self.assertEqual(limited['root_ids'], reordered['root_ids'])
        self.assertEqual((limited['adjacency'] != reordered['adjacency']).nnz, 0)
        with self.assertRaises(ValueError):
            self.graph.extract_subgraph_by_neuropil('MISSING')


if __name__ == '__main__':
    unittest.main()
