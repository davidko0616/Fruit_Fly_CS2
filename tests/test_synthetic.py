import unittest

import numpy as np

from training.synthetic import make_spiral, stratified_split


class SyntheticDataTests(unittest.TestCase):
    def test_deterministic_data_and_disjoint_balanced_splits(self):
        x, y = make_spiral(1000, seed=42)
        x_again, y_again = make_spiral(1000, seed=42)
        np.testing.assert_array_equal(x, x_again)
        np.testing.assert_array_equal(y, y_again)
        self.assertEqual(x.shape, (1000, 2))
        self.assertTrue(np.isfinite(x).all())
        splits = stratified_split(y, seed=43)
        combined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(1000))
        for indices in splits.values():
            counts = np.bincount(y[indices])
            self.assertLessEqual(counts.max() - counts.min(), 2)
            self.assertEqual(len(counts), 3)
        same = stratified_split(y, seed=43)
        for name in splits:
            np.testing.assert_array_equal(splits[name], same[name])

    def test_invalid_splits_fail(self):
        with self.assertRaises(ValueError):
            stratified_split(np.arange(30) % 3, 0.6, 0.6)
        with self.assertRaises(ValueError):
            make_spiral(3)


if __name__ == '__main__':
    unittest.main()
