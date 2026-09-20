"""Deterministic spiral data and stratified train/validation/test splits."""
import numpy as np


def make_spiral(n_samples, n_classes=3, noise=0.15, seed=42):
    if n_classes < 2 or n_samples < n_classes * 10 or noise < 0:
        raise ValueError('Need at least 10 samples per class, >=2 classes, and nonnegative noise')
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label in range(n_classes):
        count = n_samples // n_classes + int(label < n_samples % n_classes)
        radius = rng.uniform(0.05, 1.0, count)
        angle = radius * 4.0 + label * 2.0 * np.pi / n_classes + rng.normal(0, noise, count)
        features.append(np.column_stack((radius * np.sin(angle), radius * np.cos(angle))))
        labels.append(np.full(count, label, dtype=np.int64))
    return np.concatenate(features).astype(np.float32), np.concatenate(labels)


def stratified_split(labels, validation_fraction=0.15, test_fraction=0.15, seed=42):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError('Validation and test fractions must be positive and leave training data')
    rng = np.random.default_rng(seed)
    splits = {'train': [], 'validation': [], 'test': []}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        nv, nt = int(len(indices) * validation_fraction), int(len(indices) * test_fraction)
        if min(nv, nt, len(indices) - nv - nt) < 1:
            raise ValueError('Each class needs at least one example in every split')
        splits['validation'].extend(indices[:nv])
        splits['test'].extend(indices[nv:nv + nt])
        splits['train'].extend(indices[nv + nt:])
    return {name: rng.permutation(np.asarray(indices, dtype=np.int64)) for name, indices in splits.items()}
