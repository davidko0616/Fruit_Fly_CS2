"""Random controls with unique directed edges and explicit matching constraints."""
import scipy.sparse as sp
import numpy as np
from .flywire_network import FlyWireNetwork


def randomize_destinations(adjacency, seed):
    """Preserve source degrees, self-loops, and each source's count multiset.

    Destination degrees and input/output reachability are not conditioned on.
    Signs remain attached to the same source neurons in the caller.
    """
    adjacency = adjacency.tocsr(copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    n = adjacency.shape[0]
    if adjacency.shape != (n, n) or not np.isfinite(adjacency.data).all() or (adjacency.data <= 0).any():
        raise ValueError('Expected a square adjacency with positive finite counts')
    rng = np.random.default_rng(seed)
    rows, cols, counts = [], [], []
    for source in range(n):
        start, stop = adjacency.indptr[source:source + 2]
        targets, values = adjacency.indices[start:stop], adjacency.data[start:stop]
        off = targets != source
        destinations = rng.choice(np.delete(np.arange(n), source), size=int(off.sum()), replace=False)
        rows.extend([source] * len(destinations)); cols.extend(destinations)
        counts.extend(rng.permutation(values[off]))
        if (~off).any():
            rows.append(source); cols.append(source); counts.append(values[~off][0])
    return sp.csr_matrix((counts, (rows, cols)), shape=adjacency.shape, dtype=adjacency.dtype)


def create_random_sparse_network(num_neurons, num_edges, input_idx, output_idx,
                                 input_dim, output_dim, num_steps=3, *,
                                 nt_signs, seed=0, reference_adjacency=None,
                                 weight_init='normalized_synapse_count'):
    """Require explicit source signs; never invent a 50/50 sign distribution."""
    signs = np.asarray(nt_signs)
    if signs.shape != (num_neurons,) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Provide one +/-1 sign per neuron')
    if reference_adjacency is not None:
        if reference_adjacency.shape != (num_neurons, num_neurons) or reference_adjacency.nnz != num_edges:
            raise ValueError('Reference dimensions or edge budget disagree')
        adjacency = randomize_destinations(reference_adjacency, seed)
    else:
        if num_neurons < 2 or not 0 <= num_edges <= num_neurons * (num_neurons - 1):
            raise ValueError('Edge budget exceeds unique non-self pairs')
        pairs = np.random.default_rng(seed).choice(num_neurons * (num_neurons - 1), num_edges, replace=False)
        rows, reduced = pairs // (num_neurons - 1), pairs % (num_neurons - 1)
        cols = reduced + (reduced >= rows)
        adjacency = sp.csr_matrix((np.ones(num_edges), (rows, cols)), shape=(num_neurons, num_neurons))
    return FlyWireNetwork(adjacency, signs, input_idx, output_idx, input_dim, output_dim,
                          num_steps, weight_init)
