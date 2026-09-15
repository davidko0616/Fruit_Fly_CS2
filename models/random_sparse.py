import scipy.sparse as sp
import numpy as np
from .flywire_network import FlyWireNetwork

def create_random_sparse_network(num_neurons, num_edges, input_idx, output_idx, input_dim, output_dim, num_steps=3):
    """
    Creates a FlyWireNetwork but with completely random topology and neurotransmitter signs,
    matching the exact number of neurons and connections as the biological graph.
    """
    # Generate random directed edges
    rows = np.random.randint(0, num_neurons, size=num_edges)
    cols = np.random.randint(0, num_neurons, size=num_edges)
    weights = np.ones(num_edges)
    
    random_adj = sp.csr_matrix((weights, (rows, cols)), shape=(num_neurons, num_neurons))
    
    # Random neurotransmitter signs (assume ~50/50 excitatory/inhibitory for baseline)
    random_signs = np.random.choice([1.0, -1.0], size=num_neurons)
    
    return FlyWireNetwork(
        adjacency_csr=random_adj,
        nt_signs=random_signs,
        input_idx=input_idx,
        output_idx=output_idx,
        input_dim=input_dim,
        output_dim=output_dim,
        num_steps=num_steps
    )
