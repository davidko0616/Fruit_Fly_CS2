import torch
import torch.nn as nn
import numpy as np

class ConnectomeSparseLinear(nn.Module):
    def __init__(self, adjacency_csr, nt_signs, weight_init='random'):
        super().__init__()
        if adjacency_csr.shape[0] != adjacency_csr.shape[1] or adjacency_csr.shape[0] < 1:
            raise ValueError('Connectome adjacency must be nonempty and square')
        if len(nt_signs) != adjacency_csr.shape[0] or not np.isin(nt_signs, [-1, 1]).all():
            raise ValueError('Each neuron must have a sign of +1 or -1')
        
        # Convert CSR to COO for PyTorch sparse tensors
        coo = adjacency_csr.tocoo()
        
        # Indices of biological connections: shape (2, num_edges)
        # Note: transposed because PyTorch sparse.mm expects (out, in) matrix
        indices = torch.from_numpy(np.stack([coo.col, coo.row]).astype(np.int64))
        self.register_buffer('indices', indices)
        
        # Initial learnable weight magnitudes
        # We initialize randomly, but the topology is strictly fixed by `indices`
        if weight_init == 'random':
            magnitudes = torch.randn(coo.nnz) * 0.1
        elif weight_init == 'normalized_synapse_count':
            counts = np.asarray(coo.data, dtype=np.float32)
            if not np.isfinite(counts).all() or (counts <= 0).any():
                raise ValueError('Synapse counts must be finite and positive')
            # Normalize incoming absolute weights to sum to 1 per target.
            # This preserves within-target count ratios without amplifying
            # activations purely because a target has a large in-degree.
            incoming = np.bincount(coo.col, weights=counts, minlength=coo.shape[0])
            magnitudes = torch.from_numpy((counts / incoming[coo.col]).astype(np.float32))
        else:
            raise ValueError(f'Unknown weight initialization: {weight_init}')
        self.weight_magnitudes = nn.Parameter(magnitudes)
        
        # Neurotransmitter constraints
        # nt_signs is (N,) representing +1 (excitatory) or -1 (inhibitory) for each source neuron
        # coo.row contains the source (presynaptic) neuron index for each edge
        edge_signs = torch.tensor(nt_signs[coo.row], dtype=torch.float32)
        self.register_buffer('edge_signs', edge_signs)
        
        self.out_features = adjacency_csr.shape[0]
        self.in_features = adjacency_csr.shape[1]
        
        # Bias term for every neuron
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        
    def forward(self, x):
        # x shape: (batch_size, in_features)
        
        # 1. Enforce biological signs: weight = |magnitude| * sign
        actual_weights = torch.abs(self.weight_magnitudes) * self.edge_signs
        
        # 2. Construct the sparse weight matrix W
        # W shape: (out_features, in_features)
        W_sparse = torch.sparse_coo_tensor(
            self.indices, 
            actual_weights, 
            size=(self.out_features, self.in_features)
        )
        
        # 3. Sparse matrix multiplication
        # W_sparse @ x^T  -> shape (out_features, batch_size)
        out = torch.sparse.mm(W_sparse, x.t())
        
        # Transpose back and add bias -> shape (batch_size, out_features)
        return out.t() + self.bias
