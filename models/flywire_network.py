import torch
import torch.nn as nn
from .sparse_layer import ConnectomeSparseLinear

class FlyWireNetwork(nn.Module):
    def __init__(self, adjacency_csr, nt_signs, input_idx, output_idx, input_dim, output_dim,
                 num_steps=3, weight_init='random'):
        """
        adjacency_csr: Scipy CSR matrix of the brain subgraph
        nt_signs: Numpy array of +1/-1 for excitatory/inhibitory
        input_idx: List of neuron indices to receive sensory input
        output_idx: List of neuron indices to provide motor output
        input_dim: Size of environment observation
        output_dim: Size of environment action space
        num_steps: Number of recurrent message-passing steps
        """
        super().__init__()
        if num_steps < 1 or len(input_idx) == 0 or len(output_idx) == 0:
            raise ValueError('Positive num_steps and nonempty input/output groups are required')
        self.num_neurons = adjacency_csr.shape[0]
        self.num_steps = num_steps
        self.input_idx = input_idx
        self.output_idx = output_idx
        
        # Project environment observation to sensory neurons
        self.input_proj = nn.Linear(input_dim, len(input_idx))
        
        # The biological brain core (recurrent)
        self.connectome_layer = ConnectomeSparseLinear(adjacency_csr, nt_signs, weight_init=weight_init)
        self.activation = nn.ReLU()
        
        # Project motor neurons to environment action
        self.output_proj = nn.Linear(len(output_idx), output_dim)
        
    def forward(self, x):
        out, _ = self.forward_with_state(x)
        return out

    def forward_with_state(self, x, previous_state=None, temporal_decay=1.0):
        """Run connectome steps from an optional cross-decision neuron state."""
        batch_size = x.size(0)
        if previous_state is None:
            state = torch.zeros(batch_size, self.num_neurons, device=x.device, dtype=x.dtype)
        else:
            if previous_state.shape != (batch_size, self.num_neurons):
                raise ValueError('Previous state shape does not match batch and neuron counts')
            if not 0 <= temporal_decay <= 1:
                raise ValueError('Temporal decay must be between zero and one')
            state = torch.tanh(previous_state) * temporal_decay
        
        # 1. Inject sensory input
        sensory_activations = self.input_proj(x)
        state = state.clone()
        state[:, self.input_idx] = sensory_activations
        
        # 2. Process through biological connectome (Recurrent steps)
        for _ in range(self.num_steps):
            next_state = self.connectome_layer(state)
            state = self.activation(next_state)
            
        # 3. Read out motor actions
        motor_activations = state[:, self.output_idx]
        out = self.output_proj(motor_activations)
        
        return out, state
