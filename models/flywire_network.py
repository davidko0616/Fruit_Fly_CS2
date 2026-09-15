import torch
import torch.nn as nn
from .sparse_layer import ConnectomeSparseLinear

class FlyWireNetwork(nn.Module):
    def __init__(self, adjacency_csr, nt_signs, input_idx, output_idx, input_dim, output_dim, num_steps=3):
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
        self.num_neurons = adjacency_csr.shape[0]
        self.num_steps = num_steps
        self.input_idx = input_idx
        self.output_idx = output_idx
        
        # Project environment observation to sensory neurons
        self.input_proj = nn.Linear(input_dim, len(input_idx))
        
        # The biological brain core (recurrent)
        self.connectome_layer = ConnectomeSparseLinear(adjacency_csr, nt_signs)
        self.activation = nn.ReLU()
        
        # Project motor neurons to environment action
        self.output_proj = nn.Linear(len(output_idx), output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize brain state: (batch_size, num_neurons)
        state = torch.zeros(batch_size, self.num_neurons, device=x.device)
        
        # 1. Inject sensory input
        sensory_activations = self.input_proj(x)
        state[:, self.input_idx] = sensory_activations
        
        # 2. Process through biological connectome (Recurrent steps)
        for _ in range(self.num_steps):
            next_state = self.connectome_layer(state)
            state = self.activation(next_state)
            
        # 3. Read out motor actions
        motor_activations = state[:, self.output_idx]
        out = self.output_proj(motor_activations)
        
        return out
