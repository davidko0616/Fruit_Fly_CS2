import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
