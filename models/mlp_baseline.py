import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128)):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.hidden_sizes = tuple(hidden_sizes)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        return self.net(x)


def matched_hidden_sizes(parameter_budget, input_dim=2, output_dim=3, minimum=16, maximum=256):
    """Closest two-layer budget; ties prefer more balanced widths, then tuple order."""
    candidates = []
    for a in range(minimum, maximum + 1):
        for b in range(minimum, maximum + 1):
            count = (input_dim + 1) * a + (a + 1) * b + (b + 1) * output_dim
            candidates.append((abs(count - parameter_budget), abs(a - b), a, b))
    _, _, a, b = min(candidates)
    return a, b
