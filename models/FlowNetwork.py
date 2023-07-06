import torch
from torch import nn
import torch.nn.functional as F

class FlowNetwork(nn.Module):
    def __init__(self, in_channels = 2, hidden_channels = 50):
        super().__init__()

        in_channels //= 2 # Since Scaling and Translation 
                          # are only performed on half of the channels

        self.s1a = nn.Linear(in_channels, hidden_channels)
        self.s1b = nn.Linear(hidden_channels, in_channels)

        self.t1a = nn.Linear(in_channels, hidden_channels)
        self.t1b = nn.Linear(hidden_channels, in_channels)

        self.s2a = nn.Linear(in_channels, hidden_channels)
        self.s2b = nn.Linear(hidden_channels, in_channels)

        self.t2a = nn.Linear(in_channels, hidden_channels)
        self.t2b = nn.Linear(hidden_channels, in_channels)
    
    def forward(self, x: torch.Tensor):
        # x_0, x_1 = x[:, ::2], x[:, 1::2]

        s = self.s1a(x[:, 0].view(-1, 1))
        s = F.relu(s)
        s = self.s1b(s)

        t = self.t1a(x[:, 0].view(-1, 1))
        t = F.relu(t)
        t = self.t1b(t)

        z0 = x[:, 1] * s.exp().view(-1) + t.view(-1)

        s = self.s2a(z0.view(-1, 1))
        s = F.relu(s)
        s = self.s2b(s)

        t = self.t2a(z0.view(-1, 1))
        t = F.relu(t)
        t = self.t2b(t)

        z1 = x[:, 0] * s.exp().view(-1) + t.view(-1)

        z = torch.concat((z0.view(-1, 1), z1.view(-1, 1)), axis=1)

        return z
