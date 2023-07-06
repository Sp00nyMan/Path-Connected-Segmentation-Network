import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_features: int, out_features=1, hidden_neurons=130, convex=False):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, out_features)

        self.convex = convex
        if self.convex:
            self.skip1 = nn.Linear(in_features, self.fc2.out_features, bias=False)
            self.skip2 = nn.Linear(in_features, self.fc3.out_features, bias=False)
    
    def forward(self, x: torch.Tensor):
        x_in = x.clone()

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.convex: x += self.skip1(x_in)
        x = F.relu(x)

        x = self.fc3(x)
        if self.convex: x += self.skip2(x_in)
        x = F.sigmoid(x)

        return x

    def step(self):
        if not self.convex:
            raise NotImplementedError("Only on convex networks.")
        
        with torch.no_grad():
            F.relu(self.skip1.weight.data, True)
            F.relu(self.skip2.weight.data, True)