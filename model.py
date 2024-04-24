import torch
from torch.distributions import Categorical
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        layers = [nn.Linear(neurons[0], neurons[1])]
        for i, o in zip(neurons[1:-1], neurons[2:]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(i, o))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class Policy(MLP):
    def __init__(self, neurons):
        super().__init__(neurons)

    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float)
        logits = self(obs)
        dist = Categorical(logits=logits)
        return dist.sample().item()

class Value(MLP):
    def __init__(self, neurons):
        super().__init__(neurons)
        