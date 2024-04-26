import torch
import numpy as np
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

    @torch.no_grad
    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float)
        logits = self(obs)
        dist = Categorical(logits=logits)
        return dist.sample().item()
    
    def get_prob(self, obs, act, compute_grads=True):
        obs = torch.as_tensor(obs, dtype=torch.float)
        act = torch.as_tensor(act, dtype=torch.int64)
        if compute_grads:
            logits = self(obs)
        else:
            with torch.no_grad():
                logits = self(obs)
        dist = Categorical(logits=logits)
        return torch.gather(dist.probs, 1, act.unsqueeze(1)).squeeze()

class Value(MLP):
    def __init__(self, neurons):
        super().__init__(neurons)

    @torch.no_grad
    def get_value(self, obs):
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float)
        return self(obs).squeeze().tolist()