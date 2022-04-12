from typing import Dict

import torch
import torch.nn as nn


class StateDecoder(nn.Module):
    def __init__(self, visual_features: int, n_state_obs: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=visual_features, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=n_state_obs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
