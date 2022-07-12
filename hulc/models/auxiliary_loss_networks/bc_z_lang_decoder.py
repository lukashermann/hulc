import torch
import torch.nn as nn


class BCZLangDecoder(nn.Module):
    def __init__(self, in_features: int, lang_dim: int):
        super().__init__()
        # include proprio info???
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=lang_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
