import torch
import torch.nn as nn


class LangDiscriminator(nn.Module):
    def __init__(self, in_features: int, lang_dim: int, dropout_p: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features + lang_dim, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, vis_emb: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vis_emb, lang_emb], dim=-1)
        x = self.mlp(x)
        return x
