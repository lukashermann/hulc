import torch
import torch.nn as nn


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        language_features: int,
        hidden_size: int,
        out_features: int,
        word_dropout_p: float,
        activation_function: str,
    ):
        super().__init__()
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Dropout(word_dropout_p),
            nn.Linear(in_features=language_features, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
