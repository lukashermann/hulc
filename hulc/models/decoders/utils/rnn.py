import torch
import torch.nn as nn


def rnn_decoder(in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float) -> torch.nn.Module:
    return nn.RNN(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity="relu",
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )


def lstm_decoder(in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )


def gru_decoder(in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float) -> torch.nn.Module:
    return nn.GRU(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )


def mlp_decoder(in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float) -> torch.nn.Module:
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
    )
