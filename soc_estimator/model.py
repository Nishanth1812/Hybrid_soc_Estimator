from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "input_features": 5,
    "lstm_hidden_1": 64,
    "lstm_hidden_2": 32,
    "dense_units": 16,
    "dropout": 0.2,
}


class LSTMSOCEstimator(nn.Module):
    def __init__(
        self, input_features: int | None = None,
        current_soc_feature_index: int | None = None,
        max_soc_correction: float = 0.03,
    ):
        super().__init__()
        input_size = MODEL_CONFIG["input_features"] if input_features is None else input_features
        self.physics_informed = input_size == 6  # Retain loading of existing six-input checkpoints.
        self.current_soc_feature_index = current_soc_feature_index
        self.max_soc_correction = max_soc_correction
        if current_soc_feature_index is not None and not 0 <= current_soc_feature_index < input_size:
            raise ValueError("Current SOC feature index must identify an input feature")
        hidden_1 = MODEL_CONFIG["lstm_hidden_1"]
        hidden_2 = MODEL_CONFIG["lstm_hidden_2"]
        dense_units = MODEL_CONFIG["dense_units"]
        dropout = MODEL_CONFIG["dropout"]

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_2, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()
        if self.physics_informed or current_soc_feature_index is not None:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if self.current_soc_feature_index is not None:
            baseline_soc = x[:, -1, self.current_soc_feature_index : self.current_soc_feature_index + 1]
            return torch.clamp(
                baseline_soc + self.max_soc_correction * torch.tanh(out), 0.0, 1.0
            )
        if self.physics_informed:
            baseline_soc = x[:, -1, -1:]
            return torch.clamp(baseline_soc + 0.005 * torch.tanh(out), 0.0, 1.0)
        return self.sigmoid(out)
