# models/lstm_soc_model.py

import torch
import torch.nn as nn
from config_model import MODEL_CONFIG


class LSTMSOCEstimator(nn.Module):

    def __init__(self):

        super(LSTMSOCEstimator, self).__init__()

        input_size = MODEL_CONFIG["input_features"]

        hidden_1 = MODEL_CONFIG["lstm_hidden_1"]
        hidden_2 = MODEL_CONFIG["lstm_hidden_2"]

        dense_units = MODEL_CONFIG["dense_units"]

        dropout = MODEL_CONFIG["dropout"]


        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)


        self.lstm2 = nn.LSTM(
            input_size=hidden_1,
            hidden_size=hidden_2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(dropout)


        self.fc1 = nn.Linear(hidden_2, dense_units)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.sigmoid(out)

        return out