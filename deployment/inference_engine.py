import torch
import numpy as np

from deployment.buffer_manager import BufferManager
from deployment.scaler_loader import load_scaler
from models.lstm_soc_model import LSTMSOCEstimator


class InferenceEngine:

    def __init__(self, model_path, scaler_path, seq_len=100):

        self.model = LSTMSOCEstimator()

        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        self.scaler = load_scaler(scaler_path)

        self.buffer = BufferManager(seq_len)


    def step(self, v, i, t):

        x = self.scaler.transform([[v, i, t]])[0]

        self.buffer.add(*x)

        if not self.buffer.is_ready():

            return None

        seq = self.buffer.get_sequence()

        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():

            soc = self.model(seq).item()

        return soc