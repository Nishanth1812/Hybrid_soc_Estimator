import torch
import numpy as np

from models.lstm_soc_model import LSTMSOCEstimator
from evaluation.metrics import calculate_metrics


def predict(model_path, test_loader, device):

    model = LSTMSOCEstimator().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            pred = model(X)

            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    return targets, preds


def evaluate(model_path, test_loader, device):

    targets, preds = predict(model_path, test_loader, device)

    return calculate_metrics(targets, preds)
