import torch
import numpy as np

from models.lstm_soc_model import LSTMSOCEstimator
from evaluation.metrics import mae, rmse, max_error


def evaluate(model_path, test_loader, device):

    model = LSTMSOCEstimator().to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            pred = model(X)

            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    results = {

        "MAE": mae(targets, preds),
        "RMSE": rmse(targets, preds),
        "MaxError": max_error(targets, preds)
    }

    return results