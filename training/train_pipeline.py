import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm_soc_model import LSTMSOCEstimator
from training.dataset_loader import load_dataloaders
from training.trainer import Trainer


def train_model(dataset_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_dataloaders(dataset_path)

    model = LSTMSOCEstimator().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn, device)

    best_val = float("inf")

    for epoch in range(50):

        train_loss = trainer.train_epoch(train_loader)

        val_loss = trainer.validate(val_loader)

        print(f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

        if val_loss < best_val:

            best_val = val_loss

            torch.save(model.state_dict(), "models/best_model.pt")

    return model