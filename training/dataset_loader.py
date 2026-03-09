import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SOCDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


def load_dataloaders(dataset_path, batch_size=64):

    X_train = np.load(f"{dataset_path}/X_train.npy")
    y_train = np.load(f"{dataset_path}/y_train.npy")

    X_val = np.load(f"{dataset_path}/X_val.npy")
    y_val = np.load(f"{dataset_path}/y_val.npy")

    X_test = np.load(f"{dataset_path}/X_test.npy")
    y_test = np.load(f"{dataset_path}/y_test.npy")

    train_dataset = SOCDataset(X_train, y_train)
    val_dataset = SOCDataset(X_val, y_val)
    test_dataset = SOCDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader