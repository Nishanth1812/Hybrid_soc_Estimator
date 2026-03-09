import torch


class Trainer:

    def __init__(self, model, optimizer, loss_fn, device):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device


    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0

        for X, y in dataloader:

            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(X)

            loss = self.loss_fn(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def validate(self, dataloader):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for X, y in dataloader:

                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)

                loss = self.loss_fn(pred, y)

                total_loss += loss.item()

        return total_loss / len(dataloader)