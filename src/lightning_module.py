from torch import nn
import torch
import pytorch_lightning as pl
from simple_cnn import SimpleCNN


class SimpleCNNLightning(pl.LightningModule):
    def __init__(self, model = SimpleCNN()):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        mae = (self.mae(y_hat[0], y[0]), self.mae(y_hat[1], y[1]))
        mse = (self.mse(y_hat[0], y[0]), self.mse(y_hat[1], y[1]))
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)