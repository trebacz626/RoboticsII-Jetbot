from torch import nn
import torch
import pytorch_lightning as pl
from src.model.simple_cnn import SimpleCNN


class JetBotLightning(pl.LightningModule):
    def __init__(self, backbone=SimpleCNN(), lr=1e-3):
        super().__init__()
        self.backbone = backbone
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        mae = (self.mae(y_hat[0], y[0]), self.mae(y_hat[1], y[1]))
        mse = (self.mse(y_hat[0], y[0]), self.mse(y_hat[1], y[1]))
        self.log('validation_loss', loss)
        self.log('validation_mae_forward', mae[0])
        self.log('validation_mae_left', mae[1])
        self.log('validation_mse_forward', mse[0])
        self.log('validation_mse_left', mse[1])
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        mae = (self.mae(y_hat[0], y[0]), self.mae(y_hat[1], y[1]))
        mse = (self.mse(y_hat[0], y[0]), self.mse(y_hat[1], y[1]))
        self.log('test_loss', loss)
        self.log('test_mae_forward', mae[0])
        self.log('test_mae_left', mae[1])
        self.log('test_mse_forward', mse[0])
        self.log('test_mse_left', mse[1])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
