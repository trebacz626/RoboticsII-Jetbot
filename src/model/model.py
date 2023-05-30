import pytorch_lightning as pl
import timm
import torch



class TurnNetwork(torch.nn.Module):
    def __init__(self,  num_outputs=3):
        super().__init__()
        
        self.output = torch.nn.Linear(self.model.classifier.in_features, num_outputs)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)



class LineFollowingNet(pl.LightningModule):
    def __init__(self, num_outputs=2, lr=1e-4):
        super().__init__()
        self.model = TurnNetwork(num_outputs)
        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.mae = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()

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
        self.log('train_mae', mae)
        self.log('train_mse', mse)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
