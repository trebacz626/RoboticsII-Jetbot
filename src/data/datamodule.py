#create datamodule
import torch
import pytorch_lightning as pl
from src.data.dataset import LineFollowingDataset


class LineFollowingDataModule(pl.LightningDataModule):
    def __init__(self, root_folder, train_run_ids, valid_run_ids, train_transformations, valid_transformations, batch_size=32, num_workers=0):
        super().__init__()
        self.root_folder = root_folder
        self.train_run_ids = train_run_ids
        self.valid_run_ids = valid_run_ids
        self.train_transformations = train_transformations
        self.valid_transformations = valid_transformations
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(LineFollowingDataset(self.root_folder, self.train_run_ids, self.train_transformations), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(LineFollowingDataset(self.root_folder, self.valid_run_ids, self.valid_transformations), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(LineFollowingDataset(self.root_folder, self.valid_run_ids, self.valid_transformations), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
