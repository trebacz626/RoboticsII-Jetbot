import os
from typing import List
import pytorch_lightning as pl
import torch
import pandas
from PIL import Image
from torchvision.transforms import ToTensor

class LineFollowingDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder,  run_ids: List[str], transformations=None):
        self.root_folder = root_folder
        self.run_ids = run_ids
        self.transformations = transformations
        dfs = {}
        for run_id in run_ids:
            assert os.path.exists(os.path.join(
                root_folder, run_id)), f"run_id {run_id} not found in {root_folder}"
            dfs[run_id] = pandas.read_csv(os.path.join(root_folder, f"{run_id}.csv"), names=[
                                          "image_id", "forward", "left"])
            # add column run_id
            dfs[run_id]["run_id"] = run_id
        # concat dfs in one
        self.df = pandas.concat(dfs.values())
        # reset index
        self.df = self.df.reset_index(drop=True)
        self.drop_negative_forward()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load image with PIL formate image_id to dddd
        image = Image.open(os.path.join(
            self.root_folder, row["run_id"], f"{row['image_id']:04d}.jpg"))
        image = ToTensor()(image)
        # apply transformations
        if self.transformations:
            image = self.transformations(image)
        # return image and target
        return image, torch.tensor([row["forward"], row["left"]], dtype=torch.float32)

    def drop_negative_forward(self):
        self.df = self.df[self.df["forward"] >= 0]

    def keep_random_left0_half(self):
        left = self.df[self.df["left"]**2 < 0.25]
        left = left.sample(frac=0.5)
        self.df = self.df[self.df["left"]**2 >= 0.25]
        self.df = pandas.concat([self.df, left])
