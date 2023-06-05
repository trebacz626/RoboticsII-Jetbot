import os
from typing import List

import pandas
import pytorch_lightning as pl
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional


#create pl dataset that has image as input and 2 float values as target
class LineFollowingDataset(torch.utils.data.Dataset):
    def __init__(self,root_folder,  run_ids:List[str], transformations=None):
        self.root_folder = root_folder
        self.run_ids = run_ids
        self.transformations = transformations
        dfs = {}
        for run_id in run_ids:
            assert os.path.exists(os.path.join(root_folder, run_id)), f"run_id {run_id} not found in {root_folder}"
            dfs[run_id] = pandas.read_csv(os.path.join(root_folder, f"{run_id}.csv"), names=["image_id", "forward", "left"])
            #add column run_id
            dfs[run_id]["run_id"] = run_id
        #concat dfs in one
        self.df = pandas.concat(dfs.values())
        #reset index
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #load image with PIL formate image_id to dddd
        image = Image.open(os.path.join(self.root_folder, row["run_id"], f"{row['image_id']:04d}.jpg"))
        image = ImageOps.grayscale(image)
        image = functional.equalize(image)

        #apply transformations
        if self.transformations:
            image = self.transformations(image)
        #return image and target
        return image, torch.tensor([row["forward"], row["left"]], dtype=torch.float32)

