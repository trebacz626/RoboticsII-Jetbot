import torch
import torch.nn as nn
from kornia.augmentation import (
    ColorJiggle,
    Normalize,
    RandomBoxBlur,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    RandomGaussianNoise,
    RandomMotionBlur,
    RandomSaturation,
    Resize,
)
from kornia.utils import image_to_tensor
from torch import Tensor
from torchvision.transforms import ToTensor


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, prob_trans, size: tuple, mean = 0, sd = 0, standarization = False) -> None:
        super().__init__()
        self.mean = mean
        self.sd = sd
        self.standarization = standarization

        self.transforms = nn.Sequential(
            Resize(size=size, keepdim=True),
            ColorJiggle(0, 1, 1, 1, p=prob_trans, keepdim=True),
            RandomBrightness(p=prob_trans, brightness=(0.7, 0.9), keepdim=True),
            RandomContrast((0.6, 0.9), p=prob_trans, keepdim=True),
            RandomBoxBlur(p=prob_trans, keepdim=True),
            RandomGamma((0.4, 0.8), (0.6, 1), p=prob_trans, keepdim=True),
            RandomMotionBlur((2, 6), 10, 1, p=prob_trans, keepdim=True),
            RandomSaturation((0.4, 0.9), p=prob_trans, keepdim=True),
            RandomGaussianNoise(0.0, 0.005, p=prob_trans, keepdim=True),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        if self.standarization:
            norm = Normalize(self.mean, self.sd)
            x_out = self.transforms(norm(x))
        else:
            x_out = self.transforms(x)
        return x_out
    
    # def setParams(self, mean, sd):
    #     assert (self.mean == None and self.sd == None), f"mean: {self.mean}, sd:{self.sd}"
    #     self.mean = mean
    #     self.sd = sd

