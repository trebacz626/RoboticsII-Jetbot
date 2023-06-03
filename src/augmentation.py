from kornia.augmentation import ColorJiggle, RandomBrightness, RandomContrast, RandomBoxBlur, RandomGamma, RandomMotionBlur, RandomSaturation, RandomGaussianNoise, Normalize
import torch.nn as nn
import torch
from torch import Tensor, std, mean
from model import simple_cnn
import cv2
from kornia import tensor_to_image


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, type: str = 'medium') -> None:
        super().__init__()
        

        if type == 'light':
            self.transforms = nn.Sequential(
                ColorJiggle(p = 0.05),
                RandomBrightness(p = 0.05, brightness=(0.2, 0.4)),
                RandomMotionBlur((2,3), 0.1, 0.5, p = 0.05),
                RandomSaturation(p = 0.05),
            )

        elif type == 'medium':
            self.transforms = nn.Sequential(
                ColorJiggle(p = 0.1),
                RandomBrightness(p = 0.1, brightness=(0.2, 0.4)),
                RandomContrast(p = 0.1),
                RandomBoxBlur(p = 0.1),
                RandomGamma(p = 0.1),
                RandomMotionBlur((2,3), 0.1, 0.5, p = 0.1),
                RandomSaturation(p = 0.1),
                RandomGaussianNoise(p = 0.1)
            )

        elif type == 'extreme':
            self.transforms = nn.Sequential(
                ColorJiggle(p = 0.3),
                RandomBrightness(p = 0.3),
                RandomContrast(p = 0.3),
                RandomBoxBlur(p = 0.3),
                RandomGamma(p = 0.3),
                RandomMotionBlur((2,3), 0.1, 0.5, p = 0.3),
                RandomSaturation(p = 0.3),
                RandomGaussianNoise(p = 0.3)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        norm = Normalize(mean(x, dim=(0, 2, 3)), std(x, dim=(0, 2, 3)))
        x_out = norm(x)
        x_out = self.transforms(x)
        return x_out
    


# if __name__ == "__main__":
#     model = DataAugmentation('medium')
#     img = cv2.imread("dataset/1653043549.5187616/0048.jpg")
#     input = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
#     print(input.shape)
#     output = model(input)
#     print(type(output))
#     tensor_to_image(input)
    # cv2.imshow("output", output.numpy())