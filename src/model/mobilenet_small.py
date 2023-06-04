import torchvision
import torch.nn as nn

class MobileNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[3] = nn.Linear(1024, 2)

    def forward(self, x):
        return self.backbone(x)

    def __str__(self) -> str:
        return "MOBILENET_V3_SMALL"