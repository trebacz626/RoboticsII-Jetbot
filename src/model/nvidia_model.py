from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm([3, 244, 244])
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 60)
        self.fc2 = nn.Linear(60, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.tanh(x)
        return x

    def __str__(self) -> str:
        return "NvidiaModel"