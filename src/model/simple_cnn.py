from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def onnx_export():
    #to onnx with imput 224x224
    dummy_input = torch.randn(1, 3, 64, 64)

    model = SimpleCNN()
    # predict dummy input
    output = model(dummy_input)
    print(output)

    torch.onnx.export(model, dummy_input, "random_not_trained_cnn.onnx", verbose=True)


