import torchvision
import torch.nn as nn

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = num_classes
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        x = self.tanh(x)
        return x