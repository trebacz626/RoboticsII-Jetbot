import sys

sys.path.append("./")
import torch

from src.model.lightning_module import JetBotLightning
from src.model.SqueezedSqueezeNet import SqueezedSqueezeNet


def to_onnx(model: JetBotLightning):
    dummy_input = torch.randn(1, 1, 64, 64)
    # predict dummy input
    output = model.backbone(dummy_input)
    torch.onnx.export(model.backbone, dummy_input,
                      f"SqueezeNet.onnx", verbose=True)
    
if __name__ == "__main__":
    backbone = SqueezedSqueezeNet(num_classes=2)
    model = JetBotLightning.load_from_checkpoint("checkpoints/base-epoch=06-validation_loss=0.09.ckpt", backbone = backbone)
    to_onnx(model)
