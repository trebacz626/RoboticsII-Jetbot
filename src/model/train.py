import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomRotation, RandomPerspective, ColorJitter, RandomApply, \
    GaussianBlur, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize

import sys
sys.path.append(".")
from src.data.datamodule import LineFollowingDataModule
from src.model.SqueezedSqueezeNet import SqueezedSqueezeNet
from src.model.simple_cnn import SimpleCNN
import argparse

from src.model.lightning_module import JetBotLightning

parser = argparse.ArgumentParser(description='Process model arguments.')
parser.add_argument('--model', type=str,
                    default="SimpleCNN", help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--onnx', type=bool, default=False, help='onnx')
parser.add_argument('--resolution', type=int, default=64, help='resolution')
parser.add_argument('--precision', type=int, default=16, help='precision')
parser.add_argument('--lr_cycles', type=float, default=1, help='precision')
parser.add_argument('--transformation_probability', type=float, default=0.5, help='precision')


def to_onnx(model: JetBotLightning):
    dummy_input = torch.randn(1, 3, 64, 64)
    # predict dummy input
    output = model.backbone(dummy_input)
    torch.onnx.export(model.backbone, dummy_input,
                      f"{args.model}.onnx", verbose=True)


def get_model(backbone_name):
    if backbone_name == "SimpleCNN":
        backbone = SimpleCNN()
    elif backbone_name == "SqueezeNet":
        backbone = SqueezedSqueezeNet(num_classes=2)
    else:
        raise NotImplementedError(f"Backbone {backbone_name} not implemented")
    return JetBotLightning(backbone, lr=args.lr, max_epochs=args.epochs, lr_cycles=args.lr_cycles)

if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    train_run_ids = ["1652875851.3497071", "1652875901.3107166", "1652876013.741493", "1652876206.2541456",
                     "1652876485.8123376", "1652959186.4507334", "1652959347.972946", "1653042695.4914637",
                     "1653042775.5213027", "1653043202.5073502"]
    vaild_run_ids = ["1653043345.3415065",
                     "1653043428.8546412", "1653043549.5187616"]

    train_transformations = torchvision.transforms.Compose(
        [Resize((args.resolution, args.resolution)),
         RandomRotation((-4,4)),
         RandomPerspective(0.05, args.transformation_probability),
         RandomApply([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=args.transformation_probability),
         RandomApply([GaussianBlur(3, sigma=(0.1, 2.0))], p=args.transformation_probability),
         RandomAdjustSharpness(0.1, p=args.transformation_probability),
         RandomAutocontrast(p=args.transformation_probability),
         RandomEqualize(p=args.transformation_probability),
         ToTensor()])
    valid_transformations = torchvision.transforms.Compose(
        [Resize((args.resolution, args.resolution)), ToTensor()])

    data_module = LineFollowingDataModule("./dataset", train_run_ids, vaild_run_ids, train_transformations,
                                          valid_transformations, batch_size=args.batch_size, num_workers=6)
    
    model = get_model(args.model)
    trainer = pl.Trainer(accelerator="auto",
                         precision=args.precision,
                         max_epochs=args.epochs, num_sanity_val_steps=2,
                         auto_lr_find=True, logger=WandbLogger(project="jetbot", name=args.model, config=args),
                         log_every_n_steps=1,
                         callbacks=[
                             ModelCheckpoint(
                                 dirpath="./checkpoints",
                                 filename="base-{epoch:02d}-{validation_loss:.2f}",
                                 monitor="validation_loss",
                             ),
                             LearningRateMonitor(),
                             EarlyStopping(monitor="validation_loss", patience=3, verbose=True)
                         ],
                         )
    # trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    if args.onnx:
        to_onnx(model)
