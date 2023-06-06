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
from src.model.nvidia_model import NvidiaModel
from src.model.mobilenet_small import MobileNetSmall
from src.model.SqueezeNet import SqueezeNet
import argparse
from src.model.augmentation import DataAugmentation

from src.model.lightning_module import JetBotLightning

parser = argparse.ArgumentParser(description='Process model arguments.')
parser.add_argument('--model', type=str,
                    default="SimpleCNN", help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=15, help='epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--onnx', type=str, default="None", help='onnx')
parser.add_argument('--resolution', type=int, default=64, help='resolution')
parser.add_argument('--precision', type=int, default=16, help='precision')
parser.add_argument('--lr_cycles', type=float, default=1, help='precision')
parser.add_argument('--transformation_probability', type=float, default=0.5, help='precision')
parser.add_argument('--checkpoint', type=str, default="None", help='checkpoint')


def to_onnx(model: JetBotLightning):
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)
    torch.onnx.export(model.backbone, dummy_input,
                      f"{args.onnx}.onnx", verbose=True)


def get_model(backbone_name):
    if backbone_name == "SimpleCNN":
        backbone = SimpleCNN()
    elif backbone_name == "SqueezeNetCustom":
        backbone = SqueezedSqueezeNet(num_classes=2)
    elif backbone_name == "NvidiaModel":
        backbone =  NvidiaModel()
    elif backbone_name == "MobileNetSmall":
        backbone = MobileNetSmall()
    elif backbone_name == "SqueezeNet":
        backbone = SqueezeNet()
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

    # train_transformations = torchvision.transforms.Compose(
    #     [Resize((args.resolution, args.resolution)),
    #      RandomRotation((-4,4)),
    #      RandomPerspective(0.05, args.transformation_probability),
    #      RandomApply([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=args.transformation_probability),
    #      RandomApply([GaussianBlur(3, sigma=(0.1, 2.0))], p=args.transformation_probability),
    #      RandomAdjustSharpness(0.1, p=args.transformation_probability),
    #      RandomAutocontrast(p=args.transformation_probability),
    #      RandomEqualize(p=args.transformation_probability),
    #      ToTensor()])

    train_transformations = None
    valid_transformations = torchvision.transforms.Compose(
        [Resize((args.resolution, args.resolution)), ToTensor()])

    data_module = LineFollowingDataModule("./dataset", train_run_ids, vaild_run_ids, train_transformations,
                                          valid_transformations, batch_size=args.batch_size, num_workers=0)
    mean, sd = data_module.get_mean_and_std('train')
    train_transformations = DataAugmentation(args.transformation_probability, (args.resolution, args.resolution), mean, sd, standarization = True)
    data_module.train_transformations = train_transformations
    model = get_model(args.model)
    if args.checkpoint != "None":
        model = model.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(accelerator="auto",
                         precision=args.precision,
                         max_epochs=args.epochs, num_sanity_val_steps=2,
                         auto_lr_find=True, logger=WandbLogger(project="jetbot", name=args.model, config=args),
                         log_every_n_steps=1,
                         callbacks=[
                             ModelCheckpoint(
                                 dirpath="./checkpoints",
                                 filename=args.model+"-{epoch:02d}-{validation_loss:.2f}",
                                 monitor="validation_loss",
                             ),
                             LearningRateMonitor(),
                             EarlyStopping(monitor="validation_loss", patience=3, verbose=True)
                         ],
                         )
    # trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    if args.onnx != "None":
        to_onnx(model)
