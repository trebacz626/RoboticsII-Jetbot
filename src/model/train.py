import pytorch_lightning as pl
import timm
import torch
from model.simple_cnn import SimpleCNN
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default="SimpleCNN", help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gpus', type=int, default=1, help='gpus')
parser.add_argument('--precision', type=int, default=16, help='precision')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--data_dir', type=str, default='data', help='data_dir')
parser.add_argument('--output_dir', type=str,
                    default='output', help='output_dir')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--onnx', type=bool, default=False, help='onnx')

def to_onnx(model):
    dummy_input = torch.randn(1, 3, 64, 64)
    # predict dummy input
    output = model(dummy_input)
    print(output)
    torch.onnx.export(model, dummy_input, str(model), verbose=True)


def get_model(model_name):
    if model_name == "SimpleCNN":
        return SimpleCNN()
    else:
        return timm.create_model('resnet18', pretrained=True)


if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    model = get_model(args.model)
    trainer = pl.Trainer(gpus=args.gpus, precision=args.precision,
                         max_epochs=args.epochs, num_sanity_val_steps=2)
    trainer.fit(model)
    trainer.test(model)
    if args.onnx:
        to_onnx(model)
