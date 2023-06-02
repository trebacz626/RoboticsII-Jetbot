import torchvision

from src.data.datamodule import LineFollowingDataModule

train_run_ids = ["1652875851.3497071", "1652875901.3107166", "1652876013.741493", "1652876206.2541456", "1652876485.8123376", "1652959186.4507334", "1652959347.972946", "1653042695.4914637", "1653042775.5213027", "1653043202.5073502"]
vaild_run_ids = ["1653043345.3415065", "1653043428.8546412", "1653043549.5187616"]

train_transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
valid_transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_module = LineFollowingDataModule("../../dataset", train_run_ids, vaild_run_ids, train_transformations, valid_transformations, batch_size=32, num_workers=0)
train_loader = data_module.train_dataloader()
print(train_loader.dataset.df.head())

for batch in train_loader:
    print(batch[0].shape, batch[1].shape)
    break

