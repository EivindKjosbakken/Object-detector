import sys, os
sys.path.append(os.path.dirname(os.getcwd())) # Include ../SSD in path
import numpy as np
import torch
import matplotlib.pyplot as plt
from vizer.draw import draw_boxes
from tops.config import instantiate, LazyConfig
from ssd import utils
np.random.seed(0)


config_path = "./configs/task2_2_only_horizontalFlipPIs05.py"
cfg = LazyConfig.load(config_path)
dataset_to_visualize = "train" # or "val"

dataset_to_visualize = "train" # or "val"
cfg.train.batch_size = 1
if dataset_to_visualize == "train":
    # Remove GroundTruthBoxesToAnchors transform
    if cfg.data_train.dataset._target_ == torch.utils.data.ConcatDataset:
        for dataset in cfg.data_train.dataset.datasets:
            dataset.transform.transforms = dataset.transform.transforms[:-1]
    else:
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
    dataset = instantiate(cfg.data_train.dataloader)
    gpu_transform = instantiate(cfg.data_train.gpu_transform)
else:
    cfg.data_val.dataloader.collate_fn = utils.batch_collate
    dataset = instantiate(cfg.data_val.dataloader) 
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

# Assumes that the first GPU transform is Normalize
# If it fails, just change the index from 0.
image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(1, 3, 1, 1)
image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(1, 3, 1, 1)
sample = next(iter(dataset))
sample = gpu_transform(sample)


print("The first sample in the dataset has the following keys:", sample.keys())
for key, item in sample.items():
    print(
        key, ": shape=", item.shape if hasattr(item, "shape") else "", 
        "dtype=", item.dtype if hasattr(item, "dtype") else type(item), sep="")

        
image = (sample["image"] * image_std + image_mean)

from copy import deepcopy
image2 = deepcopy(image)

image = (image*255).byte()[0]
boxes = sample["boxes"][0]
boxes[:, [0, 2]] *= image.shape[-1]
boxes[:, [1, 3]] *= image.shape[-2]
im = image.permute(1, 2, 0).cpu().numpy()
im = draw_boxes(im, boxes.cpu().numpy(), sample["labels"][0].cpu().numpy().tolist(), class_name_map=cfg.label_map)


plt.imshow(im)
plt.show()



config_path = "./configs/task2_1.py"
cfg = LazyConfig.load(config_path)
dataset_to_visualize = "train" # or "val"

dataset_to_visualize = "train" # or "val"
cfg.train.batch_size = 1
if dataset_to_visualize == "train":
    # Remove GroundTruthBoxesToAnchors transform
    if cfg.data_train.dataset._target_ == torch.utils.data.ConcatDataset:
        for dataset in cfg.data_train.dataset.datasets:
            dataset.transform.transforms = dataset.transform.transforms[:-1]
    else:
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
    dataset = instantiate(cfg.data_train.dataloader)
    gpu_transform = instantiate(cfg.data_train.gpu_transform)
else:
    cfg.data_val.dataloader.collate_fn = utils.batch_collate
    dataset = instantiate(cfg.data_val.dataloader) 
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

# Assumes that the first GPU transform is Normalize
# If it fails, just change the index from 0.
image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(1, 3, 1, 1)
image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(1, 3, 1, 1)
sample = next(iter(dataset))
sample = gpu_transform(sample)


print("The first sample in the dataset has the following keys:", sample.keys())
for key, item in sample.items():
    print(
        key, ": shape=", item.shape if hasattr(item, "shape") else "", 
        "dtype=", item.dtype if hasattr(item, "dtype") else type(item), sep="")

        
image = (sample["image"] * image_std + image_mean)

image = (image2*255).byte()[0]
boxes = sample["boxes"][0]
boxes[:, [0, 2]] *= image.shape[-1]
boxes[:, [1, 3]] *= image.shape[-2]
im = image.permute(1, 2, 0).cpu().numpy()
im = draw_boxes(im, boxes.cpu().numpy(), sample["labels"][0].cpu().numpy().tolist(), class_name_map=cfg.label_map)


plt.imshow(im)
plt.show()
