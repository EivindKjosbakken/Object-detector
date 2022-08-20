from ast import List, Tuple
from collections import OrderedDict


import torchvision
import torch
from torch import nn


# The line belows inherits the configuration set for the tdt4265 dataset
from .task2_4NewFeatureSize import (
    train,
    anchors,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)


#new aspect ratios
anchors.aspect_ratios = [[1.5, 2], [1.5, 2], [2, 3],[2,3], [2, 3], [3, 4], [3, 4]]

