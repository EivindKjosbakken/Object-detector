from ast import List, Tuple
from collections import OrderedDict

from grpc import insecure_channel
from numpy import imag
from ssd.modeling.anchor_boxes import AnchorBoxes
import torchvision
import torch
from torch import nn
from tops.config import LazyCall as L


# The line belows inherits the configuration set for the tdt4265 dataset
from .task2_4NewFeatureSizeAndChangedRatios import (
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

#new anchor boxes
anchors = L(AnchorBoxes)(
    feature_sizes=[[64, 512], [32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[2,2], [4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[8, 8], [16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios = [[1.5, 2], [1.5, 2], [2, 3],[2,3], [2, 3], [3, 4], [3, 4]],# endra
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)