from turtle import forward
import torchvision
import torch
from torch import nn
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir
import numpy as np
from typing import OrderedDict, Tuple, List
from torchvision.models import resnet34
from ssd.modeling import backbones
from typing import OrderedDict, Tuple, List
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir

#from .task2_2_only_horizontalFlip import (
from .task2_1 import ( #not using data augment
    train,
    anchors,
    optimizer,
    schedulers,
    loss_objective,
    model,
    #backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)
train.seed=0


#FPN backbone
backbone = L(backbones.justRestnetBackbone)(
    output_channels=[64, 128, 256, 512, 256, 128], #TODO endra litt,fikk over 0.5 med dette
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)



