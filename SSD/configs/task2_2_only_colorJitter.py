from ssd.data.transforms.gpu_transforms import ColorJitter
from tops.config import LazyCall as L
import torchvision
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)

# The line belows inherits the configuration set for the tdt4265 dataset

from .task2_1 import (
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
    #gpu_transform,
    label_map
)


gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ColorJitter)(),
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

