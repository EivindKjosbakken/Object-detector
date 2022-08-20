from ssd.modeling.ssdImprovedWeightInitAndDeepRegHeads import SSD300DeepRegHeadsAndWeightInit
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .ssd300 import train, anchors, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes


from .task2_2_only_horizontalFlipPIs03ExtendedDataset import ( 
    train,
    anchors,
    optimizer,
    schedulers,
    loss_objective,
    #model,
    #backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)
#train.seed=0

train.epochs = 75


#FPN backbone
backbone = L(backbones.ModifiedResnetBackbone)(
    output_channels=[256, 256, 256, 256, 256, 256], #output channels i got when running through the model, FPN gives out 64 at each layer
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

anchors.aspect_ratios=[[2,3], [2,3], [2,3], [2,3], [2,3], [2,3]] #need same number of anchors per feature map for deeper reg heads to work
model = L(SSD300DeepRegHeadsAndWeightInit)( #new model init with improved weights
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    isImprovedWeightInit = True #do not have improved weight init here
)