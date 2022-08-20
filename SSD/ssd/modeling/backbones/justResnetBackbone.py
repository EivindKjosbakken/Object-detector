from ast import List, Tuple
from collections import OrderedDict

from grpc import insecure_channel
from numpy import imag
import torchvision
import torch
from torch import nn


class justRestnetBackbone(torch.nn.Module):
    def __init__(self, output_channels, image_channels: int, output_feature_sizes):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        import torchvision.models as models
        self.model = models.resnet34(pretrained=True)
        """
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        
        #not freezing these as that makes the network perform worse
        #for param in self.model.layer3.parameters():
        #    param.requires_grad = False
        #for param in self.model.layer4.parameters():
        #    param.requires_grad = False
        """
        """
        self.model.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels = image_channels,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        #changing last two layers (is bad naming, but easier than adding new layers):
        self.model.avgpool = nn.Sequential(
            
            nn.Conv2d(
                in_channels = 512,
                out_channels= 256,
                kernel_size = 2,
                stride = 2,
                padding = 0,                
            ),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.model.fc = nn.Sequential(
            nn.Conv2d(
                in_channels = 256,
                out_channels= 128,
                kernel_size= 2,
                stride = 2,
                padding = 0,
            ),
            nn.ReLU(),

        )
        

        self.FPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 256, 128], 256)
        
            
    
    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out5 = (self.model.layer1(out))
        out6 = (self.model.layer2(out5))
        out7 = (self.model.layer3(out6))
        out8 = (self.model.layer4(out7))
        out9 = (self.model.avgpool(out8))
        out10 = (self.model.fc(out9))

        out_features = []
        out_features.append(out5) 
        out_features.append(out6)
        out_features.append(out7)
        out_features.append(out8)
        out_features.append(out9)
        out_features.append(out10)
    
       
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        
        return tuple(out_features)
