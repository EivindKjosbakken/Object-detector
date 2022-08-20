from ast import List, Tuple
from collections import OrderedDict

from grpc import insecure_channel
from numpy import imag
import torchvision
import torch
from torch import nn


class ModifiedResnetBackbone(torch.nn.Module):
    def __init__(self, output_channels, image_channels: int, output_feature_sizes):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        import torchvision.models as models
        self.model = models.resnet34(pretrained=True)
       
        

        #removing last 2 layers and adding to extra layers
        self.model.avgpool = nn.Sequential()
        self.model.fc = nn.Sequential()

        self.extraLayer1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels = 512,
                        out_channels= 256,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0,          #evt 1x1 conv her       
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels = 256,
                        out_channels= 256,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,          #evt 1x1 conv her       
                    ),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 2,2,0),
                    nn.ReLU(),
            )

        self.extraLayer2 = nn.Sequential(
                    nn.Conv2d(
                        in_channels = 256,
                        out_channels= 128,
                        kernel_size= 1,
                        stride = 1,
                        padding = 0,
                    ),
                    nn.ReLU(),
                   nn.Conv2d(
                        in_channels = 128,
                        out_channels= 128,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,          #evt 1x1 conv her       
                    ),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 2,2,0),
                    nn.ReLU(),
                )
        

        self.FPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 256, 128], 256)

    
    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out) #TODO evt fjerne denne
        out = self.model.relu(out)
        out = self.model.maxpool(out)


        out5 = (self.model.layer1(out)) #32,256
        out6 = (self.model.layer2(out5)) #16, 128
        out7 = (self.model.layer3(out6)) #8, 64
        out8 = (self.model.layer4(out7)) #4 , 32
        out9 = (self.extraLayer1(out8)) #2 ,16
        out10 = (self.extraLayer2(out9)) #1,8

        out_features = []
        #out5 = self.oneXoneConv1(out5)
        out_features.append(out5) 

        #out6 = self.oneXoneConv2(out6)
        out_features.append(out6)

        #out7 = self.oneXoneConv3(out7)
        out_features.append(out7)

        #out8 = self.oneXoneConv4(out8)
        out_features.append(out8)

        #out9 = self.oneXoneConv5(out9)
        out_features.append(out9)

       # out10 = self.oneXoneConv6(out10)
        out_features.append(out10)

        orderedDict = OrderedDict()
        orderedDict["feat0"] = out_features[0] 
        orderedDict["feat1"] = out_features[1] 
        orderedDict["feat2"] = out_features[2]  
        orderedDict["feat3"] = out_features[3] 
        orderedDict["feat4"] = out_features[4] 
        orderedDict["feat5"] = out_features[5] 
        fpn_out_features = []
        output = self.FPN(orderedDict)
        for k, v in output.items():
            fpn_out_features.append(v)

        for idx, feature in enumerate(fpn_out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(fpn_out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(fpn_out_features)}"
        
        return tuple(fpn_out_features)
 