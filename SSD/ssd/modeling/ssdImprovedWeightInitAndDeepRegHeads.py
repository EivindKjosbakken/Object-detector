from opcode import hasjabs
from shutil import ExecError
import numpy as np
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class SSD300DeepRegHeadsAndWeightInit(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            isImprovedWeightInit = False): #weight init is not activated as a baseline
        super().__init__()
        """
            Implements the SSD network.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.regression_heads = []
        self.classification_heads = []
        self.isImprovedWeightInit = isImprovedWeightInit


    
        # Initialize output heads that are applied to each feature map from the backbone.

        for n_boxes, out_ch in zip(anchors.num_boxes_per_fmap, self.feature_extractor.out_channels):
            #TODO changed here with deeper regression heads
            regressionBlock = nn.Sequential(
                nn.Conv2d(out_ch, 128, kernel_size=3, stride = 1, padding=1), #TODO as described on page 5 in paper except two 3x3 conv layers, not four. Layers have C = 256 filters from backbone, and one for the out filters)
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, n_boxes * 4, kernel_size=3, stride = 1, padding=1), #TODO kan den gi ut 256?
            )
            classificationBlock = nn.Sequential(
                nn.Conv2d(out_ch, 128, kernel_size=3, stride = 1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, n_boxes * self.num_classes, kernel_size=3, stride = 1, padding=1),
            )
            #self.regression_heads.append(nn.Conv2d(out_ch, n_boxes * 4, kernel_size=3, padding=1))
            self.regression_heads.append(regressionBlock) #deeper regression heads
            #self.classification_heads.append(nn.Conv2d(out_ch, n_boxes * self.num_classes, kernel_size=3, padding=1))
            self.classification_heads.append(classificationBlock)
      

        self.regression_heads = nn.ModuleList(self.regression_heads)
        self.classification_heads = nn.ModuleList(self.classification_heads)
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()


    def _init_weights(self):
        #print("Initializing with original weight init")   #TODO husk å endre tilbake 
        layers = [*self.regression_heads, *self.classification_heads]
        
        #old weight init:
        if (not self.isImprovedWeightInit): 
            print("Doing old weigth init")
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

        #improved weight init
        else:
            print("DOING IMPROVED WEIGHT INIT")
            for layer in layers: #for each conv layer (avoid relu with hasattr)
                for param in layer.children(): #since deep reg heads gave sequential objects (and not just a Conv2d), I have to iterate over all Conv layers in the Sequential object
                    if hasattr(param, "weight"):
                        nn.init.normal_(param.weight.data, mean = 0.0, std = 0.01)
    
                    if (hasattr(param, "bias")):
                        nn.init.constant_(param.bias.data, 0.0)
        

            lastClassLayer = self.classification_heads[-1]
            numAnchorsInLastFeatureMap = 6 #because I have two ratios: [2,3]
            p = 0.99
            k = 9
            import math
            backroundBias = math.log( p* (k-1)/(1-p)  )

            for param in lastClassLayer.children():
                #print("LAST LAYER: \n\n\n")
                if (hasattr(param, "bias")):
                    nn.init.constant_(param.bias[:numAnchorsInLastFeatureMap].data, backroundBias)
            
            """ #just to check implementation
            for layer in layers:
                for param in layer.children():
                    if (hasattr(param, "weight")):
                        print("WEIGHT:", param.weight.data)
                    if (hasattr(param, "bias")):
                        print("BIAS:", param.bias.data)

            print("RELU COUNTER IS:", reluCounter)
            """        
        

       
          

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for idx, x in enumerate(features):
            bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads[idx](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]