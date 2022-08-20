import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def calcFocalLoss(input, target, gamma, alpha):
    #TODO lage og bruke funksjonen
   
    p = F.softmax(input, dim=1)
    p = torch.exp(-p) 
    logp = F.log_softmax(input, dim=1)
    logp = torch.exp(-logp) 

    del1 = alpha*((1-p)**gamma)
    target = torch.reshape(target, (32, 9, 65440))
    #print(target.shape, logp.shape, "________")
    del2 = target*logp
    #print("DEL1 SHAPE:", del1.shape, "DEL2 SHAPE:", del2.shape)
    del2 = torch.reshape(del2, (32, 65440, 9))
    del1 = del1.cpu().detach().numpy()
    del2 = del2.cpu().detach().numpy()
    loss = (np.dot(del1, del2))
    focal_loss = np.mean(loss)
    #print("FOCAL LOSS IS:", focal_loss)
    return focal_loss


def hard_negative_mining(loss, labels, neg_pos_ratio):
    #print("loss:", loss)
    #print("losshape",loss.shape)
    pos_mask = labels > 0
    #print("Posmask", pos_mask)
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    #print("numpos", num_pos)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    #print("negmask", neg_mask)
    #print("a", pos_mask | neg_mask)
    return pos_mask | neg_mask


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]


        """ hard negative mining previosely
        with torch.no_grad():
            to_log = - F.log_softmax(confs, dim=1)[:, 0]
            mask = hard_negative_mining(to_log, gt_labels, 3.0)
        classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        classification_loss = classification_loss[mask].sum()
        print("OLD LOSS:", classification_loss)
        #print("ORIGINAL CLASSIFICATION LOSS IS:", classification_loss, "and type: ", type(classification_loss))
        """

        #changing classification loss from hard negative mining to focal loss:
        #with torch.no_grad():
        alpha, gamma = torch.tensor([10, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]).cuda(), 2
        alpha = torch.reshape(alpha, (1, 9, 1)) #endre shape
        oneHotEncoding = F.one_hot(gt_labels, 9) #the target 
        oneHotEncoding = oneHotEncoding.transpose(1, 2)
        pk = F.softmax(confs, dim=1)
        logpk = F.log_softmax(confs, dim = 1)

        focal = (1-pk)**gamma

        ans1 = focal*oneHotEncoding*logpk
        ans2 = alpha*ans1
        #print("ShAPE: ", ans2.shape)
        #print("try loss:", -ans2.sum())
        loss = -ans2.sum(dim=1) #summing over the axis which has shape 9
        #print(loss.shape)
        focal_loss = loss.mean()

        classification_loss = focal_loss
        #print("NEW LOSS:", classification_loss) #0.02 was good

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        #total_loss = regression_loss/num_pos + classification_loss/num_pos
        total_loss = regression_loss/num_pos + classification_loss #TODO changed here
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss,
            total_loss=total_loss
        )
        return total_loss, to_log
