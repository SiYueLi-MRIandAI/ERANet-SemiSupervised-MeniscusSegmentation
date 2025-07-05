import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
sys.path.append('..')
from utils import ramps
from einops import rearrange

def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)
#    loss1= torch.nn.functional.cross_entropy(pred, gt)
    return  loss.sum() / N

class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :, :], y_true[:, i:i + 1, :, :], activation=self.activation))
        mean_dice = sum(class_dice)/self.num_classes
        return 1-mean_dice

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def weight_self_pro_softmax_mse_loss(input_logits, target_logits, entropy):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    target_logits = target_logits.view(target_logits.size(0), target_logits.size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_softmax.detach() - target_softmax) ** 2
    # entropy =  1-entropy.unsqueeze(-1).detach()
    # mse_loss =entropy*mse_loss
    return mse_loss
