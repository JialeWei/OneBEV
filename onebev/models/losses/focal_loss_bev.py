import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

@MODELS.register_module()
class FocalLossBEV(nn.Module):

    def __init__(self, alpha=-1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def sigmoid_focal_loss(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                     targets,
                                                     reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss   

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

    def forward(self, pred, batch_data_samples):
        losses = {}
        target = torch.stack(
            [data_sample.gt_sem_seg for data_sample in batch_data_samples],
            dim=0)

        for index, name in enumerate(batch_data_samples[0].map_classes):
            loss = self.sigmoid_focal_loss(pred[:, index], target[:, index])
            losses[f"{name}/focal"] = loss
        return losses