import torch
import torch.nn as nn


class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred, target, weight):
        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        return loss.sum()