from enum import Enum

import torch.nn.functional
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self,weight=[1.0,1.0], reduction='mean'):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=torch.tensor(weight))

    def forward(self, input, target):
        return self.loss(input,target)


class LOSSES(Enum):
    cross_entropy = CrossEntropyLoss
