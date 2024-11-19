from unittest.mock import Mock

import torch

from models.common.base_loss import CrossEntropyLoss
from models.resnet.resnet_entrypoint import get_loss_function


def test_get_loss():
    train_config = Mock()
    train_config.reduction = 'mean'
    train_config.loss = 'cross_entropy'
    train_config.loss_weight = [1.0,1.0]
    loss = get_loss_function(train_config)
    input = torch.randn(5,2)
    target = torch.ones(5).long()
    assert isinstance(loss, CrossEntropyLoss)
    loss_value = loss(input,target)
    assert loss_value>0





