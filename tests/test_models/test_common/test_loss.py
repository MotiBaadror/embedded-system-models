import torch

from models.common.base_loss import CrossEntropyLoss

def test_cross_entropy_loss():
    loss_func = CrossEntropyLoss()
    input = torch.rand(5,2)
    target = torch.ones(5, dtype=torch.long)
    loss_value = loss_func(input,target)
    print(loss_value.item())
    assert loss_value>0
