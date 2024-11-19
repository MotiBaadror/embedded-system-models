import torch
from models.common.base_augmentations import NoTransform


def test_no_transform():
    x = torch.randn(10,10)
    aug = NoTransform()
    out = aug(x)
    assert out.shape == x.shape
    assert torch.all(out == x)