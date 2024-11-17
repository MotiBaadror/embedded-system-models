from dataclasses import dataclass

import torch
from _pytest.fixtures import fixture
from torch import nn

from common.base_model import BaseModel, Activations


def test_activation_enum():
    input = torch.randn(10,3)
    sigmoid = Activations['sigmoid'].value
    assert isinstance(sigmoid, nn.Sigmoid)
    relu = Activations['relu'].value
    assert isinstance(relu, nn.ReLU)
    sigmoid_out = sigmoid(input)
    assert torch.all(sigmoid_out <torch.tensor(1.0))
    assert torch.all(sigmoid_out >torch.tensor(0.0))


class TestBaseModel():
    @fixture
    def results(self):
        @dataclass
        class Result:
            model = BaseModel()
            input = torch.randn(2,3,224,224)
            out_size = torch.Size([2,1000])
        return Result

    # def test_init(self,results):
    #     print(results.model)
    #     y = results.model(x)

    def test_forward(self,results):
        y = results.model.forward(results.input)
        assert y.shape == results.out_size
