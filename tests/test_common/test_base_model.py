from dataclasses import dataclass
from unittest.mock import create_autospec

import torch
from _pytest.fixtures import fixture
from torch import nn

from models.common import BaseModel, Activations, LinearHead
from models.common import BaseModelConfig


def test_activation_enum():
    input = torch.randn(10,3)
    sigmoid = Activations['sigmoid'].value
    assert isinstance(sigmoid, nn.Sigmoid)
    relu = Activations['relu'].value
    assert isinstance(relu, nn.ReLU)
    sigmoid_out = sigmoid(input)
    assert torch.all(sigmoid_out <torch.tensor(1.0))
    assert torch.all(sigmoid_out >torch.tensor(0.0))


class TestHead():
    @fixture
    def results(self):
        @dataclass
        class Result:
            head_layers = [10,20,30,40]
            head = LinearHead(head_layers=head_layers)
            input = torch.randn(2,10)
            out_size = torch.Size([2,40])

        return Result()

    def test_init(self,results):
        assert isinstance(results.head, nn.Module)
        # print(results.head)

    def test_forward(self,results):
        out = results.head(results.input)
        assert out.shape == results.out_size


class TestBaseModel():
    @fixture
    def results(self):
        @dataclass
        class Result:
            config = create_autospec(BaseModelConfig)
            config.head_layers = [1000,10,1]
            model = BaseModel(
                config=config
            )
            input = torch.randn(2,3,224,224)

            # out_size = torch.Size([2,1000])

            out_size = torch.Size([2, 1])
        return Result

    def test_init(self,results):
        # print(results.model)
        y = results.model(results.input)

    def test_forward(self,results):
        y = results.model.forward(results.input)
        assert y.shape == results.out_size

