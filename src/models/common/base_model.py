from enum import Enum
from typing import Any

from pytorch_lightning import LightningModule
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class Activations(Enum):
    sigmoid = nn.Sigmoid()
    relu = nn.ReLU()


class LinearHead(nn.Module):
    def __init__(self, head_layers, activation='relu', last_activation='sigmoid'):
        super().__init__()
        self.head_layers = head_layers
        self.activation = activation
        self.last_activation = last_activation
        self.head = self.get_head_layers()

    def get_head_layers(self):
        head = nn.Sequential()
        k = 0
        for i in range(len(self.head_layers)-2):
            head.add_module(
                f'linear_{k}', nn.Linear(self.head_layers[i], self.head_layers[i+1])
            )
            head.add_module(
                f'activation_{k}',Activations[self.activaion].value
            )
            k = k + 1
        if len(self.head_layers)>2:
            length = len(self.head_layers)
            head.add_module(
                f'linear_{k}', nn.Linear(self.head_layers[length-2], self.head_layers[length-1])
            )
            head.add_module(
                f'activation_{k}', Activations[self.last_activation].value
            )
            k = k + 1
        return head

    def forward(self,x):
        return self.head(x)


class BaseModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model, self.head = self.init_model()


    def init_model(self):
        model = resnet18(weights=ResNet18_Weights)

        head = LinearHead(
            head_layers=self.config.head_layers
        )
        return model, head

    def forward(self,x):
        y = self.model(x)
        y = self.head(y)
        return y

    # def get_loss(self):
