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
        self.activaion = activation
        self.last_activation = last_activation

    def get_head_layers(self):
        head = nn.ModuleList()
        k = 0
        for i in range(len(self.head_layers-2)):
            head.add_module(
                f'linear_{k}', nn.Linear(self.head_layers[i], self.head_layers[i+1])
            )
            head.add_module(
                f'activation_{k}',Activations[self.activaion].value
            )


    # def


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights)

    def forward(self,x):
        y = self.model(x)
        return y

    # def get_loss(self):
