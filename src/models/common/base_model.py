from enum import Enum
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy
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
                f'activation_{k}',Activations[self.activation].value
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
    def __init__(self, config, head, loss):
        super().__init__()
        self.config = config
        self.model = self.init_model()
        self.head = head
        self.loss = loss
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.save_hyperparameters()

    def get_loss(self, input, target):
        return self.loss(input,target)

    def init_model(self):
        model = resnet18(weights=ResNet18_Weights)

        # head = LinearHead(
        #     head_layers=self.config.head_layers
        # )
        return model

    def forward(self,x):
        y = self.model(x)
        y = self.head(y)
        return y

    def get_cls(self, pred):
        _, pred_labels = torch.max(pred, dim=-1)
        return pred_labels


    def training_step(self, input):
        x, y = input
        pred = self.forward(x)
        val_loss = self.loss(pred, y)
        self.log('train_loss', value=val_loss, on_step=True, on_epoch=True, prog_bar=True)
        pred_labels = self.get_cls(pred=pred)
        self.train_accuracy.update(pred_labels, y)
        return val_loss


    def validation_step(self,input, *args: Any, **kwargs: Any):
        x, y = input
        pred = self.forward(x)
        val_loss = self.loss(pred,y)
        self.log('val_loss',value= val_loss, on_step=True, on_epoch=True, prog_bar=True)
        pred_labels = self.get_cls(pred=pred)
        self.val_accuracy.update(pred_labels, y)
        return val_loss

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_accuracy.compute(), on_epoch=True, prog_bar = True, logger = True)
        self.train_accuracy.reset()

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_accuracy.compute(), on_epoch=True, prog_bar = True, logger = True)
        self.val_accuracy.reset()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma, verbose=True)
        # self.lr_scheduler = ReduceLROnPlateau(optimizer, patience=self., verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # "scheduler": ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-6),
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    # def get_loss(self):
