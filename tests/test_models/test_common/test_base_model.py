from dataclasses import dataclass
from unittest.mock import create_autospec, patch

import pytest
import torch
from _pytest.fixtures import fixture
from torch import nn

from models.common.base_loss import LOSSES
from models.common.base_model import LinearHead, BaseModel, Activations
import pytorch_lightning as pl

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




class TestbaseModel:
    @fixture
    def results(self):
        @dataclass
        class Config:
            labels = [1,0,0,1,1]
            # target = []
            # for i in labels:
            #     t = [0,0]
            #     t[i]=i
            #     target.append(t)

            input = (torch.rand(5,3,224,224), torch.tensor(labels, dtype=torch.long))
            learning_rate = 1e-3
            head_layers = [1000,10,2]
            step_size = 5
            gamma = 0.1
            # backbone_layers = [5,768]
            reduction = 'sum'
            num_labels = 2
            num_classes = 2

        @dataclass
        class Result:
            config = Config()
            head = LinearHead(head_layers=config.head_layers, activation='relu')
            loss = LOSSES['cross_entropy'].value(reduction=config.reduction)
            base_model = BaseModel(
                head=head,
                loss=loss,
                config=config
            )
        return Result()

    def test_init(self, results):
        assert isinstance(results.base_model.model, nn.Module)

    def test_forward(self, results):
        out = results.base_model.forward(results.config.input[0])
        assert out.shape == torch.Size([5,2])
        assert isinstance(out, torch.Tensor)

    def test_training_step(self, results):
        out = results.base_model.training_step(results.config.input)
        assert isinstance(out, torch.Tensor)
        assert out>0

    def test_validation_step(self, results):
        out = results.base_model.validation_step(results.config.input)
        assert isinstance(out, torch.Tensor)
        assert out > 0
        # assert isinstance(results.base_model.model, nn.Module)

    def test_lr_scheduler(self,results):
        val_dataset = torch.utils.data.TensorDataset(results.config.input[0], results.config.input[1])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)


        trainer = pl.Trainer(max_epochs=10, check_val_every_n_epoch=1, accelerator='cpu', logger= False , enable_checkpointing=False)

        trainer.fit(results.base_model, train_dataloaders=val_dataloader, val_dataloaders=val_dataloader)

        assert results.base_model.optimizers().param_groups[0]["lr"] > 1e-6
        assert results.base_model.optimizers().param_groups[0]["lr"] < 1e-4


    @pytest.mark.parametrize(
        "accuracy_task,label_size,pred_size,pred,target, acc, num_classes",
        [
            ('binary',[5],[5,2], torch.tensor([[0.8,0.2]]).repeat(5,1), torch.zeros(5),1.0,2),
            ('binary',[5],[5,2], torch.tensor([[0.8,0.2]]).repeat(5,1), torch.ones(5),0.0,2),
         ]
    )
    def test_get_cls(self, accuracy_task,label_size,pred_size,pred,target,acc,num_classes,results):
        results.base_model.config.accuracy_task = accuracy_task
        results.base_model.config.num_classes = num_classes
        assert pred.shape == torch.Size(pred_size)
        accuracy_metric = results.base_model.train_accuracy
        pred_labels = results.base_model.get_cls(pred=pred)
        assert pred_labels.shape == torch.Size(label_size)
        accuracy_metric.update(pred_labels,target)
        acc_value = accuracy_metric.compute()
        assert acc_value.item() == acc




