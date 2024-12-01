from dataclasses import dataclass

import torch
from _pytest.fixtures import fixture
from torchmetrics import Accuracy

from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer


class TestTransformersModel:
    @fixture
    def results(self):
        @dataclass
        class Result:
            model = GLUETransformer("albert-base-v2", task_name="cola",num_labels=2, loss_weights=[1.0,1.0])
            dm  = GLUEDataModule(
                "distilbert-base-uncased"
            )
            dm.prepare_data()
            dm.setup()
        return Result()

    def test_init(self,results):
        assert isinstance(results.model,GLUETransformer)
        print(results.model)

    def test_training_step(self,results):
        for data in results.dm.train_dataloader():
            results.model.training_step(data, 1)
        pass

    # def test_forward(self):

    def test_accuracy(self):
        acc = Accuracy(task='binary')
        acc.update(torch.tensor([1]),torch.tensor([1]))
        # acc.update([0],[0])
        print(acc.compute())
        acc.reset()
        print(acc.compute())

    def test_loss(self, results):

        input = torch.randn([32,2])
        target = torch.ones([32], dtype=torch.int64)
        results.model.get_loss(input,target)
        # results.model.loss_func(input,target)


