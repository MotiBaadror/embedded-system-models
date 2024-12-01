from dataclasses import dataclass

from _pytest.fixtures import fixture

from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer


class TestTransformersModel:
    @fixture
    def results(self):
        @dataclass
        class Result:
            model = GLUETransformer("albert-base-v2", task_name="cola",num_labels=2)
            dm  = GLUEDataModule(
                "distilbert-base-uncased"
            )
            dm.prepare_data()
            dm.setup()
        return Result()

    def test_init(self,results):
        assert isinstance(results.model,GLUETransformer)
        print(results.model)

    def test_forward(self,results):
        for data in results.dm.train_dataloader():
            results.model.training_step(data, 1)
        pass



