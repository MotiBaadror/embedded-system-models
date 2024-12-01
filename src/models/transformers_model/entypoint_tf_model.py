from dataclasses import dataclass, field

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer
import pytorch_lightning as pl

@dataclass
class MyTrainingConfig:
    epochs: int
    exp_name: str = 'tiny_v2_losswt_10_1'
    model: str = 'hf-internal-testing/tiny-albert'
    checkpoint_path: str = add_rootpath(
        'data/trainings/tiny_v2_losswt_10_1/lightning_logs/version_0/checkpoints/epoch=2-step=315.ckpt'
    )
    loss_weights: list[float] = field(default_factory = lambda: [10.0,1.0])
    base_dir: str = add_rootpath('data/trainings')

    # def __post_init__(self):
    #     self.out_dir =

    @staticmethod
    def from_dict(input_dict: dict):
        return MyTrainingConfig(**input_dict)




config = MyTrainingConfig(
    epochs=10
)

pl.seed_everything(42)

dm = GLUEDataModule(model_name_or_path=config.model, task_name="cola")
dm.prepare_data()
dm.setup("fit")

model = GLUETransformer(
    model_name_or_path=config.model,
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
    learning_rate=1e-4,
    loss_weights=config.loss_weights
)

def run_training(dm=dm, model=model, config=config):

    exp_name = config.exp_name
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(save_dir=add_rootpath(f'data/trainings/{exp_name}')),
    )
    trainer.fit(model, datamodule=dm, ckpt_path=config.checkpoint_path)

if __name__ == '__main__':
    run_training()
