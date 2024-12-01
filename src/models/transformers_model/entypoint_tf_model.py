import torch
from pytorch_lightning.loggers import TensorBoardLogger

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer
import pytorch_lightning as pl

pl.seed_everything(42)

dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola")
dm.prepare_data()
dm.setup("fit")
checkpoint_path = add_rootpath('data/trainings/lightning_logs/lightning_logs/version_0/checkpoints/epoch=0-step=97.ckpt')

model = GLUETransformer(
    model_name_or_path="albert-base-v2",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
    learning_rate=1e-4
)



trainer = pl.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    logger=TensorBoardLogger(save_dir=add_rootpath('data/trainings/lightning_logs')),
)
trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)
