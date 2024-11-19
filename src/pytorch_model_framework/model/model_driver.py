import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_model_framework.io_strategy import IoStrategy
from pytorch_model_framework.model.model_configs import BaseTrainConfig, DataConfig
from pytorch_model_framework.utils import EnhancedJSONEncoder


@dataclass
class CommandLineArgs:
    model_config_path: str


def parse_cli_args(input_args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_config_path", required=True, type=str)
    known, _ = parser.parse_known_args(input_args)
    return CommandLineArgs(
        model_config_path=known.model_config_path,
    )


class ModelDriver:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: BaseTrainConfig,
        data_module: LightningDataModule,
        model: pl.LightningModule,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.data_module = data_module
        self.model = model
        self.io_strategy = IoStrategy.from_location(location=self.data_config.experiment_path)

    def save_configs(self) -> None:
        data_config = json.dumps(self.data_config, cls=EnhancedJSONEncoder)
        self.io_strategy.save_file(file_name="data_config.json", contents=data_config)
        model_config = json.dumps(self.model_config, cls=EnhancedJSONEncoder)
        self.io_strategy.save_file(file_name="model_config.json", contents=model_config)

    def run_training(self):
        self.save_configs()
        trainer = self.setup_trainer()
        trainer.fit(
            model=self.model,
            train_dataloaders=self.data_module.train_dataloader(),
            val_dataloaders=self.data_module.val_dataloader(),
            ckpt_path=self.model_config.checkpoint_path,
        )

    @staticmethod
    def get_callbacks() -> List[Callback] | None:
        # TODO add learning rate callback
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [lr_monitor]

    def setup_trainer(self) -> pl.Trainer:
        logger = TensorBoardLogger(self.data_config.experiment_path, name="lightning_logs")
        device = self.model_config.device
        os.makedirs(self.data_config.experiment_path, exist_ok=True)
        if device == torch.device("cuda"):
            accelerator = "gpu"
        elif device == torch.device("mps"):
            accelerator = "mps"
        else:
            accelerator = "cpu"
        return pl.Trainer(
            default_root_dir=self.data_config.experiment_path,
            max_epochs=self.model_config.num_epochs,
            accelerator=accelerator,
            logger=logger,
            callbacks=self.get_callbacks(),
        )
