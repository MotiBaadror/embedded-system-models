import json
import os
import sys

from models.common.base_datamodule import BaseDataModule
from models.common.base_loss import LOSSES
from models.common.base_model import LinearHead, BaseModel
from models.common.base_model_config import BaseDataConfig
from models.resnet.resnet_configs import ResnetModelConfig
from models.resnet.resnet_full_config import ResnetFullConfig
from pytorch_model_framework.model.model_driver import CommandLineArgs, parse_cli_args, ModelDriver


def get_data_module(config: BaseDataConfig, stage=None):
    data_module = BaseDataModule(
        config=config
    )
    data_module.setup(stage=stage)
    return data_module


def build_config(model_config_path: str) -> ResnetFullConfig:
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r") as f:
            config_json = json.loads(f.read())
            return ResnetFullConfig.from_dict(input_dict=config_json)
    else:
        raise RuntimeError(f"No file exists: {model_config_path}")


def get_loss_function(train_config: ResnetModelConfig):
    loss = LOSSES[train_config.loss].value(
        reduction=train_config.reduction,
        weight=train_config.loss_weight
    )
    return loss


def get_head_module(train_config: ResnetModelConfig):
    head = LinearHead(head_layers=train_config.head_layers, activation=train_config.activation, last_activation= train_config.last_activation)
    return head


def train_resnet(cli_args: CommandLineArgs):
    config = build_config(model_config_path=cli_args.model_config_path)
    data_module = get_data_module(config=config.data_config)

    model = BaseModel(
        config=config.model_config,
        head=get_head_module(train_config=config.model_config),
        loss=get_loss_function(config.model_config),
    )

    # model = freeze_custom(model, config.model_config)
    ModelDriver(
        data_config=config.data_config, model_config=config.model_config, data_module=data_module, model=model
    ).run_training()


if __name__ == "__main__":
    args = parse_cli_args(sys.argv)
    # args = CommandLineArgs(os.path.join(ROOT_DIR,"data/training/input_config.json"))
    train_resnet(cli_args=args)
