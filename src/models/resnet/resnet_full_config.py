from dataclasses import dataclass

import yaml

from models.common.base_model_config import BaseDataConfig
from models.resnet.resnet_configs import ResnetModelConfig


@dataclass(frozen=True)
class ResnetFullConfig(yaml.YAMLObject):
    data_config: BaseDataConfig
    model_config: ResnetModelConfig

    @staticmethod
    def from_dict(input_dict: dict) -> "ResnetFullConfig":
        return ResnetFullConfig(
            data_config=BaseDataConfig.from_dict(input_dict=input_dict.get("data_config")),
            model_config=ResnetModelConfig.from_dict(input_dict=input_dict.get("model_config"), loss_weight=[]),
        )