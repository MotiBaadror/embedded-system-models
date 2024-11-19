import json
import os
from typing import Union, Callable, Optional

from local.ec2_training import handle_instance_create_config
from local.training_struct import TrainingLocation
from models.resnet.resnet_full_config import ResnetFullConfig

from pytorch_model_framework.model.model_driver import CommandLineArgs
from pytorch_model_framework.utils import EnhancedJSONEncoder


def get_input_path(location: TrainingLocation, sub_dir: Optional[str] = ""):
    if location == TrainingLocation.LOCAL:
        return os.path.join(location.value.input_path, sub_dir)
    else:
        return location.value.input_path


def execute_training(
    location: TrainingLocation,
    config: Union[
        ResnetFullConfig,
        dict,
    ],
    execution: Callable,
):
    config_json = json.dumps(config, cls=EnhancedJSONEncoder)

    if location == TrainingLocation.LOCAL:
        os.makedirs(config.data_config.experiment_path, exist_ok=True)
        path_to_input_config = os.path.join(config.data_config.experiment_path, "input_config.json")
        with open(path_to_input_config, "w") as f:
            f.write(config_json)
        cli_args = CommandLineArgs(model_config_path=path_to_input_config)
        execution(cli_args=cli_args)
    elif location == TrainingLocation.EC2:
        handle_instance_create_config(
            config_json=config_json,
            training_location=location.value,
            experiment_descriptor=config.data_config.experiment_descriptor,
        )
