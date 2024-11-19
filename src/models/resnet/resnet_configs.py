from dataclasses import dataclass, field
from typing import List, Optional

from pytorch_model_framework.model.model_configs import BaseTrainConfig


@dataclass(frozen=True)
class ResnetModelConfig(BaseTrainConfig):
    reduction: str
    activation: str
    last_activation: str
    loss: str
    lr_scheduler_mode: str   # has to be 'min' or 'max',
    lr_scheduler_factor: float
    lr_scheduler_patience : int
    head_layers: List[int] = field(default_factory=list)

    @staticmethod
    def from_dict(input_dict: dict, loss_weight: Optional[List[float]] = None) -> "ResnetModelConfig":
        runtime_parameters = BaseTrainConfig.clean_runtime_parameters(input_dict=input_dict, loss_weight=loss_weight)

        return ResnetModelConfig(
            loss_weight=runtime_parameters.loss_weight,
            device=runtime_parameters.device,
            **runtime_parameters.input_dict
        )