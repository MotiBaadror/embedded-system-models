import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, field
from datetime import datetime
from typing import List, Optional

import torch
import yaml
from torch.types import Device

from pytorch_model_framework.io_strategy import S3Strategy, S3UriComponents, InvalidUri, LocalStrategy


class InvalidConfig(Exception):
    pass


class DirectoryNotEmpty(Exception):
    pass


@dataclass(frozen=True)
class DilationConfig(yaml.YAMLObject):
    factor: int = 0
    indexes: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.factor != 0 and len(self.indexes) == 0:
            raise InvalidConfig(
                f"You must define indexes to dilate if you provide a dilation factor. Factor: {self.factor}"
            )

    @staticmethod
    def from_dict(input_dict: dict) -> "DilationConfig":
        return DilationConfig(**input_dict)


@dataclass(frozen=True)
class DataConfig(yaml.YAMLObject):
    input_path: str
    output_path: str
    batch_size: int
    test_batch_size: int
    time_window: int
    train_size: float
    num_workers: int
    prefetch_factor: int
    num_classes: int
    background_class: int = 20
    dilation: DilationConfig = DilationConfig()
    s3_source_input_path: Optional[str] = None
    experiment_descriptor: Optional[str] = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def from_dict(input_dict: dict) -> "DataConfig":
        dilation_dict = input_dict.get("dilation", {})
        dilation = DilationConfig.from_dict(input_dict=dilation_dict)
        input_dict.pop("dilation", None)
        return DataConfig(**input_dict, dilation=dilation)

    @property
    def experiment_path(self) -> str:
        return os.path.join(self.output_path, self.experiment_descriptor)

    def maybe_get_data(self) -> None:
        os.makedirs(self.input_path, exist_ok=True)
        with os.scandir(self.input_path) as it:
            if any(it):
                raise DirectoryNotEmpty(
                    f"input_path is not empty: {self.input_path}. Will not attempt to download files."
                )

        S3Strategy.download_many(source=self.s3_source_input_path, destination=self.input_path)

    def __post_init__(self):
        if self.background_class >= self.num_classes:
            raise InvalidConfig(
                f"background_class must be less than num_classes. background_class: {self.background_class}, num_classes: {self.num_classes}"
            )


@dataclass(frozen=True)
class RuntimeParameters:
    input_dict: dict
    loss_weight: List[float]
    device: Device = torch.device(
        # "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )


class ABCYamlMeta(type(yaml.YAMLObject), type(ABC)):
    pass


@dataclass(frozen=True)
class BaseTrainConfig(yaml.YAMLObject, ABC, metaclass=ABCYamlMeta):
    device: Device
    num_epochs: int
    learning_rate: float
    step_size: int
    gamma: float
    focal_loss_gamma: float
    focal_loss_alpha: float
    loss_weight: List[float]
    checkpoint_path: Optional[str]

    def __post_init__(self):
        allowed_to_be_none: List[str] = ["checkpoint_path"]
        invalid_fields = []
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if field_value is None and field_name not in allowed_to_be_none:
                invalid_fields.append(field_name)

        if self.checkpoint_path is not None:
            try:
                s3_exists = S3Strategy(location=self.checkpoint_path).file_exists()
            except InvalidUri:
                s3_exists = False
            local_exists = LocalStrategy(location=self.checkpoint_path).file_exists()
            if not s3_exists and not local_exists:
                raise InvalidConfig(
                    f"checkpoint_path does not exist locally or on s3. checkpoint_path: {self.checkpoint_path}"
                )
        if invalid_fields:
            raise InvalidConfig(f"No config fields may be NoneType. The following fields were: {invalid_fields}")

    @staticmethod
    def clean_runtime_parameters(input_dict: dict, loss_weight: Optional[List[float]] = None) -> RuntimeParameters:
        input_dict.pop("device", None)

        loss_weight_from_user = input_dict.get("loss_weight", [])
        input_dict.pop("loss_weight", None)
        if type(loss_weight_from_user) == list and len(loss_weight_from_user) == 0:
            loss_weight_to_use = loss_weight if loss_weight is not None and len(loss_weight) > 0 else []
        elif loss_weight_from_user is None:
            loss_weight_to_use = []
        else:
            loss_weight_to_use = loss_weight_from_user

        return RuntimeParameters(input_dict=input_dict, loss_weight=loss_weight_to_use)

    @staticmethod
    @abstractmethod
    def from_dict(input_dict: dict, loss_weight: Optional[List[float]] = None):
        raise RuntimeError("Your dataclass must implement this method.")
