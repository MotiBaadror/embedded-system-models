from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from pytorch_model_framework.model.model_configs import DataConfig


@dataclass
class BaseModelConfig:
    learning_rate: float
    epochs: int
    head_layers: List[int] = field(default_factory=list)

    def from_dict(self,input_dict):
        return BaseModelConfig(**input_dict)

@dataclass
class BaseDataConfig:
    input_path: str
    output_path: str
    batch_size: int
    test_batch_size: int
    time_window: int
    train_size: float
    val_size: float
    test_size: float
    num_workers: int
    prefetch_factor: int
    num_classes: int
    background_class: int = 20
    s3_source_input_path: Optional[str] = None
    experiment_descriptor: Optional[str] = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    split_file_name: str = None
    data_version: int = None



    def from_dict(self, input_dict):
        return BaseDataConfig(**input_dict)



