from dataclasses import dataclass, field
from typing import List


@dataclass
class BaseModelConfig:
    learning_rate: float
    epochs: int
    head_layers: List[int] = field(default_factory=list)

    def from_dict(self,input_dict):
        return BaseModelConfig(**input_dict)

@dataclass
class BaseDataConfig:
    data_dir: float
    train_size: float
    test_size: float
    val_size: float


    def from_dict(self, input_dict):
        return BaseDataConfig(**input_dict)



