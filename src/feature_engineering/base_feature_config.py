import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from dir_configs import add_rootpath


class ClassMapping(Enum):
    performing = 1
    non_performing = 0


@dataclass
class BaseFeatureConfig:
    base_path: str
    input_version: int
    output_version: int
    output_dir: str
    size: List[int] = field(default_factory=lambda: [224,224])



    @staticmethod
    def from_dict(input_dict):
        return BaseFeatureConfig(**input_dict)

    def __post_init__(self):
        self.base_path = add_rootpath(
            os.path.join(self.base_path, f'version_{self.input_version}')
        )
        self.output_dir = os.path.join(
            add_rootpath(self.output_dir),
            f'version_{self.output_version}/tensors'
        )
        os.makedirs(self.output_dir, exist_ok=True)
