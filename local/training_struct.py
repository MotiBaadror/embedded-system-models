import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dir_configs import ROOT_DIR


@dataclass
class TrainingBasePaths:
    name: str
    input_path: str
    download_data: bool=False
    output_path: Optional[str] = None
    queue: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None

    def __post_init__(self):
        if not self.output_path and self.s3_bucket and self.s3_prefix:
            self.output_path = f"s3://{self.s3_bucket}/{self.s3_prefix}"


class TrainingLocation(Enum):
    LOCAL = TrainingBasePaths(
        name="local",
        input_path=os.path.join(ROOT_DIR, "data"),
        output_path=os.path.join(ROOT_DIR, "data", "training"),
    )
    EC2 = TrainingBasePaths(
        name="EC2",
        input_path="/workspace/app/data/input",
        s3_bucket="ad-creative-performance",
        s3_prefix="training",
        download_data=True,
    )