from dataclasses import dataclass

from pytorch_model_framework.model.model_configs import  DataConfig



@dataclass(frozen=True)
class BaseDataConfig(DataConfig):
    test_batch_size: int
    val_size: float = None
    test_size: float = None
    split_file_name: str = None

    @staticmethod
    def from_dict(input_dict):
        return BaseDataConfig(**input_dict)





