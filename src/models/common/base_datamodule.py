import json
import os.path
import random

import sklearn.model_selection
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dir_configs import add_rootpath
from models.common.base_augmentations import NoTransform
from models.common.base_dataset import BaseCreativeDataset
from models.common.base_model_config import BaseDataConfig


class BaseDataModule(LightningDataModule):
    def __init__(self, config:BaseDataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.base_path = add_rootpath(config.input_path)
        self.tensor_path = os.path.join(self.base_path,f'version_{config.data_version}/tensors')

    def load_split_file(self, split_file):
        with open(split_file, 'r') as f:
            ids = json.loads(f.read())
        return ids['train_ids'], ids['test_ids'], ids['val_ids']

    def get_split_file_name(self):
        file_name = self.config.split_file_name+(
            f'_train_{self.config.train_size}'
            f'_test_{self.config.test_size}_val_{self.config.val_size}.json')
        split_file = os.path.join(self.base_path, file_name)

        return split_file

    def split_new_ids(self, split_file):
        ids = os.listdir(self.tensor_path)
        rest_size = 1-self.config.train_size
        train_ids, rest_ids = train_test_split(ids,train_size= 1-rest_size,test_size= rest_size, shuffle=True, random_state=12345)
        val_ids, test_ids = train_test_split(rest_ids, train_size= self.config.val_size/rest_size, test_size= self.config.test_size/rest_size, shuffle=True, random_state=12345)
        splits = dict(
            train_ids=train_ids,
            test_ids=test_ids,
            val_ids=val_ids
        )

        print(f"saving new split, {split_file}")
        with open(split_file,'w') as f:
            json.dump(splits,f, indent=4)
        return train_ids,test_ids,val_ids

    def setup(self, stage: str = None):
        split_file = self.get_split_file_name()
        if os.path.exists(split_file):
            train_ids , test_ids,val_ids = self.load_split_file(split_file)
        else:
            train_ids, test_ids, val_ids = self.split_new_ids(split_file)

        self.train_dataset = BaseCreativeDataset(
            file_names=train_ids, base_path=self.tensor_path, transforms=NoTransform()
        )
        self.test_dataset = BaseCreativeDataset(
            file_names=test_ids, base_path=self.tensor_path, transforms=NoTransform()
        )
        self.val_dataset = BaseCreativeDataset(
            file_names=val_ids, base_path=self.tensor_path, transforms=NoTransform()
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.test_batch_size
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.test_batch_size
        )





