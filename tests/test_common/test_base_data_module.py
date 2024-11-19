import os
import random
from dataclasses import dataclass
from unittest.mock import Mock, patch

import torch
from _pytest.fixtures import fixture
from pytorch_lightning import LightningDataModule

from models.common.base_datamodule import BaseDataModule
from models.common.base_model_config import BaseDataConfig


class TestBaseDataModule:
    @fixture
    def results(self):
        @dataclass
        class Result:
            config = BaseDataConfig(
            input_path='data/version_0',
            output_path ='some-path',
            batch_size = 2,
            test_batch_size=1,
            time_window =0,
            train_size =0.6,
            val_size = 0.2,
            test_size =0.2,
            num_workers=1,
            prefetch_factor=1,
            num_classes=1,
            split_file_name='train_test_split_dummy'
            )
            data_module = BaseDataModule(
                config=config
            )
        return Result()

    def test_init(self,results):
        assert isinstance(results.data_module, LightningDataModule)

    @patch('models.common.base_datamodule.os')
    @patch('models.common.base_datamodule.random')
    @patch('models.common.base_datamodule.open')
    @patch('models.common.base_datamodule.json')
    def test_split_new_ids(self,mock_json,mock_open,mock_random,mock_os, results):
        mock_os.listdir.return_value = [str(i) for i in range(10)]
        # mock_os.path.join = os.path.join
        split_file = results.data_module.get_split_file_name()
        ids = results.data_module.split_new_ids(split_file)
        assert len(ids[0]) == 6
        assert len(ids[1]) == 2
        assert len(ids[2]) == 2

    def test_setup(self,results):
        results.data_module.split_new_ids = Mock()
        results.data_module.split_new_ids.return_value = ([1,2,3],[4],[5])
        results.data_module.setup()

        assert len(results.data_module.train_dataset)
        assert len(results.data_module.test_dataset)
        assert len(results.data_module.val_dataset)

    @patch('models.common.base_dataset.torch')
    def test_train_dataloader(self,mock_torch,results):
        results.data_module.split_new_ids = Mock()
        results.data_module.split_new_ids.return_value = (['1', '2', '3'], ['4'], ['5'])
        results.data_module.setup()
        train_dataloader = results.data_module.train_dataloader()
        mock_torch.load.return_value = (torch.randn(10),torch.tensor(1))
        assert len(train_dataloader) > 0
        for data in train_dataloader:
            x,y = data
            assert x.shape == torch.Size([2,10])
            assert y.shape == torch.Size([2])
            # print(data)
            break
