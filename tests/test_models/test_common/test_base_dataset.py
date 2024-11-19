from dataclasses import dataclass
from unittest.mock import patch

from _pytest.fixtures import fixture
from torch import nn

from models.common.base_augmentations import NoTransform
from models.common.base_dataset import BaseCreativeDataset


class TestBaseDataset():
    @fixture
    def result(self):
        @dataclass
        class Result:
            dataset = BaseCreativeDataset(
                file_names=[str(i) for i in range(10)],
                transforms=NoTransform(),
                base_path='some-base-path'
            )
        return Result()

    def test_init(self, result):
        assert isinstance(result.dataset,BaseCreativeDataset)
        assert isinstance(result.dataset.transforms, nn.Module)
        assert isinstance(result.dataset.file_names, list)
        assert result.dataset.file_names[0]=='0'
        assert len(result.dataset)==10

    @patch('models.common.base_dataset.os')
    @patch('models.common.base_dataset.torch')
    def test_get_item(self,mock_torch, mock_os,result):
        mock_torch.load.return_value = (1,1)
        value = result.dataset.__getitem__(id=0)
        assert value == (1,1)
        mock_os.path.join.assert_called_once()
        mock_torch.load.assert_called_once()



