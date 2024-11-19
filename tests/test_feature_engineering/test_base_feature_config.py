from unittest.mock import patch

from feature_engineering.base_feature_config import BaseFeatureConfig


@patch('feature_engineering.base_feature_config.os.makedirs')
def test_feature_config(mock_os_makedirs):
    config = BaseFeatureConfig(
        base_path='some-path',
        input_version=1,
        output_version=1,
        output_dir='some-output-dir',
    )
    assert config.size == [224,224]
    assert isinstance(config.size,list)
    print(config)