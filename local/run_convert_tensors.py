import os

from feature_engineering.convert_tensors import convert_tensors, BaseFeatureConfig



config = BaseFeatureConfig(
    input_version=0,
    output_version=0,
    base_path='data/raw_data',
    output_dir='data/tensors'
)

for cls in os.listdir(config.base_path):
    convert_tensors(
        cls=cls,
        config=config,
    )

