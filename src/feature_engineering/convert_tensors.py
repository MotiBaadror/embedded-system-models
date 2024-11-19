import os

import torch
from PIL import Image
from torchvision import transforms

from feature_engineering.base_feature_config import BaseFeatureConfig, ClassMapping
from pytorch_model_framework.io_strategy import LocalStrategy

def get_transforms():
    transforms_list = transforms.Compose(
        [
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ]
    )
    return transforms_list


class ConvertTensorHandler:
    def __init__(self, config:BaseFeatureConfig):
        self.config = config
        self.transforms_list = get_transforms()

    def read_image(self,path):
        image = Image.open(path).convert("RGB")
        return image

    def transform_image(self, image):
        return self.transforms_list(image)

    def save_file(self,tensor, out_path):
        torch.save(tensor,out_path)



def convert_tensors(cls, base_path, config:BaseFeatureConfig):
    handler =ConvertTensorHandler(
        config=config
    )
    os.makedirs(config.output_dir, exist_ok=True)
    cls_path = os.path.join(base_path, cls)
    files = os.listdir(cls_path)
    for file in files:
        out_file = os.path.join(config.output_dir,f'{file[:-4]}.pt')
        image = handler.read_image(
            os.path.join(cls_path,file)
        )
        transformed_image = handler.transform_image(image)
        handler.save_file(
            (torch.tensor(ClassMapping[cls].value,dtype=torch.float64),transformed_image),
            out_file
        )
        # break






