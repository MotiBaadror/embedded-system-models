import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from dir_configs import add_rootpath


class BaseCreativeDataset(Dataset):
    def __init__(self, file_names, transforms, base_path, stage='train'):
        super().__init__()
        self.file_names = file_names
        self.transforms = transforms
        self.base_path = add_rootpath(base_path)

    def __getitem__(self, id):
        x,y = torch.load(os.path.join(self.base_path,self.file_names[id]))
        x = self.transforms(x)
        return (x,y)

    def __len__(self):
        return len(self.file_names)


