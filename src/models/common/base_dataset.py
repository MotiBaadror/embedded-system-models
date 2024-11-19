import os

from PIL import Image
from torch.utils.data import Dataset

from dir_configs import add_rootpath


class BaseCreativeDataset(Dataset):
    def __init__(self, file_names, transforms, base_path, stage='train'):
        super().__init__()
        self.file_names = file_names
        self.transforms = transforms
        self.base_path = add_rootpath(base_path)

    def __getitem__(self, file_name):
        image_path = os.path.join(self.base_path,file_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.file_names)


