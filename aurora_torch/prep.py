import os

import torch
import torchvision
from torchvision import transforms
from PIL import Image


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, transformer):
        self.img_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
        self.transformer = transformer
        

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img_tensor = self.transformer(img)

        return img_tensor, img_path
    

    def __len__(self):
        return len(self.img_paths)



class Prep:
    def __init__(self, data_path, batch_size, shape):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape


    def transform(self):
        transformer = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            ])
        return transformer


    def fetch_train_val(self):
        train_ds = torchvision.datasets.ImageFolder(root=self.data_path["train_val"], loader=lambda img_path: Image.open(img_path), transform=self.transform())
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        return train_dl


    def fetch_test(self):
        test_ds = TestDataset(self.data_path["test"], self.transform())
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        return test_dl


