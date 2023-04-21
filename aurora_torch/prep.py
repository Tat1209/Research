import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw

from trans import Trans



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.img_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        self.transform = transform


    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        if self.transform is None: return 0, img_path
        img_tensor = self.transform(img)
        return img_tensor, img_path


    def __len__(self):
        return len(self.img_paths)



class Prep:
    def __init__(self, data_path, batch_size, train_ratio=1.0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        
        base_ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=None)
        self.tr = Trans(None, base_ds, batch_size)

        self.data_num = len(base_ds)
        self.rand_idxs = list(range(self.data_num))
        random.shuffle(self.rand_idxs)
        self.num_train = int(self.data_num * train_ratio)


    def fetch_train(self, transform):
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform)
        if self.train_ratio is not None: ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:self.num_train])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_val(self, transform):
        if self.num_train >= self.data_num: return None
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform)
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[self.num_train:])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_test(self, transform):
        ds = TestDataset(self.data_path["unlabeled"], transform)
        dl = self.fetch_loader(ds)
        return dl


    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        



