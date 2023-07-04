import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw
import pandas as pd

from trans import Trans


class Prep:
    def __init__(self, batch_size, val_range=0., seed=None):
        self.batch_size = batch_size
        
        base_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
        self.tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

        self.data_num = len(base_ds)
        self.rand_idxs = list(range(self.data_num))
        random.seed(seed)
        random.shuffle(self.rand_idxs)

        if not isinstance(val_range, tuple): self.val_range = (0, 0)
        else: self.val_range = val_range


    def fetch_train(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:idx[0]]+self.rand_idxs[idx[1]:])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_val(self, transform):
        if self.val_range[0] == self.val_range[1]: return None
        ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[idx[0]:idx[1]])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_test(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        dl = self.fetch_loader(ds)
        return dl


    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        




