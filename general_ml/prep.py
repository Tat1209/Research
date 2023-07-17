import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw
import pandas as pd


class Prep:
    def __init__(self, batch_size, val_range=None, seed=None):
        self.batch_size = batch_size
        
        base_ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

        self.data_num = len(base_ds)
        self.rand_idxs = list(range(self.data_num))
        random.seed(seed)
        random.shuffle(self.rand_idxs)

        # self.val_range = (0, 1)


    def fetch_train(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        # idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        # ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:idx[0]]+self.rand_idxs[idx[1]:])
        dl = self.fetch_loader(ds)
        
        return dl


    def fetch_val(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        # if self.val_range[0] == self.val_range[1]: return None
        # ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        # idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        # ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[idx[0]:idx[1]])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_test(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        dl = self.fetch_loader(ds)
        return dl


    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        




