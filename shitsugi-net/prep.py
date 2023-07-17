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


    def train(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        dl = self.fetch_loader(ds)
        
        return dl


    def val(self, transform):
        ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        dl = self.fetch_loader(ds)

        return dl


    def val_in(self, transform):
        ds = torchvision.datasets.ImageNet(root='./data', split="val", transform=transform)
        dl = self.fetch_loader(ds)

        return dl



    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        




