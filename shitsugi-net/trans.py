import os
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from PIL import Image
from PIL import ImageDraw



class Trans:
    def __init__(self, info=None):

        if info is None: info = {'mean':[0.5, 0.5, 0.5], 'std':[0.25, 0.25, 0.25]}

        # baseには、全ての訓練に対して適用する処理を記述する。
        self.base = []
        self.tsr = [transforms.ToTensor()]
        self.norm = [transforms.Normalize(mean=info["mean"], std=info["std"])]

        self.rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
        self.flip90 = [transforms.Lambda(lambda image: rotate(image, 90))]
        self.flip180 = [transforms.Lambda(lambda image: rotate(image, 180))]
        self.flip270 = [transforms.Lambda(lambda image: rotate(image, 270))]
        self.hflip = [transforms.RandomHorizontalFlip(p=1)]
        self.vflip = [transforms.RandomVerticalFlip(p=1)]
        self.rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        # self.color = [transforms.Lambda(convert_to_rgb())]
        
        # flipとか回転は、PILの状態で操作しなければいけないものもある。テンソルにする前に行う必要がある。
        self.base_tsr = self.compose(self.base + self.tsr)
        self.gen = self.compose(self.base + self.tsr + self.norm)
        self.aug = self.compose(self.base + self.rotate_flip + self.tsr + self.norm)
        self.flip_aug = self.compose(self.base + self.rflip + self.tsr + self.norm)


    def compose(self, args):
        return transforms.Compose(args)

