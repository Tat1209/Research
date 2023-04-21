import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw

def basic_trans():
    pass

def transform(args):
    def blacken_region(x1, y1, x2, y2):
        def transform(image):
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], fill=0)
            return image
        return transform

    def convert_to_rgb():
        def transform(image): return image.convert('RGB')
        return transform


    pipe = [
            transforms.CenterCrop(85),
            transforms.Lambda(blacken_region(0, 0, 24, 5)),
            transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
            ]


    if "aug" in args: 
        pipe += [
                transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5), 
                ]

    if "color" in args: pipe += [transforms.Lambda(convert_to_rgb())]

    pipe += [
            transforms.ToTensor(), 
            
            ]

    return transforms.Compose(pipe)