import random

import torch
import torchvision


working_dir = "/home/haselab/Documents/tat/Research/"
root=f"{working_dir}assets/datasets/"

torchvision.datasets.CIFAR100(root=root, train=True, download=True)
torchvision.datasets.CIFAR100(root=root, train=False, download=True)

