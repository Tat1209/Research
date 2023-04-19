import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw


def transform(aug=False, rgb=True):
    def blacken_region(x1, y1, x2, y2):
        def transform(image):
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], fill=0)
            return image
        return transform

    def convert_to_rgb():
        def transform(image): return image.convert('RGB')
        return transform


    pipe_gen = [
            transforms.CenterCrop(85),
            transforms.Lambda(blacken_region(0, 0, 24, 5)),
            transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
            ]
    if rgb: pipe_gen += [transforms.Lambda(convert_to_rgb())]

    if aug: 
        pipe = [
                transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(), 
                ]
    else:
        pipe = [ transforms.ToTensor(), ]

    transform = pipe_gen + pipe

    return transforms.Compose(transform)



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.img_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        self.transform = transform


    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img_tensor = self.transform(img)

        return img_tensor, img_path


    def __len__(self):
        return len(self.img_paths)



class Prep:
    def __init__(self, data_path, batch_size, train_ratio=1.0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.data_num = len(torchvision.datasets.ImageFolder(root=self.data_path["labeled"]))
        self.rand_idxs = list(range(self.data_num))
        random.shuffle(self.rand_idxs)
        self.num_train = int(self.data_num * train_ratio)


    def fetch_train(self, aug=False, rgb=False):
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform(aug=aug, rgb=rgb))
        if self.train_ratio is not None: ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:self.num_train])
        dl = self.data_load(ds)
        
        return dl


    def fetch_val(self, aug=False, rgb=False):
        if self.num_train == self.data_num: return None
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform(aug=aug, rgb=rgb))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[self.num_train:])
        dl = self.data_load(ds)

        return dl


    def fetch_test(self, aug=False, rgb=False):
        ds = TestDataset(self.data_path["unlabeled"], transform(aug, rgb))
        dl = self.data_load(ds)
        return dl


    def data_load(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)




