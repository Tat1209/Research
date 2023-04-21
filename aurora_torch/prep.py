import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw


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
    def __init__(self, data_path, batch_size, train_ratio=1.0, color=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.color = color

        self.data_num = len(torchvision.datasets.ImageFolder(root=self.data_path["labeled"]))
        self.rand_idxs = list(range(self.data_num))
        random.shuffle(self.rand_idxs)
        self.num_train = int(self.data_num * train_ratio)


    def fetch_train(self, *args):
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform(args))
        if self.color: args += ("color", )
        if self.train_ratio is not None: ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:self.num_train])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_val(self, *args):
        if self.num_train >= self.data_num: return None
        if self.color: args += ("color", )
        ds = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], transform=transform(args))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[self.num_train:])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_test(self, *args):
        if self.color: args += ("color", )
        ds = TestDataset(self.data_path["unlabeled"], transform(args))
        dl = self.fetch_loader(ds)
        return dl


    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        

"""
    def fetch_normdata(self):
        self.fetch_test():
        mean = {"r":0., "g":0., "b":0.}
        std = {"r":0., "g":0., "b":0.}

        for input_b, label_b in dl:
        
        avg_loss = stats["total_loss"] / len(dl.dataset)
        acc = stats["total_corr"] / len(dl.dataset)
"""




