import os
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw



class Trans():
    def __init__(self, info=None):
        if info == None: self.info = Trans.fetch_normal()
        else: self.info = info
    
        
    def base(self):
        def blacken_region(x1, y1, x2, y2):
            def transform(image):
                draw = ImageDraw.Draw(image)
                draw.rectangle([x1, y1, x2, y2], fill=0)
                return image
            return transform

        # def convert_to_rgb():
        #     def transform(image): return image.convert('RGB')
        #     return transform

        pipe = [
                transforms.CenterCrop(85),
                transforms.Lambda(blacken_region(0, 0, 24, 5)),
                transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
                ]

        return pipe
    

    def tsr(self):
        pipe = self.base()
        pipe += [transforms.ToTensor()]
        return pipe

    
    def gen(self):
        pipe = self.tsr()
        pipe += [transforms.Normalize(mean=self.info["mean"], std=self.info["std"])]
        return pipe


    def aug(self):
        pipe += [
                transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5), 
                ]

        if "color" in args: pipe += [transforms.Lambda(convert_to_rgb())]

        pipe += [
                transforms.ToTensor(), 
                
                ]

    def transform(args):
        return transforms.Compose(pipe)

    @classmethod
    def fetch_normal(cls, base_ds, batch_size):
        base_ds.transform = cls.tsr()
        base_dl = torch.utils.data.DataLoader(base_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。

        pv = None

        for input_b, label_b in base_dl:
            input_b = input_b.to(device)
            label_b = label_b.to(device)

            pv_list = [input_b[:,i,:,:].flatten() for i in range(input_b.shape[-3])]
            pv_tensor = torch.stack(pv_list, dim=0)
            if pv is None: pv = pv_tensor
            else: pv = torch.cat((pv_tensor, pv), dim=1)
        
        info = dict()
        info["mean"] = pv.mean(dim=1).cpu()
        info["std"] = pv.std(dim=1).cpu()
        
        return info