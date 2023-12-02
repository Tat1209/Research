import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from torchvision.transforms import RandAugment



class Trans:
    tsr = transforms.ToTensor()
    pil = transforms.ToPILImage()

    cf_norm = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404], inplace=True)
    in_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

    color = transforms.Lambda(lambda image: image.convert('RGB'))
    rotate90 = transforms.Lambda(lambda image: rotate(image, 90))
    rotate180 = transforms.Lambda(lambda image: rotate(image, 180))
    rotate270 = transforms.Lambda(lambda image: rotate(image, 270))
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)

    def cf_raug(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.BILINEAR):
        RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins, interpolation=interpolation)

    # torch.tileだと、次元数をかえたときにタプルの記述を変えなければいけないため(1d => (n, 1), 2d => (n, 1, 1)、その必要が無いようにcatで実装
    def repeat_data(n): return transforms.Lambda(lambda tensor: torch.cat([tensor for _ in range(n)], dim=0))
    def rotate(th): return transforms.Lambda(lambda image: rotate(image, th))

    # flipや回転など、PILの状態で操作しなければいけないものもあり、それらはテンソルにする前に行う必要がある。
    rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
    light_aug = [transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]

    tsr_l = [tsr]

    cf_gen = [tsr, cf_norm] 
    cf_crop = [transforms.RandomCrop(32, padding=4,padding_mode='reflect'), transforms.RandomHorizontalFlip(), tsr, cf_norm] 
    cf_git = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, cf_norm] 

    in_gen = [torchvision.transforms.Resize((224, 224)), tsr, in_norm] 


    # @classmethod
    # def compose(cls, transform_list):
    #     return transforms.Compose(transform_list)

