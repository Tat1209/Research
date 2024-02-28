import torch
import torchvision

# from torchvision.transforms import v2 as transforms

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


class Trans:
    tsr = transforms.ToTensor()
    pil = transforms.ToPILImage()
    scale_rgb = transforms.Lambda(lambda image: image / 255.0)
    permute = transforms.Lambda(lambda tsr: tsr.permute(2, 0, 1))

    cf_norm = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        inplace=True,
    )
    in_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    as_norm = transforms.Normalize(
        mean=[0.18503567576408386, 0.27679356932640076, 0.43360984325408936],
        std=[0.08373230695724487, 0.07494986057281494, 0.06476051360368729],
        inplace=True,
    )

    np_trance = transforms.Lambda(lambda x: -x)
    color = transforms.Lambda(lambda image: image.convert("RGB"))
    rotate90 = transforms.Lambda(lambda image: rotate(image, 90))
    rotate180 = transforms.Lambda(lambda image: rotate(image, 180))
    rotate270 = transforms.Lambda(lambda image: rotate(image, 270))
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)

    def cf_raug(**kwargs):
        torchvision.transforms.RandAugment(**kwargs)

    def repeat_data(n):
        # torch.tileだと、次元数をかえたときにタプルの記述を変えなければいけないため(1d => (n, 1), 2d => (n, 1, 1)、その必要が無いようにcatで実装
        return transforms.Lambda(lambda tensor: torch.cat([tensor for _ in range(n)], dim=0))

    def rotate(th):
        return transforms.Lambda(lambda image: rotate(image, th))

    def resize(h, w):
        return torchvision.transforms.Resize((h, w), antialias=True)  # antialias を True にしないと、warning がでる

    # flipや回転など、PILの状態で操作しなければいけないものもあり、それらはテンソルにする前に行う必要がある。
    rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
    light_aug = [transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]

    tsr_l = [tsr]

    cf_gen = [tsr, cf_norm]
    cf_crop = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip(), tsr, cf_norm]
    cf_git = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, cf_norm]

    in_gen = [torchvision.transforms.Resize((224, 224)), tsr, in_norm]

    as_gen = [tsr, scale_rgb, as_norm]
    as_da = [tsr, scale_rgb, transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), as_norm]
    as_da2 = [tsr, scale_rgb, transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR), as_norm]
    as_da3 = [tsr, scale_rgb, transforms.ColorJitter(brightness=0.15, contrast=0.15), transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR), as_norm]
    as_da4 = [tsr, scale_rgb, *rflip, transforms.ColorJitter(brightness=0.08, contrast=0.08), transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.93, 1.07), interpolation=InterpolationMode.BILINEAR), as_norm]
