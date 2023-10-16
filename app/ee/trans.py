import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from torchvision.transforms import RandAugment



class Trans:
    def __init__(self):
        def convert_to_rgb():
            def transform(image): return image.convert('RGB')
            return transform

        self.tsr = [transforms.ToTensor()]
        self.cf_norm = [transforms.Normalize(mean=[0.5070751309394836, 0.48654884099960327, 0.44091784954071045], std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967], inplace=True)]
        self.in_norm = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)]

        self.color = [transforms.Lambda(convert_to_rgb())]
        self.res = [torchvision.transforms.Resize((32, 32))]
        self.res224 = [torchvision.transforms.Resize((224, 224))]
        self.rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
        self.light_aug = [transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
        self.crop32 = [transforms.RandomCrop(32, padding=4,padding_mode='reflect'), transforms.RandomHorizontalFlip()]
        self.crop224 = [transforms.RandomCrop(224, padding=4,padding_mode='reflect'), transforms.RandomHorizontalFlip()]
        self.flip90 = [transforms.Lambda(lambda image: rotate(image, 90))]
        self.flip180 = [transforms.Lambda(lambda image: rotate(image, 180))]
        self.flip270 = [transforms.Lambda(lambda image: rotate(image, 270))]
        self.hflip = [transforms.RandomHorizontalFlip(p=1)]
        self.vflip = [transforms.RandomVerticalFlip(p=1)]
        self.rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        self.pil = [transforms.ToPILImage()]

        # flipとか回転は、PILの状態で操作しなければいけないものもある。テンソルにする前に行う必要がある。
        self.cf_gen = self.compose(self.tsr + self.cf_norm)
        self.cf_crop = self.compose(self.crop32 + self.tsr + self.cf_norm)
        # self.aug = self.compose(rotate_flip + tsr + norm)
        # self.laug = self.compose(light_aug + tsr + norm)
        # self.flip_aug = self.compose(rflip + tsr + norm)

        self.in_gen = self.compose(self.res224 + self.tsr + self.in_norm)

        # self.calgen = self.compose(color + res + tsr + norm)
        # self.calgen_2 = self.compose(color + res224 + tsr + norm)
        # self.calcrop = self.compose(color + res224 + crop224 + tsr + norm)
        
    def cf_raug(self, num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.BILINEAR):
        rand_aug = [RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins, interpolation=interpolation)]
        return self.compose(rand_aug + self.tsr + self.in_norm)



    def compose(self, args):
        return transforms.Compose(args)

