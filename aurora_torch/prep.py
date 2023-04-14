import os

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw



def transform(shape=None):
    def blacken_region(x1, y1, x2, y2):
        def transform(image):
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], fill=0)
            return image
        return transform

    transformer = transforms.Compose([
        # transforms.CenterCrop(min(shape[-2], shape[-1])),
        transforms.CenterCrop(85),
        # transforms.Resize(shape),
        transforms.Lambda(blacken_region(0, 0, 24, 5)),
        transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
        transforms.ToTensor(), 
    ])

    return transformer



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
    def __init__(self, data_path, batch_size, shape=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shape = shape


    def fetch_train_val(self):
        train_ds = torchvision.datasets.ImageFolder(root=self.data_path["train_val"], loader=lambda img_path: Image.open(img_path), transform=transform(self.shape))
        train_dl = self.batch_processing(train_ds)
        return train_dl


    def fetch_test(self):
        test_ds = TestDataset(self.data_path["test"], transform(self.shape))
        test_dl = self.batch_processing(test_ds)
        return test_dl
    
    
    def batch_processing(self, dataset):
        # return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

        


