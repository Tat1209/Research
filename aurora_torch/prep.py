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
        transforms.CenterCrop(85),
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


    def fetch_train(self):
        ds_train = torchvision.datasets.ImageFolder(root=self.data_path["labeled"], loader=lambda img_path: Image.open(img_path), transform=transform(self.shape))
        dl_train = self.batch_processing(ds_train)
        print(f"{len(dl_train.dataset)} datas were fetched to train.")
        return dl_train


    def fetch_test(self):
        ds_test = TestDataset(self.data_path["unlabeled"], transform(self.shape))
        dl_test = self.batch_processing(ds_test)
        print(f"{len(dl_test.dataset)} datas were fetched to test.")
        return dl_test
    
    
    def batch_processing(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

        


