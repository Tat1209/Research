import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset


class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.clone().detach()
        target = index
        # target = 1
        return data, target

    def __len__(self):
        return 10000


class Datasets:
    '''
        Ex.)
        ds = Datasets(root='dir/to/datasets/')
        train_loader = dl(ds("cifar100_train", train_trans), batch_size, shuffle=True)
        for input, label in train_loader:
        ...
    '''
    def __init__(self, root=None):
        self.root = root
        
    
    def _fetch_base_ds(self, ds_str, transform):
        match(ds_str): 
            case "cifar100_train": return torchvision.datasets.CIFAR100(root=self.root, train=True, transform=transform)
            case "cifar100_val": return torchvision.datasets.CIFAR100(root=self.root, train=False, transform=transform)
            case "cifar10_train": return torchvision.datasets.CIFAR10(root=self.root, train=True, transform=transform)
            case "cifar10_val": return torchvision.datasets.CIFAR10(root=self.root, train=False, transform=transform)
            case "caltech": return torchvision.datasets.Caltech101(root=self.root, target_type="category", transform=transform)
            case "imagenet": return torchvision.datasets.ImageNet(root=self.root, split='val', transform=transform)
            case "fix_rand": return FixRandomDataset((3, 32, 32))

            case _: raise Exception("Invalid name.")


    def __call__(self, ds_str, transform_l, seed=None, in_range=1.0, ex_range=None):
        # seed はデータセットの順番 "arange" は並び替えなし
        transform = torchvision.transforms.Compose(transform_l)
        ds = self._fetch_base_ds(ds_str, transform)

        data_num, idx_list = self._idx_setter(ds, seed)
        
        if ex_range is None:
            if not isinstance(in_range, tuple): in_range = (0, in_range)
            idx_range = (int(in_range[0] * data_num), int(in_range[1] * data_num))
            indices = idx_list[idx_range[0]:idx_range[1]]

        else:
            if not isinstance(ex_range, tuple): ex_range = (0, ex_range)
            idx_range = (int(ex_range[0] * data_num), int(ex_range[1] * data_num))
            indices = idx_list[:idx_range[0]]+idx_list[idx_range[1]:]

        sds = Subset(ds, indices=indices)
        sds.ds_name = ds.__class__.__name__
        
        return sds


    def _idx_setter(self, dataset, seed):
        data_num = len(dataset)
        
        if seed == "arange": idx_list = torch.arange(data_num)
        else:
            if seed is not None: torch.manual_seed(seed)
            idx_list = torch.randperm(data_num)
        
        return data_num, idx_list



def dl(ds, batch_size, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)



# # torch.manual_seedを避けたいとき
# def _idx_setter(self, dataset, seed):
#     data_num = len(dataset)
#     idx_list = list(range(data_num))
#     if seed != "arange":
#         random.seed(seed)
#         random.shuffle(idx_list)
    
#     return data_num, idx_list


# return _Loader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
# class _Loader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         self.args_init = args
#         self.kwargs_init = kwargs
#         super().__init__(*args, **kwargs)
        

#     def __repr__(self) -> str:
#         format_string = f'{self.__class__.__name__}(\n'

#         for value in self.args_init:
#             format_string += f'dataset = {value.__class__.__name__}()\n'    # 必ず dataset の引数が入る

#         for key, value in self.kwargs_init.items():
#             if key == 'dataset': format_string += f'{getattr(self, key).__class__.__name__}()\n'
#             else: format_string += f"{key} = {value}\n"
#         format_string += ')'
#         return format_string


# class _Subset(Subset):
#     def __init__(self, *args, **kwargs):
#         # self.args_init = args
#         # self.kwargs_init = kwargs
#         super().__init__(*args, **kwargs)
#         self.ds_name = self.dataset.ds_name

#     def __repr__(self) -> str:  
#         format_string = self.__class__.__name__ + ' (\n'
#         for attr in dir(self):
#             if not attr.startswith("_") and not callable(getattr(self, attr)): # exclude special attributes and methods
#                 value = getattr(self, attr)
#                 format_string += f"{attr} = {value}\n"
#         format_string += ')'
#         return format_string

    





