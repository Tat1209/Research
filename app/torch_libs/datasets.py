from pathlib import Path
import random
import itertools
import pickle
from PIL import Image
from copy import copy

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from trans import Trans


class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.detach().clone()
        target = index
        # target = 1
        return data, target

    def __len__(self):
        return 10000


class PklToDataset(Dataset):
    def __init__(self, pkl_path, transform=None, target_transform=None):
        with open(pkl_path, "rb") as f:
            (self.datas, self.targets) = pickle.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        target = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

class TinyImageNet(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        ti_path = root / Path("tiny-imagenet-200")
        self.paths = []
        self.targets = []
        self.dirname_label = {}
        
        if not ti_path.exists():
            raise FileNotFoundError
        
        # wnids.txtを参照し、クラス名と値の対応付けを行う。その後、self.dirname_labelに対応を格納
        with open(ti_path / Path("wnids.txt"), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                key = line.strip()
                value = i
                self.dirname_label[key] = value
        
        if train:
            ti_train_path = ti_path / Path("train")
            for class_dir in ti_train_path.iterdir():
                if class_dir.is_dir():
                    images_dir = class_dir / Path("images")

                    for image_path in images_dir.iterdir():
                        if image_path.is_file():
                            self.paths.append(image_path)
                            self.targets.append(self.dirname_label[class_dir.name])
                            
        else:
            ti_val_path = ti_path / Path("val")
            images_dir = ti_val_path / Path("images")

            with open(ti_val_path / Path("val_annotations.txt"), 'r') as f:
                lines = f.readlines()

            for line in lines:
                elems = line.split()
                image_name = elems[0]
                class_name = elems[1]

                image_path = images_dir / Path(image_name)

                if image_path.is_file():
                    self.paths.append(image_path)
                    self.targets.append(self.dirname_label[class_name])
            

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = Image.open(self.paths[idx]).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

class Datasets:
    def __init__(self, root=None):
        self.root = root

    def _base_ds(self, ds_str):
        match (ds_str):
            case "mnist_train":
                return torchvision.datasets.MNIST(root=self.root, train=True)
            case "mnist_val":
                return torchvision.datasets.MNIST(root=self.root, train=False)
            case "cifar100_train":
                return torchvision.datasets.CIFAR100(root=self.root, train=True)
            case "cifar100_val":
                return torchvision.datasets.CIFAR100(root=self.root, train=False)
            case "cifar10_train":
                return torchvision.datasets.CIFAR10(root=self.root, train=True)
            case "cifar10_val":
                return torchvision.datasets.CIFAR10(root=self.root, train=False)
            case "stl10_train":
                return torchvision.datasets.STL10(root=self.root, split="train")
            case "stl10_val":
                return torchvision.datasets.STL10(root=self.root, split="test")
            case "caltech101_trainval":
                return torchvision.datasets.Caltech101(root=self.root, target_type="category")
            case "tiny-imagenet_train":
                return TinyImageNet(root=self.root, train=True)
            case "tiny-imagenet_val":
                return TinyImageNet(root=self.root, train=False)
            case "cars_train":
                return torchvision.datasets.StanfordCars(root=self.root, split="train")
            case "cars_val":
                return torchvision.datasets.StanfordCars(root=self.root, split="test")
            case "pets_train":
                return torchvision.datasets.OxfordIIITPet(root=self.root, split="trainval", target_types="category")
            case "pets_val":
                return torchvision.datasets.OxfordIIITPet(root=self.root, split="test", target_types="category")
            case "flowers_train":
                return torchvision.datasets.Flowers102(root=self.root, split="train")
            case "flowers_val":
                return torchvision.datasets.Flowers102(root=self.root, split="val")
            case "flowers_test":
                return torchvision.datasets.Flowers102(root=self.root, split="test")
            case "imagenet":
                return torchvision.datasets.ImageNet(root=self.root, split="val")
            case "mnist_ddim":
                return torchvision.datasets.ImageFolder(root=str(Path(self.root) / Path("mnist_ddim")))
            case "ai-step_l":
                return PklToDataset(f"{self.root}fukui_train_32_60_ver2.pkl")
            case "ai-step_ul":
                return PklToDataset(f"{self.root}kanazawa_test_32_60_ver2.pkl")
            case "fix_rand":
                return FixRandomDataset((3, 32, 32))
            case _:
                raise Exception("Invalid name.")

    def __call__(self, ds_str, transform_l=[], target_transform_l=[], u_seed=None):
        ds = self._base_ds(ds_str)
        ds.ds_str = ds_str
        ds.ds_name = ds.__class__.__name__
        ds.access = self
        ds.u_seed = u_seed

        indices = np.arange(len(ds))
        transform = torchvision.transforms.Compose(transform_l)
        target_transform = torchvision.transforms.Compose(target_transform_l)

        return DatasetHandler(ds, indices, transform, target_transform)


# クラスの数を減らすなら、indicesのほかにclasses (ラベル名を保管しているリスト) を作ってそれも毎回コピる必要がある。その後、__getitem__のtargetを修正する必要あり
class DatasetHandler(Dataset):
    # self.indicesは、常にnp.array(), label_l, label_dのvalueはlist
    def __init__(self, dataset, indices, transform, target_transform):
        self.dataset = dataset
        self.indices = indices
        self._transform = transform
        self._target_transform = target_transform

        self.ds_str = dataset.ds_str
        self.ds_name = dataset.ds_name

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]

        if self._transform:
            data = self._transform(data)
        if self._target_transform:
            target = self._target_transform(target)

        return data, target

    def __len__(self):
        return len(self.indices)

    def shuffle(self, seed=None):
        # データセットそのものの順序をシャッフル ただし、ロードごとにシャッフルしたいならDataLoaderでシャッフルさせるべき
        indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        tmp = self.dataset.u_seed
        if tmp is not None:
            seed = tmp

        if seed != "arange":
            if seed is not None and not isinstance(seed, int):
                raise TypeError("Variables must be of type int, 'None', or the string 'range'.")
            np.random.seed(seed)
            np.random.shuffle(indices_new)

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def in_ratio(self, a, b=None):
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)
        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        data_num = len(self.indices)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = self.indices[idx_range[0] : idx_range[1]]

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def ex_ratio(self, a, b=None):
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        data_num = len(self.indices)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = list(self.indices[: idx_range[0]]) + list(self.indices[idx_range[1] :])
        indices_new = np.array(indices_new)

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)
    
    def split_ratio(self, ratio, balance_label=False, seed=None):
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        if balance_label:
            label_l, label_d = self.fetch_ld()
            a_d = {}
            b_d = {}
            for key in label_d:
                lst = label_d[key]
                if seed != "arange":
                    if seed is not None and not isinstance(seed, int):
                        raise TypeError("Variables must be of type int, 'None', or the string 'arange'.")
                    np.random.seed(seed)
                    np.random.shuffle(lst)
                length = len(lst)
                a_idx = lst[: int(length * ratio)]
                b_idx = lst[int(length * ratio) :]

                a_d[key] = a_idx
                b_d[key] = b_idx

            indices_a_new = np.array(list(itertools.chain(*a_d.values())), dtype=np.int32)
            indices_b_new = np.array(list(itertools.chain(*b_d.values())), dtype=np.int32)
            
            indices_a_new.sort()
            indices_b_new.sort()

        else:
            indices_new = self.indices.copy()
            if seed != "arange":
                if seed is not None and not isinstance(seed, int):
                    raise TypeError("Variables must be of type int, 'None', or the string 'arange'.")
                np.random.seed(seed)
                np.random.shuffle(indices_new)
            length = len(indices_new)
            indices_a_new = indices_new[: int(length * ratio)]
            indices_b_new = indices_new[int(length * ratio) :]

        a = DatasetHandler(self.dataset, indices_a_new, transform_new, target_transform_new)
        b = DatasetHandler(self.dataset, indices_b_new, transform_new, target_transform_new)

        return a, b

    def transform(self, transform_l):
        indices_new = self.indices.copy()
        # transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        transform_new = torchvision.transforms.Compose(transform_l)
        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def target_transform(self, target_transform_l):
        indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        # target_transform_new = copy(self._target_transform)

        target_transform_new = torchvision.transforms.Compose(target_transform_l)
        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def __add__(self, other):
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        indices_new = np.concatenate((self.indices, other.indices))

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def balance_label(self, seed=None):
        # len(classes)ごとに取り出したとき、常に要素の数が極力均等になるようにデータセットのincicesを構成
        # seed="arange"で、該当indeicesをクラスが若い順から順番に、indicesの小さい順でとってくる
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        tmp = self.dataset.u_seed
        if tmp is not None:
            seed = tmp

        if seed != "arange":
            np.random.seed(seed)

        label_l, label_d = self.fetch_ld()

        # 各クラスのラベル数を取得・クラス名を取得
        lens_d = {k: len(v) for k, v in label_d.items()}
        class_d = lens_d.keys()

        # クラス名だけで構成されたclass_arrayを作成
        class_array = []
        counter = lens_d.copy()
        while True:
            valid_keys = [c for c in class_d if 0 < counter[c]]  # len_dのkeyのうち、まだvalue個格納されていないものを選択する
            if max(counter.values()) == 0:
                break
            for _ in range(min([labels for labels in counter.values() if 0 < labels])):
                m = len(valid_keys)
                perm = np.arange(m)
                if seed != "arange":
                    np.random.shuffle(perm)
                shuffled_keys = [valid_keys[p] for p in perm]

                for key in shuffled_keys:  # シャッフルされたkeyに対してループする
                    class_array.append(key)
                    counter[key] -= 1

        # label_dのすべてのkeyについて、その要素を全て並び替えたshuffled_label_dを作成
        shuffled_label_d = dict()
        for key in label_d.keys():
            perm = np.array(label_d[key])
            if seed != "arange":
                perm = np.random.permutation(perm)
            shuffled_label_d[key] = perm

        # class_arrayでクラス、shuffled_label_dでindexを取得
        indices_new = []
        for key in class_array:
            value = shuffled_label_d[key][0]  # label_d[key]のリストから先頭の要素を取り出す
            indices_new.append(value)
            shuffled_label_d[key] = shuffled_label_d[key][1:]  # shuffled_label_d[key]のリストから先頭の要素を削除する

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)

    def mult_label(self, mult_dict=None, seed=None):
        # indices_new = self.indices.copy()
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        tmp = self.dataset.u_seed
        if tmp is not None:
            seed = tmp

        if seed != "arange":
            np.random.seed(seed)

        label_l, label_d = self.fetch_ld()
        for k, v in mult_dict.items():
            label_d[k] *= v
        indices_new = [item for sublist in label_d.values() for item in sublist]

        if seed == "arange":
            indices_new.sort()
        else:
            np.random.seed(seed)
            np.random.shuffle(indices_new)

        return DatasetHandler(self.dataset, indices_new, transform_new, target_transform_new)
    
    def fetch_classes(self, base_classes=False):
        blabel_l, blabel_d = self.fetch_base_ld()
        return len(blabel_d)
        

    def fetch_ld(self, output=False):
        # try:
        #     label_l = torch.tensor(dl.dataset.dataset.targets)
        # except AttributeError:

        blabel_l, blabel_d = self.fetch_base_ld()

        label_l = []  # label のリストを作成
        label_d = dict()  # index と対応させ、label を key とし、index を item とした dict を作成
        for idx in self.indices:
            label = blabel_l[idx]
            label_l.append(label)
            if label_d.get(label) is None:
                label_d[label] = [idx]
            else:
                label_d[label].append(idx)

        label_d = dict(sorted(label_d.items()))

        if output:
            for label in label_d.keys():
                print(f"{label}: {len(label_d[label])} items")

        return label_l, label_d

    def fetch_weight(self, base_classes=False, num_classes=-1):
        """
        Ex.)
        loss_func = torch.nn.CrossEntropyLoss(weight=train_ds.fetch_weight(base_classes=True).to(device))
        """
        label_l, label_d = self.fetch_ld()
        if base_classes:
            blabel_l, blabel_d = self.fetch_base_ld()
            classes = max(blabel_d.keys()) + 1

        elif num_classes != -1:
            classes = num_classes

        else:
            classes = max(label_d.keys()) + 1
        label_count_iv = [1.0 / len(label_d.get(i, [])) for i in range(classes)]  # インデックスが数字以外だと機能しない
        weight_tsr = torch.tensor(label_count_iv, dtype=torch.float) / sum(label_count_iv) * classes
        
        print(weight_tsr)

        return weight_tsr

    def fetch_base_ld(self):
        try:
            ld_path = self.dataset.access.root / (self.ds_str + ".ld")
            label_l, label_d = torch.load(ld_path)
        except FileNotFoundError:
            label_l, label_d = self._save_labels()

        return label_l, label_d

    def _save_labels(self):
        base_dsh = self.dataset.access(self.ds_str, u_seed="arange")  # DatasetHandler().fetch_ldを使うため、一時的に作成
        # base_ds = self.dataset.access._base_ds(ds_str)
        label_l, label_d = base_dsh._make_ld()

        save_obj = (label_l, label_d)
        ld_path = self.dataset.access.root / (self.ds_str + ".ld")
        torch.save(save_obj, ld_path)
        print(f"Saved label data to the following path: {ld_path}")

        return label_l, label_d

    def _make_ld(self):
        # fetch_ldほぼ同じだが、ところどころ違うので別で定義した方が楽そう
        # base_dsh = self.dataset.access(self.ds_str, seed="arange")  # base_dshインスタンスを作成し、selfで呼び出すことを前提

        label_l = []  # label のリストを作成
        label_d = dict()  # index と対応させ、label を key とし、index を item とした dict を作成
        for idx in self.indices:
            _, label = self.dataset[idx]
            label_l.append(label)
            if label_d.get(label) is None:
                label_d[label] = [idx]
            else:
                label_d[label].append(idx)

        label_d = dict(sorted(label_d.items()))

        return label_l, label_d

    def calc_mean_std(self, batch_size=256, formatted=False):
        elem = None
        for input, _ in dl(self, batch_size, shuffle=False):
            # p は、バッチの次元を除いたものが2次元データなら(1, 0, 2, 3)、1次元データなら(1, 0, 2)
            p = torch.arange(len(input.shape))
            p[0], p[1] = 1, 0
            p = tuple(p)

            elem_b = input.permute(p).reshape(input.shape[1], -1)
            if elem is None:
                elem = elem_b
            else:
                elem = torch.cat([elem, elem_b], dim=1)

        mean = elem.mean(dim=1).tolist()
        std = elem.std(dim=1).tolist()

        if formatted:
            return f"transforms.Normalize(mean={mean}, std={std}, inplace=True)"
        else:
            return {"mean": mean, "std": std}

    def calc_min_max(self, batch_size=256, formatted=False):
        elem = None
        for input, _ in dl(self, batch_size, shuffle=False):
            # p は、バッチの次元を除いたものが2次元データなら(1, 0, 2, 3)、1次元データなら(1, 0, 2)
            p = torch.arange(len(input.shape))
            p[0], p[1] = 1, 0
            p = tuple(p)

            elem_b = input.permute(p).reshape(input.shape[1], -1)
            if elem is None:
                elem = elem_b
            else:
                elem = torch.cat([elem, elem_b], dim=1)

        min = elem.min(dim=1).values.tolist()
        max = elem.max(dim=1).values.tolist()

        if formatted:
            return f"transforms.Normalize(min={min}, max={max}, inplace=True)"
        else:
            return {"min": min, "max": max}



def dl(ds, batch_size, shuffle=True, num_workers=2, pin_memory=True, **kwargs):
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, **kwargs)
