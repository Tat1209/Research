import sys

import torch
import torchvision

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager
from trainer import Model, Ens, MultiTrain
from trans import Trans
import utils

work_path = "/home/haselab/Documents/tat/Research/"
ds = Datasets(root=f"{work_path}assets/datasets/")

base_ds = ds("ai-step_l", u_seed=3).shuffle().transform(Trans.as_gen)
# base_ds = ds("ai-step_l", u_seed=3).shuffle().transform(Trans.as_gen)
# tmp_ds = base_ds.in_range(0.8)
# tmp_ds2 = base_ds.ex_range(0.8)
tmp_ds, tmp_ds2 = base_ds.split(ratio=0.7, shuffle=True, balance_label=True)
print(tmp_ds.fetch_weight())


# print(len(tmp_ds))
# label_l, label_d = tmp_ds.fetch_ld(output=True)
# print(label_d)
# print(label_l)

# print(base_ds.indices)
# print(tmp_ds.indices)
# print(tmp_ds2.indices)


def print_labels(dl, file=None):
    labels = None
    for input, label in dl:
        if labels is None:
            labels = label
        else:
            labels = torch.cat([labels, label])
    print(input.shape)

    if file:
        with open(file, "w") as fh:
            print(labels.tolist(), file=fh)
    else:
        print(labels)


loader = dl(base_ds, batch_size=500, shuffle=False)
print_labels(loader)
loader = dl(tmp_ds, batch_size=500, shuffle=False)
print_labels(loader)
loader = dl(tmp_ds2, batch_size=500, shuffle=False)
print_labels(loader)


# loader = dl(tmp_ds, batch_size=100, shuffle=False)
# print_labels(loader)
# loader = dl(tmp_ds2, batch_size=100, shuffle=False)
# print_labels(loader)
