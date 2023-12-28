import time

import utils
from pathlib import Path
import torch
from datasets import Datasets, dl
from trans import Trans
import torchvision



work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")
# tmp_ds = ds("fix_rand", [Trans.resize(12, 12)], seed="arange", label_balance=True)
# r = ((0.2, 0.4))
# ds_a = tmp_ds.in_range(r).transform([Trans.resize(25, 25)])
# ds_b = tmp_ds.ex_range(r).transform([Trans.resize(15, 15)])
# print(len(ds_a))
# print(len(ds_b))
# loader = dl(ds_a, batch_size=5, shuffle=False)
# loader_b = dl(ds_b, batch_size=5, shuffle=False)

loader = dl(ds("cifar100_train", [Trans.tsr], seed='arange', label_balance=False).in_range(1), batch_size=125, shuffle=False)
# loader = dl(ds("caltech", [Trans.color, Trans.tsr, Trans.resize(1, 1)], label_balance=True, seed='arange', in_range=0.001), batch_size=100, shuffle=False)


# labels = None

start = time.time()
for input_b, label_b in loader:
    input_b.to('cuda')
    pass
print(time.time() - start)

start = time.time()
for input_b, label_b in loader:
    input_b.to('cuda')
    pass
print(time.time() - start)

#     if labels is None: labels = label_b
#     else: labels = torch.cat([labels, label_b])
    
# print(labels)

# labels = None
# for input_b, label_b in loader_b:
#     if labels is None: labels = label_b
#     else: labels = torch.cat([labels, label_b])
    
# print(labels)

    
    
    

# print(len(labels.unique()))


# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)

    





