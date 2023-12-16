from pathlib import Path
import torch
from datasets import Datasets, dl
from trans import Trans



work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

# loader = dl(ds("fix_rand", [Trans.tsr], label_balance=True, in_range=(0, 0.01)), batch_size=5, shuffle=False)
# loader = dl(ds("cifar100_train", [Trans.tsr], seed='arange', label_balance=True, in_range=0.003), batch_size=100, shuffle=False)
loader = dl(ds("caltech", [Trans.color, Trans.tsr, Trans.resize(1, 1)], label_balance=True, seed='arange', in_range=0.001), batch_size=100, shuffle=False)


labels = None
for input_b, label_b in loader:
    if labels is None: labels = label_b
    else: labels = torch.cat([labels, label_b])

print(len(labels.unique()))


# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)

    





