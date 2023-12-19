from pathlib import Path
import torch
from datasets import Datasets, dl
from trans import Trans



work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

# loader = dl(ds("fix_rand", seed='arange', transform_l=[Trans.tsr], ex_range=(0, 0.001)), batch_size=5, shuffle=False)

# loader = dl(ds("cifar100_train", [Trans.tsr], seed=None, label_balance=False, in_range=0.003), batch_size=100, shuffle=False)
# loader = dl(ds("caltech", [Trans.color, Trans.tsr, Trans.resize(1, 1)], label_balance=True, seed='arange', in_range=0.001), batch_size=100, shuffle=False)

# loader = dl(ds._fetch_base_ds("cifar100_train", Trans.tsr), batch_size=100, shuffle=False)

train_range = (0, 0.2)
# labeled_ds = ds("fix_rand", seed=None, transform_l=[Trans.tsr])
labeled_ds = ds("ai-step_train", seed='arange', label_balance=False)
train_dl = dl(labeled_ds(transform_l=Trans.as_da, in_range=train_range), batch_size=5, shuffle=False)
val_dl = dl(labeled_ds(transform_l=Trans.as_gen, ex_range=train_range), batch_size=5, shuffle=False)

loader = train_dl
loader2 = val_dl

labels = None
for input_b, label_b in loader:
    if labels is None: labels = label_b
    else: labels = torch.cat([labels, label_b])

labels2 = None
for input_b, label_b in loader2:
    if labels2 is None: labels2 = label_b
    else: labels2 = torch.cat([labels2, label_b])
    
print(labels)
print(len(labels))
print(labels2)
print(len(labels2))

# print(len(labels.unique()))


# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)

    





