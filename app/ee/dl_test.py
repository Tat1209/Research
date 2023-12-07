import torch
from datasets import Datasets, dl
from trans import Trans



work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

# loader = dl(ds("fix_rand", [Trans.tsr], seed=5, in_range=(0, 0.001)), batch_size=5, shuffle=False)
loader = dl(ds("cifar10_train", [Trans.tsr], seed='arange', in_range=0.001), batch_size=10, shuffle=False)



# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)

labels = None

for input_b, label_b in loader:
    if labels is None: labels = label_b
    else: labels = torch.cat([labels, label_b])
    
labels = torch.stack([labels for i in range(2)])
    
print(torch.numel(torch.unique(labels)))

torch.Tensor()
# print(labels)



    



