import sys
from pathlib import Path

import torch

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from trans import Trans

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# tmp = ds("cifar100_train", transform_l=[Trans.tsr])

# print(ds("mnist_train", transform_l=[Trans.tsr]).calc_min_max(formatted=True))
# print(ds("cifar10_train", transform_l=[Trans.tsr]).calc_min_max(formatted=True))
print(ds("mnist_train", transform_l=[Trans.tsr]).calc_mean_std(formatted=True))
