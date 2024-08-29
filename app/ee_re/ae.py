import numpy as np
import sys
from pathlib import Path

import torch
from torchvision import transforms
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.visualize import make_graph

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer
# from trainer import Model, MyMultiTrain
from trans import Trans

from models.csg_nets import AutoEncoder
from trainer_ae import AETrainer

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_ds = ds("cifar10_train", transform_l=[transforms.ToTensor()])
# train_ds = ds("cifar10_train", transform_l=[transforms.Resize((32, 32)), transforms.ToTensor()])
# train_dl = dl(train_ds, batch_size=1024, shuffle=False)
ae_dl = dl(train_ds, batch_size=32, shuffle=False)


# exp_name = "exp_csg_tmp"
exp_name = "exp_csg"
ae_epochs = 100
lr = 0.001

run_mgr = RunManager(exc_path=__file__, exp_name=exp_name)
run_mgr.log_param("lr", lr)

ae = AutoEncoder(train_ds[0][0].shape, force=False)
# optimizer = torch.optim.SGD(ae.parameters(), lr=0.03)
optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ae_epochs, eta_min=0, last_epoch=-1), "epoch")
loss_func = torch.nn.MSELoss()
ae_model = AETrainer(ae, optimizer=optimizer, loss_func=loss_func, scheduler_t=None, device=device)
# ae_model = AETrainer(ae, optimizer=optimizer, loss_func=loss_func, scheduler_t=scheduler_t, device=device)

for e in range(ae_epochs):
    train_loss = ae_model.train_1epoch(ae_dl)
    met_dict = {"epoch": e + 1, "train_loss": train_loss}

    ae_model.printlog(met_dict, e + 1, ae_epochs, itv=ae_epochs / 4)
    run_mgr.ref_stats(itv=5, step=e + 1, last_step=ae_epochs)
    
run_mgr.log_torch_save(ae_model.get_sd(), "ae_sd.pt")
    
emb_dl = dl(train_ds, batch_size=128)
embs, labels = ae_model.fetch_embs(emb_dl)
# images, labels = next(iter(emb_dl))

# 画像をベクトルにフラット化
X = embs.view(embs.shape[0], -1).numpy()  # 画像をフラット化してnumpy配列に変換
tsne = TSNE(n_components=3, random_state=0, verbose=1, n_jobs=2, n_iter=10000)
X = tsne.fit_transform(X)
y = labels.numpy()  # ラベルをnumpy配列に変換


# CumulativeGradientEstimatorの初期化
estimator = CumulativeGradientEstimator(M_sample=100, k_nearest=3)

# CIFAR-10データに対してエステメータをフィット
estimator.fit(data=X, target=y)

# Cumulative Spectral Gradient (CSG)の値を取得
print(estimator.csg)
# print(estimator.evals)
# print(estimator.evecs)

# You can plot the dataset with:
# make_graph(estimator.difference, title="Your dataset", classes=["A", "B", "C"])