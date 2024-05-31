import sys
import math
from pathlib import Path

import torch

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer
from trainer import Model, MyMultiTrain
from trans import Trans
import utils

# from models.resnet_ee import resnet18 as net
# from torchvision.models import resnet18 as net
from models.gitresnet_ee import resnet18 as net


ds = Datasets(root=f"{work_path}assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


base_fi = 32
for fi in [[1, 2, 4, 8, 16, 32]]:
    for di in [0.1, 0.075, 0.05, 0.048, 0.046, 0.044, 0.042, 0.04, 0.038, 0.036, 0.034, 0.032, 0.03, 0.02, 0.015, 0.01]:
        # for di in [1.0, 0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.048, 0.046, 0.044, 0.042, 0.04, 0.038, 0.036, 0.034, 0.032, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025]:
        exp_name = f"exp_{base_fi}"
        runs = [RunManager(exc_path=__file__, exp_name=exp_name) for _ in fi]
        runs_mgr = RunsManager(runs)

        runs_mgr.log_param("max_lr", max_lr := 0.005)  # Adam
        runs_mgr.log_param("epochs", epochs := int(100 / di))
        runs_mgr.log_param("batch_size", batch_size := 125)

        runs_mgr.log_param("ensemble_type", ensemble_type := ["easy", "merge", "pure"][0])
        runs_mgr.log_param("train_trans", repr(train_trans := Trans.cf_git))
        runs_mgr.log_param("val_trans", repr(val_trans := Trans.cf_gen))

        train_loader = dl(ds("cifar10_train", train_trans).balance_label().in_range(di), batch_size, shuffle=True)
        val_loader = dl(ds("cifar10_val", val_trans), batch_size, shuffle=True)

        runs_mgr.log_param("num_data", len(train_loader.dataset))
        runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_loader))
        runs_mgr.log_param("dataset", train_loader.dataset.ds_str)
        runs_mgr.log_param("model_arc", net.__module__)

        models = []
        for i, fils in enumerate(fi):
            runs_mgr[i].log_param("fils", fils)
            runs_mgr[i].log_param("ensembles", ensembles := int((base_fi / fils) ** 2))
            # runs_mgr[i].log_param("ensembles", ensembles := min(int((base_fi / fils) ** 2), 2048))
            network = net(num_classes=10, nb_fils=fils, ee_groups=ensembles)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
            scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

            model = Model(network, loss_func, optimizer, scheduler_t, device)
            models.append(model)
        mmodel = MyMultiTrain(models, device)

        runs_mgr.log_param("params", mmodel.count_params())
        hp_dict = {
            "loss_func": mmodel.repr_loss_func(),
            "optimizer": mmodel.repr_optimizer(),
            "scheduler": mmodel.repr_scheduler(),
        }
        runs_mgr.log_params(hp_dict)
        runs_mgr.log_text(mmodel.repr_network(), "model_layers.txt")
        runs_mgr.log_text(mmodel.arc_check(dl=train_loader), "model_structure.txt")

        print(f"{di=}, {fi=}, {sum(fi)=}, {batch_size=}")

        for e in range(epochs):
            runs_mgr.log_metric("lr", mmodel.get_lr(), step=e + 1)

            train_loss, train_acc = mmodel.train_1epoch(train_loader)
            val_loss, val_acc = mmodel.val_1epoch(val_loader)

            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}

            runs_mgr.log_metrics(met_dict, step=e + 1)
            mmodel.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
            runs_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

        runs_mgr.log_torch_save(mmodel.get_sd(), "state_dict.pt")

        rv = RunViewer(exc_path=__file__, exp_name=exp_name)
        rv.write_stats()
