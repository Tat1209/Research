import sys

import torch
import polars as pl

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager
from trainer import Model, Ens, MultiTrain
from trans import Trans
import utils

from mymodel import MyModel

# from models.gitresnet_ee import resnet18 as net

# model_arc = "gitresnet_ee"

# from models.tmpnet import Net as net
# model_arc = "sample_net"

from torchvision.models import efficientnet_v2_s as net

model_arc = "efficientnet_v2_s"

for lr in [0.00001, 0.00003, 0.0001, 0.0003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    ds = Datasets(root=f"{work_path}assets/datasets/")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_mgr = RunManager(exc_path=__file__, exp_name="exp_tl_check")

    run_mgr.log_param("max_lr", max_lr := lr)
    run_mgr.log_param("epochs", epochs := 50)
    run_mgr.log_param("batch_size", batch_size := 125)

    run_mgr.log_param("train_trans", repr(train_trans := Trans.as_da))
    run_mgr.log_param("val_trans", repr(val_trans := Trans.as_gen))

    labeled_ds = ds("ai-step_l").shuffle()
    # labeled_ds = ds("ai-step_l").shuffle().label_n_mult({0: 1250, 1: 3, 2: 16, 4: 125})
    val_range = (0.9, 1.0)
    train_ds = labeled_ds.ex_range(val_range).transform(train_trans)
    val_ds = labeled_ds.in_range(val_range).transform(val_trans)
    unlabeled_ds = ds("ai-step_ul").shuffle()
    test_ds = unlabeled_ds.transform(val_trans)
    train_loader = dl(train_ds, batch_size, shuffle=True)
    val_loader = dl(val_ds, batch_size=2000, shuffle=True)
    test_loader = dl(test_ds, batch_size=2000, shuffle=True)

    run_mgr.log_param("num_data", len(train_loader.dataset))
    run_mgr.log_param("iters/epoch", iters_per_epoch := len(train_loader))
    run_mgr.log_param("dataset", train_loader.dataset.ds_name)
    run_mgr.log_param("model_arc", model_arc)

    # network = net(num_classes=7)
    network = net(weights="IMAGENET1K_V1")

    network.classifier[1] = torch.nn.Linear(1280, 7)
    for param in network.parameters():
        param.requires_grad = False
    for param in network.classifier[1].parameters():
        param.requires_grad = True

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
    scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

    model = MyModel(network, loss_func, optimizer, scheduler_t, device)

    run_mgr.log_param("params", model.count_params())
    hp_dict = {
        "loss_func": repr(model.loss_func),
        "optimizer": repr(model.optimizer),
        "scheduler": utils.sched_repr(model.scheduler_t[0]),
    }
    run_mgr.log_params(hp_dict)
    run_mgr.log_text(repr(model.network), "model_layers.txt")
    run_mgr.log_text(model.arc_check(dl=train_loader), "model_structure.txt")

    for e in range(epochs):
        run_mgr.log_metric("lr", model.get_lr(), step=e + 1)

        train_loss, train_acc, train_f1 = model.train_1epoch(train_loader)
        val_loss, val_acc, val_f1 = model.val_1epoch(val_loader)

        met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}

        run_mgr.log_metrics(met_dict, step=e + 1)
        model.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
        run_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

    # run_mgr.log_torch_save(model.get_sd(), "state_dict.pt")

    outputs, labels = model.pred_1iter(test_loader)

    df_out = pl.DataFrame({"fname": labels, "pred": outputs.tolist()})
    df_out.write_csv("/home/haselab/Documents/tat/Research/app/ai-step2/submit.csv", has_header=False)
