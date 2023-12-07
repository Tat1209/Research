import torch
import mlflow

from datasets import Datasets, dl
from trainer import Model, Ens, EEModel
from trans import Trans
import utils

# from models.resnet_ee import resnet18 as net
# model_arc = "official_ee"
# from torchvision.models import resnet18 as net
# model_arc = "official"
from models.gitresnet_ee import resnet18 as net
model_arc = "gitresnet_ee"

work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("tmp")
# mlflow.set_experiment("fil_ens")


for fi in [32]:
    with mlflow.start_run(run_name=f"{fi}") as run:
        mlflow.log_metric("max_lr",     max_lr := 0.1)
        mlflow.log_metric("epochs",     epochs := 10)
        mlflow.log_metric("batch_size", batch_size := 125)
        mlflow.log_metric("fils",       fils := fi)
        mlflow.log_metric("ensembles",  ensembles := min([(64 // fi) ** 2, 2048]))

        mlflow.log_param("ensemble_type",   ensemble_type := ['easy', 'merge', 'pure'][0])
        mlflow.log_param("mixup",       mixup := False)
        mlflow.log_param("train_trans", repr(train_trans := Trans.cf_git))
        mlflow.log_param("val_trans",   repr(val_trans := Trans.cf_gen))

        train_loader = dl(ds("cifar100_train", train_trans, in_range=1.0), batch_size, shuffle=True)
        val_loader = dl(ds("cifar100_val", val_trans), batch_size, shuffle=True)

        mlflow.log_metric("num_data", len(train_loader.dataset))
        mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(train_loader))
        mlflow.log_param("dataset", train_loader.dataset.ds_name)
        mlflow.log_param("model_arc", model_arc)

        network = net(num_classes=100, nb_fils=fils, ee_groups=ensembles)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
        scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

        model = EEModel(network, loss_func, optimizer=optimizer, scheduler_t=scheduler_t, device=device)

        # ens.load_sds('tmp.pkl')

        mlflow.log_metric("params", model.count_params())
        hp_dict = {"loss_func":repr(model.loss_func), "optimizer":repr(model.optimizer), "scheduler":utils.sched_repr(model.scheduler_t[0])}
        mlflow.log_params(hp_dict)
        mlflow.log_text(repr(model.network), "model_layers.txt")
        mlflow.log_text(model.arc_check(dl=train_loader), "model_structure.txt")

        print(f"{fi}fils, {ensembles}ensembles, {ensemble_type}")

        for e in range(epochs):
            mlflow.log_metric("lr", model.get_lr(), step=e+1)

            tLoss, tAcc = model.train_1epoch(train_loader, mixup=mixup)
            vLoss, vAcc ,lLoss, lAcc= model.val_1epoch(val_loader,log_layer=model.network.fc[2])
            # vLoss, vAcc = model.val_1epoch(val_loader)

            met_dict = {"epoch":e+1, "tLoss":tLoss, "tAcc":tAcc, "vLoss":vLoss, "vAcc":vAcc, "lLoss":lLoss, "lAcc":lAcc}
            # met_dict = {"epoch":e+1, "tLoss":tLoss, "tAcc":tAcc, "vLoss":vLoss, "vAcc":vAcc}

            mlflow.log_metrics(met_dict, step=e+1)
            model.printlog(met_dict, e+1, epochs, itv=epochs/4)

        # model.save_sds('tmp.pkl')
        # model.mlflow_save_sds(mlflow)

