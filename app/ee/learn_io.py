import torch
import mlflow

from prep import Prep
from model import Model, Ens
from trans import Trans

import utils

# from myresnet import resnet18 as net
# from torchvision.models import resnet18 as net
# model_arc = "official"
from models.gitresnet import resnet18 as net
model_arc = "gitresnet"

work_path = '/home/haselab/Documents/tat/Research/'

tr = Trans()
pr = Prep(root=f"{work_path}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("tmp")

for fi in [2 ** i for i in range(6, 4, -1)]:
# for fi in [2 ** i for i in range(6, 0, -1)]:
    for ei in range(2):

        with mlflow.start_run(run_name=f"{fi}") as run:
            mlflow.log_metric("max_lr",     max_lr := 0.1)
            mlflow.log_metric("epochs",     epochs := 10)
            mlflow.log_metric("batch_size", batch_size := 125)
            mlflow.log_metric("fils",       fils := 64)
            mlflow.log_metric("ensembles",  ensembles := (64 // fi) ** 2)
            print(f'{fi}fils, {ensembles}ensembles')

            mlflow.log_param("data_range",  data_range := 1.0)
            mlflow.log_param("mixup",       mixup := False)
            mlflow.log_param("train_trans", repr(trans := tr.cf_git))

            train_loader = pr.dl("cifar_train", trans, batch_size, in_range=data_range)
            val_loader = pr.dl("cifar_val", tr.cf_gen, batch_size)

            mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(train_loader))
            mlflow.log_param("model_arc", model_arc)

            models = []
            for _ in range(ensembles):
                network = net(num_classes=100, nb_fils=fi)
                loss_func = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
                sched_tuple = (scheduler, "epoch")

                model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                models.append(model)
            ens = Ens(models)

            # ens.load_state_dict('tmp.pkl')

            hp_dict = {"loss_func":repr(ens.models[0].loss_func), "optimizer":repr(ens.models[0].optimizer), "scheduler":utils.sched_repr(ens.models[0].sched_tuple[0])}
            mlflow.log_params(hp_dict)
            mlflow.log_metric("params", ens.count_params())
            mlflow.log_text(ens.models[0].arc_check(dl=train_loader), "model_structure.txt")



            for e in range(epochs):
                mlflow.log_metric("lr", ens.models[0].get_lr(), step=e+1)

                if ei == 0: Loss, Acc = ens.me_train_1epoch(train_loader, mixup=mixup)
                if ei == 1: Loss, Acc = ens.pe_train_1epoch(train_loader, mixup=mixup)
                vLoss, vAcc = ens.val_1epoch(val_loader)

                met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
                mlflow.log_metrics(met_dict, step=e+1)
                model.printlog(met_dict, e, epochs, itv=epochs/10)


            mlflow.log_text(text=ens.models[0].arc_check(dl=train_loader), artifact_file="model_structure.txt")
            # ens.save_state_dict('tmp.pkl')

            ens.mlflow_save_state_dict(mlflow)
                    