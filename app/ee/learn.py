import torch
import mlflow

from prep import Prep
from model import Model, Ens
from trans import Trans

import utils

# from myresnet import resnet18 as net
# from torchvision.models import resnet18 as net
# model_arc = "official"
from gitresnet import resnet18 as net
model_arc = "gitresnet"

work_path = '/home/haselab/Documents/tat/Research/'

tr = Trans()
pr = Prep(root=f"{work_path}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
# mlflow.set_experiment("scheduler_lr_gridsearch_v2")
mlflow.set_experiment("tmp")

for k in range(2):
    for j in range(-20, 0, 2):
        for i in range(6):

            with mlflow.start_run(run_name=f"{i}") as run:
                mlflow.log_metric("max_lr",     max_lr := 10**(j / 10))
                mlflow.log_metric("epochs",     epochs := 200)
                if k == 0: mlflow.log_metric("batch_size", batch_size := 125)
                elif k == 1: mlflow.log_metric("batch_size", batch_size := 250)
                mlflow.log_metric("fils",       fils := 64)
                mlflow.log_metric("ensembles",  ensembles := 1)
                mlflow.log_param("data_range",  data_range := 1.0)
                mlflow.log_param("mixup",       mixup := False)
                mlflow.log_param("trans",       repr(trans := tr.cf_git))

                train_loader = pr.dl("cifar_train", trans, batch_size, in_range=data_range)
                val_loader = pr.dl("cifar_val", tr.cf_gen, batch_size)

                mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(pr.dl("cifar_train", tr.cf_gen, batch_size, in_range=data_range)))
                mlflow.log_param("model_arc", model_arc)


                match i:
                    case 0:
                        mlflow.log_param("train_trans", repr(trans := tr.cf_git))
                        mlflow.log_param("mixup",   mixup := False)
                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=iters_per_epoch)
                            sched_tuple = (scheduler, "batch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)

                    case 1: 
                        mlflow.log_param("train_trans", repr(trans := tr.cf_git))
                        mlflow.log_param("mixup",   mixup := False)

                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
                            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs/4), T_mult=1, eta_min=0, last_epoch=-1)
                            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
                            sched_tuple = (scheduler, "epoch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)


                    case 2: 
                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
                            sched_tuple = (scheduler, "epoch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)

                    case 3:
                        mlflow.log_param("warmup_split",    warmup_split := 2)

                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs/warmup_split), T_mult=1, eta_min=0, last_epoch=-1)
                            sched_tuple = (scheduler, "epoch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)
                        
                        
                        
                    case 4:
                        mlflow.log_param("warmup_split",    warmup_split := 4)

                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs/warmup_split), T_mult=1, eta_min=0, last_epoch=-1)
                            sched_tuple = (scheduler, "epoch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)
                        
                        
                        
                    case 5:
                        mlflow.log_param("warmup_split",    warmup_split := "T_0=5, T_mult=3")

                        models = []
                        for _ in range(ensembles):
                            network = net(num_classes=100)
                            loss_func = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3, eta_min=0, last_epoch=-1)
                            sched_tuple = (scheduler, "epoch")

                            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
                            models.append(model)
                        ens = Ens(models)
                        
                    case _: break


                hp_dict = {"loss_func":repr(ens.models[0].loss_func), "optimizer":repr(ens.models[0].optimizer), "scheduler":utils.sched_repr(ens.models[0].sched_tuple[0])}
                mlflow.log_params(hp_dict)
                mlflow.log_metric("params", ens.count_params())
                mlflow.log_text(ens.models[0].arc_check(dl=train_loader), "model_structure.txt")

                for e in range(epochs):
                    mlflow.log_metric("lr", ens.models[0].get_lr(), step=e+1)

                    Loss, Acc = ens.me_train_1epoch(train_loader, mixup=mixup)
                    vLoss, vAcc = ens.me_val_1epoch(val_loader)

                    met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
                    mlflow.log_metrics(met_dict, step=e+1)
                    model.printlog(met_dict, e, epochs, itv=epochs/10)

                    
                    
