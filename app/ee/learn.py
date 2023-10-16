import torch
import polars as pl
import mlflow

from time import time

from prep import Prep
from model import Model, Ens
from trans import Trans

from myresnet import resnet18 as net

working_dir = "/home/haselab/Documents/tat/"

tr = Trans()
pr = Prep(root=f"{working_dir}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{working_dir}mlruns')
mlflow.set_experiment("GridSearch")

epochs = 200
learning_rate = 0.001  
batch_size = 400        

fils = 64
ensembles = 1
data_range = 1.0

iters = len(pr.dl("cifar_train", tr.cf_gen, batch_size, in_range=data_range))
    

for i in range(8):
    models = []
    for _ in range(ensembles):
        network = net(nb_fils=fils, num_classes=100)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=iters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

        model = Model(network, loss_func, optimizer=optimizer, scheduler=scheduler, device=device)
        models.append(model)
    ens = Ens(models)

    match(i):
        case 0: 
            trans = tr.cf_crop
            mixup = False
        case 1: 
            trans = tr.cf_raug(2)
            mixup = False
        case 2: 
            trans = tr.cf_raug(3)
            mixup = False
        case 3: 
            trans = tr.cf_raug(4)
            mixup = False
        case 4: 
            trans = tr.cf_crop
            mixup = True
        case 5: 
            trans = tr.cf_raug(2)
            mixup = True
        case 6: 
            trans = tr.cf_raug(3)
            mixup = True
        case 7: 
            trans = tr.cf_raug(4)
            mixup = True

    with mlflow.start_run(run_name=f"{i}") as run:

        hp_dict = {"epochs":epochs, "learning_rate":learning_rate, "batch_size":batch_size, "fils":fils, "ensembles":ensembles, 
                   "loss_func":repr(models[0].loss_func), "optimizer":repr(models[0].optimizer), "scheduler":repr(models[0].scheduler), 
                   "trans":trans, "mixup":mixup}
        mlflow.log_params(hp_dict)

        for e in range(epochs):
            # Loss, Acc = ens.me_train_1epoch(pr.dl("cifar_train", tr.cf_raug(3), batch_size, in_range=data_range), mixup=False, sched_iter=False)
            # Loss, Acc = ens.me_train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size, in_range=data_range), mixup=False, sched_iter=False)
            Loss, Acc = ens.me_train_1epoch(pr.dl("cifar_train", trans, batch_size, in_range=data_range), mixup=mixup, sched_iter=False)
            vLoss, vAcc = ens.me_val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))

            met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
            mlflow.log_metrics(met_dict, step=e)
            model.printlog(met_dict, e, epochs, itv=1) # itv = epochs

        
        
