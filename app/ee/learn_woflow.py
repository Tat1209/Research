import torch
import polars as pl

from prep import Prep
from model import Model
from trans import Trans

from myresnet import resnet18 as net

tr = Trans()
pr = Prep(root="/home/haselab/Documents/tat/assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

save_path = "/home/haselab/Documents/tat/app/ee/results/"

learning_rate = 0.0001  
batch_size = 500        
nb_fils = 4
ens = 1

fil_times = 7
times = 1 
epochs = 500

# data_range = 1.0
data_range = (0, 0.5)


for fi in range(fil_times):
    fils = nb_fils * 2 ** fi
    run_name = f"r18_{fils}f"

    try: df = pl.read_csv(f"{save_path}{run_name}_ntrain.csv")
    except: df = pl.DataFrame()

    for ni in range(times):
        print(f'{fils}filters')
        network = [net(nb_fils=fils, num_classes=100) for _ in range(ens)]
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = [torch.optim.Adam(network[i].parameters(), lr=learning_rate) for i in range(ens)]
        scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=epochs, eta_min=0, last_epoch=-1) for i in range(ens)]
        model = Model(network, loss_func, optimizer=optimizer, scheduler=scheduler, device=device)

        print(f'train {ni+1}/{times}')

        for e in range(epochs):
            Loss, Acc = model.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size, in_range=data_range), mixup=True)
            vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))

            met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
            model.log_met(met_dict)
            model.printlog(met_dict, e, epochs, itv=5)
            # model.printlog(met_dict, e, epochs, itv=epochs)

        # 初回だけ保存
        if ni == 0: model.hist_to_csv(f"{save_path}{run_name}_1train.csv")
        
        # metrix保存 dfには学習後metrixが試行ごとに記録される
        # if ni == 0: df = model.get_last_met()
        # else: df = pl.concat([df, model.get_last_met()])
        df = pl.concat([df, model.get_last_met()])
        
        
    df.write_csv(f"{save_path}{run_name}_ntrain.csv")
    df_stat = df.describe()
    print(df_stat)
    df_stat.write_csv(f"{save_path}{run_name}_stat.csv")


# model.save_ckpt(f"{save_path}{run_name}.ckpt")
