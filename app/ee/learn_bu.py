import torch
import mlflow

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net



tr = Trans()
pr = Prep(root='/home/haselab/Documents/tat/app/shitsugi-net/data')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

save_path = "/home/haselab/Documents/tat/app/ee/"

mlflow.set_tracking_uri(f'{save_path}/mlruns')
mlflow.set_experiment("EasyEnsemble")

run_name = "tmp2"

learning_rate = 0.0001  
epochs = 4
batch_size = 500        

network = net(pretrained=True)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
model = Model(network, loss_func, device=device, optimizer=optimizer, scheduler=scheduler)

with mlflow.start_run(run_name=run_name) as run:

    hp_dict = {'epochs':epochs, 'batch_size':batch_size, 'learning_rate':learning_rate}
    mlflow.log_params(hp_dict)

    for e in range(epochs):
        Loss, Acc = model.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size), mixup=True)
        vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))

        met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
        model.printlog(met_dict, e, epochs, log_itv=1)
        mlflow.log_metrics(met_dict, step=e)
            

    # モデルを記録
    model.save_ckpt(f"{save_path}{run_name}.ckpt")
    model.hist_to_csv(f"{save_path}{run_name}.csv")
    mlflow.pytorch.log_model(model.network, "test_model")
    mlflow.log_artifact(f"{save_path}{run_name}.ckpt")

