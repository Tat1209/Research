import torch
import mlflow.pytorch as flow

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net


tr = Trans()
pr = Prep()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

save_path = "/home/haselab/Documents/tat/app/ee/"
model_name = "tmp"

learning_rate = 0.0001  
epochs = 10
batch_size = 500        

network = net(pretrained=True)
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
model = Model(network, loss_func, device=device, optimizer=optimizer, scheduler=scheduler)

for e in range(epochs):
    Loss, Acc = model.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size), mixup=True)
    vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))

    log_dict = {"epoch":model.get_last_epoch()+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}

    model.logging(log_dict)
    model.printlog(log_dict, e, epochs, log_itv=1)

model.save_ckpt(f"{save_path}{model_name}.ckpt")
model.hist_to_csv(f"{save_path}{model_name}.csv")


