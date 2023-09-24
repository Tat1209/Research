import torch

from prep import Prep
from model import Model
from trans import Trans

from myresnet import resnet50 as net
# from torchvision.models import resnet50 as net

tr = Trans()
pr = Prep(root="/home/haselab/Documents/tat/assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

save_path = "/home/haselab/Documents/tat/app/ee/"

run_name = "tmp2"

learning_rate = 0.0001  
epochs = 10
batch_size = 500        

network = net(nb_fils=64, num_classes=100)
# network = net()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
scheduler = None
model = Model(network, loss_func, device=device, optimizer=optimizer, scheduler=scheduler)


# from torchinfo import summary
# import sys

# batch_size_tmp = 500
# flnm = "tmp.txt"
# # flnm = "my_32x2.txt"
# with open(flnm, 'w') as f:
#     original_stdout = sys.stdout
#     sys.stdout = f
#     summary(model=model.network, input_size=(batch_size_tmp, 3, 256, 256), verbose=1, col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])
#     # summary(model=model.network, input_size=(batch_size_tmp, 6, 32, 32), verbose=1, col_names=["kernel_size", "input_size", "output_size", "num_params"], row_settings=["var_names"])
#     sys.stdout = original_stdout

    

for e in range(epochs):
    Loss, Acc = model.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size), mixup=True)
    vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))

    met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
    model.log_met(met_dict)
    model.printlog(met_dict, e, epochs, log_itv=1)

# model.save_ckpt(f"{save_path}{run_name}.ckpt")
model.hist_to_csv(f"{save_path}{run_name}.csv")
