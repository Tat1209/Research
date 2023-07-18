import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net



def merge_sd(state_dict0, state_dict1):
    state_dict_ave = {}
    for k in state_dict0.keys():
        state_dict_ave[k] = (state_dict0[k] + state_dict1[k]) / 2
        
    return state_dict_ave

tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})



model_name = "cifar_base"

batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
epochs = 400              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.001   # 学習率 (重みをどの程度変更するか？)
weight_decay = 0.001

network = net()
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)


pr = Prep(batch_size)
model = Model(network, loss_func, optimizer)

for e in range(epochs):
    model.train_1epoch(pr.train(tr.crop), mixup=True)
    model.val_1epoch(pr.val(tr.gen))
    # model.val_1epoch(pr.val_in(tr.gen), log_loss="iLoss", log_acc="iAcc")

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
    # if (e+1) % 100 == 0: model.save_ckpt(f"model{i}_{e+1}.ckpt")
    
model.save_ckpt(f"{model_name}.ckpt")
model.hist_to_csv(f"{model_name}.csv")
    

model_name = "cifar_ft"

batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
epochs = 400              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)
weight_decay = 0.001

network = net()
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)


pr = Prep(batch_size)
model = Model(network, loss_func, optimizer)

for e in range(epochs):
    model.train_1epoch(pr.train(tr.crop), mixup=True)
    model.val_1epoch(pr.val(tr.gen))

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
    # if (e+1) % 100 == 0: model.save_ckpt(f"model{i}_{e+1}.ckpt")
    
model.save_ckpt(f"{model_name}.ckpt")
model.hist_to_csv(f"{model_name}.csv")
    


model_name = f"cifar_cifar"

batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
epochs = 400              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.001   # 学習率 (重みをどの程度変更するか？)
weight_decay = 0.001

network = net()
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1, verbose=False)

network_m = net()
network_m.load_state_dict(torch.load('cifar_base.ckpt')["network_sd"])

pr = Prep(batch_size)
model = Model(network, loss_func, optimizer)

for e in range(epochs):
    if (e) % 10 == 0: model.network.load_state_dict(merge_sd(model.network.to("cuda").state_dict(), network_m.to("cuda").state_dict()))

    model.train_1epoch(pr.train(tr.crop), mixup=True)
    model.val_1epoch(pr.val(tr.gen))

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
    # if (e+1) % 100 == 0: model.save_ckpt(f"model{i}_{e+1}.ckpt")
    
model.save_ckpt(f"{model_name}.ckpt")
model.hist_to_csv(f"{model_name}.csv")
    


model_name = f"cifar_image"


batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
epochs = 400              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.001   # 学習率 (重みをどの程度変更するか？)
weight_decay = 0.001

network = net()
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1, verbose=False)

network_m = net()
network_m.load_state_dict(torch.load('cifar_ft.ckpt')["network_sd"])
pr = Prep(batch_size)
model = Model(network, loss_func, optimizer)

for e in range(epochs):
    if (e) % 10 == 0: model.network.load_state_dict(merge_sd(model.network.to("cuda").state_dict(), network_m.to("cuda").state_dict()))

    model.train_1epoch(pr.train(tr.crop), mixup=True)
    model.val_1epoch(pr.val(tr.gen))

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
model.save_ckpt(f"{model_name}.ckpt")
model.hist_to_csv(f"{model_name}.csv")
    