import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net

# from resnet9 import ResNet9 as net




tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

for i in range(1):
    batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
    epochs = 120              # エポック数 (学習を何回実施するか？という変数)
    learning_rate = 0.0003   # 学習率 (重みをどの程度変更するか？)

    # network = net(3, 100).to(torch.device("cuda"))
    network = net()
    learning_rate = learning_rate
    loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1, verbose=False)

    model_name = "crop_misup_model{i}"
    pr = Prep(batch_size, val_range=(0.9, 1.00), seed=0)
    model = Model(pr, network, learning_rate, loss_func, optimizer)

    for e in range(epochs):
        model.train_1epoch(tr.crop, mixup=False)
        model.val_1epoch(tr.gen)

        model.logging()
        model.printlog(e, epochs, log_itv=5)
        
        # if (e+1) % 100 == 0: model.save_ckpt(f"model{i}_{e+1}.ckpt")
        
    model.save_ckpt(f"{model_name}.ckpt")
    model.hist_to_csv(f"{model_name}.csv")
    


