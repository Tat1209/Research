import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net


tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

for i in range(2):
    batch_size = 250        # バッチサイズ (並列して学習を実施する数)  
    epochs = 2000              # エポック数 (学習を何回実施するか？という変数)
    learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

    network = net()
    learning_rate = learning_rate
    loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
    if i == 0: optimizer = torch.optim.RAdam(network.parameters(), lr=learning_rate)    
    elif i == 1: optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)    

    pr = Prep(batch_size, val_range=(0.9, 1.00), seed=0)
    model = Model(pr, network, learning_rate, loss_func, optimizer)

    for e in range(epochs):
        model.train_1epoch(tr.aug, mixup=True)
        model.val_1epoch(tr.gen)

        model.logging()
        model.printlog(e, epochs, log_itv=5)
        
        if (e+1) % 100 == 0: model.save_ckpt(f"model{i}_{e+1}.ckpt")
        
    model.save_ckpt(f"model{i}.ckpt")
    model.hist_to_csv(f"{i}.csv")
    


