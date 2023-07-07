import time

import torch

from prep import Prep
from model import Model
from trans import Trans


from torchvision.models import resnet18 as net

batch_size = 250        # バッチサイズ (並列して学習を実施する数)  
epochs = 1              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)


tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

for i in range(2):
    pr = Prep(batch_size, val_range=(0.9, 1.00), seed=0)
    network = net()
    learning_rate = learning_rate
    loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
    optimizer = torch.optim.RAdam(network.parameters(), lr=learning_rate)    

    model = Model(pr, network, learning_rate, loss_func, optimizer)

    for e in range(epochs):
        model.train_1epoch(tr.gen, mixup=True)
        model.val_1epoch(tr.gen)

        model.logging()
        model.printlog(e, epochs, log_itv=5)
        
    model_name = "model" + i
    model_weight_name = model_name + "weight.pth"
    model_hist_name = model_name + "hist.csv"
    model.save_model(model_weight_name)
    model.hist_to_csv(model_hist_name)

state_dict1 = torch.load('model1.pth')
state_dict2 = torch.load('model2.pth')


pr = Prep(batch_size, val_range=(0.9, 1.00), seed=0)
network = net()
learning_rate = learning_rate
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.RAdam(network.parameters(), lr=learning_rate)    

model = Model(pr, network, learning_rate, loss_func, optimizer)

for param1, param2, param3 in zip(state_dict1.values(), state_dict2.values(), model.parameters()):
    param3.data.copy_((param1 + param2) / 2)

for e in range(epochs):
    model.train_1epoch(tr.gen, mixup=True)
    model.val_1epoch(tr.gen)

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
model.save_model()
model.hist_to_csv()



