import numpy as np

from prep import Prep
from model import Model
import post
import ens

# from torchvision.models import efficientnet_v2_s as net1
# from torchvision.models.regnet import regnet_y_3_2gf as net2
# from torchvision.models.regnet import regnet_y_800mf as net3
from torchvision.models.regnet import regnet_y_400mf as net4

data_dir = "/root/app/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 80              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)


model_list = []
# for i in range(3): model_list.append(Model(net1(num_classes=4), epochs, learning_rate))
# for i in range(3): model_list.append(Model(net2(num_classes=4), epochs, learning_rate))
# for i in range(5): model_list.append(Model(net3(num_classes=4), epochs, learning_rate))
for i in range(2): model_list.append(Model(net4(num_classes=4), epochs, learning_rate))

pr = Prep(data_path, batch_size)
ens.ens(model_list, pr, categorize=True, fit_aug_ratio=0.90, tta_times=8, tta_aug_ratio=0.75, mixup_alpha=0.2)
        


