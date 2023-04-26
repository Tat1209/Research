from prep import Prep
from model import Model
import post
import ens

# from nn import NeuralNet as net
# from torchvision.models import resnet18 as net
# from torchvision.models import efficientnet_v2_s as net
# from torchvision.models.regnet import regnet_y_1_6gf as net
from torchvision.models.regnet import regnet_y_400mf as net

data_dir = "/root/app/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 1              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)


pr = Prep(data_path, batch_size, val_range=(0.2, 0.4))
network = net(num_classes=4)
model = Model(pr, network, epochs, learning_rate)

hist = model.fit(fit_aug_ratio=1.0, mixup_alpha=0.2)
# hist = model.fit(pr, fit_aug_ratio=0.8, mixup_alpha=0.2)
result = model.pred(categorize=True, tta_times=10, tta_aug_ratio=0.8)

post.postprocess(result, hist, model)




