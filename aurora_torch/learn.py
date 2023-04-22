from prep import Prep
from model import Model
import post

# from nn import NeuralNet as net
from torchvision.models import resnet50 as net
# from torchvision.models import efficientnet_v2_s as net
# from torchvision.models.regnet import regnet_y_1_6gf as net

data_dir = "/root/app/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 50              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

network = net(num_classes=4)

model = Model(network, epochs, learning_rate)

# hist = model.fit(pr, aug_ratio=0.5, mixup_alpha=0.2)
pr = Prep(data_path, batch_size, train_range=(0.2, 0.7))
hist = model.fit(pr)

result = model.pred(pr, categorize=True, times=5, aug_ratio=0.6)

post.postprocess(pr, result, hist, model)


