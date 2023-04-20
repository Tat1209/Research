from prep import Prep
from model import Model
import post

# from nn import NeuralNet
from torchvision.models import resnet18 as net
# from torchvision.models import efficientnet_v2_m as net


data_dir = "aurora/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 1000              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

pr = Prep(data_path, batch_size, train_ratio=0.92, color=True)
# network = NeuralNet()
network = net(num_classes=4)
model = Model(network, learning_rate)
hist = model.fit(pr, epochs, aug_prob=0.98)

# result = model.pred(pr, categorize=True)
result = model.pred_tta(pr, times=2, aug_ratio=0.8, categorize=True)

post.postprocess(pr, result, hist, model)


