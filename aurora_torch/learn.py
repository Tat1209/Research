from torchvision.models import efficientnet_v2_s as Net
from prep import Prep
from nn import NeuralNet
from model import Model
import post


data_dir = "aurora/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 3000              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

# pr = Prep(data_path, batch_size, train_ratio=0.92)
pr = Prep(data_path, batch_size)
# network = NeuralNet()
network = Net(num_classes=4)




model = Model(network, learning_rate)

model.fit(pr, epochs, aug_prob=0.9, rgb=True)

# result = model.pred(pr, categorize=True, rgb=True)
result = model.pred_tta(pr, times=20, aug_ratio=0.8, categorize=True, rgb=True)

post.postprocess(pr.fetch_test(), result, model)


