from torchvision.models import resnet50
from prep import Prep
from nn import NeuralNet
from model import Model
import post


data_dir = "aurora/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 80        # バッチサイズ (並列して学習を実施する数)  
epochs = 1200              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0002   # 学習率 (重みをどの程度変更するか？)

pr = Prep(data_path, batch_size)
network = NeuralNet()
# network = resnet50(num_classes=4)
model = Model(network, learning_rate, epochs)

dl_train = pr.fetch_train()
model.fit(dl_train)

dl_test = pr.fetch_test()
result = model.pred(dl_test, categorize=True)

# post.postprocess(dl_test, result, model)


