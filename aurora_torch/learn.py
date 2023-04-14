from prep import Prep
from nn import NeuralNet
from model import Model
import post


data_dir = "aurora/competition01_gray_128x128/"
data_path = {"train_val":data_dir+"train_val", "test":data_dir+"test"}
class_names = ["aurora", "clearsky", "cloud", "milkyway"]

batch_size = 128        # バッチサイズ (並列して学習を実施する数)  
epochs = 300              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0002   # 学習率 (重みをどの程度変更するか？)
# num_train = 1200         # 訓練データの枚数(n/1200)
# shape = (85, 85)

pr = Prep(data_path, batch_size)
network = NeuralNet
model = Model(network, learning_rate, epochs)

dl_train = pr.fetch_train_val()
dl_test = pr.fetch_test()

model.fit(dl_train)
result = model.pred(dl_test, categorize=True)

post.postprocess(dl_test, result, model)


