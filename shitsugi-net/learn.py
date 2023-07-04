from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net

batch_size = 250        # バッチサイズ (並列して学習を実施する数)  
epochs = 30              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

tf = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

pr = Prep(batch_size, val_range=(0.9, 1.00))
network = net()
# network = net(num_classes=4)
model = Model(pr, tf, network, epochs, learning_rate, log_itv=1)
# model = Model(pr, network, epochs, learning_rate, log_itv=100, fit_aug_ratio=1.0, mixup_alpha=0.2, pred_times=25, tta_aug_ratio=0.75)
# model = Model(pr, network, epochs, learning_rate, log_itv=1, fit_aug_ratio=1.0, mixup_alpha=0.2)

hist = model.fit()
# result = model.pred(categorize=False, val=True)
result = model.pred(categorize=True)

model.save_model()
model.hist_to_csv(hist)
# model.result_to_csv(result)
model.result_to_out(result)





