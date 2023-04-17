import torch
import numpy as np

class Model:
    def __init__(self, network, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。
        self.model = network.to(self.device)                                                        # ニューラルネットワークの生成して、GPUにデータを送る
        self.loss_func = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定（説明省略）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)             # 最適化手法の設定（説明省略）


    def fit(self, dl_train):
        self.model.train()  # モデルを訓練モードにする
        for epoch in range(self.epochs):
            total_loss = 0.
            total_acc = 0.
            for inputs, labels in dl_train:
                # dl_trainには、バッチが複数格納されている。forごとには1バッチ分のデータが渡される。複数形sは、バッチ内にデータがbatch_sizeぶん格納されているからついてるっぽい
                inputs = inputs.to(self.device)     # GPUが使えるならGPUにデータを送る
                labels = labels.to(self.device)

                # 学習処理
                outputs = self.model(inputs)            # ニューラルネットワークの処理を実施
                loss = self.loss_func(outputs, labels)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
                loss.backward()                         # 誤差逆伝播 勾配計算
                self.optimizer.step()                   # 重み更新して計算グラフを消す

                total_loss += loss.item() * len(inputs) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(outputs, dim=1)
                total_acc += torch.sum(pred == labels.data) 

            avg_loss = total_loss / len(dl_train.dataset)
            avg_acc = total_acc / len(dl_train.dataset)

            disp_str = f"Epoch: {epoch+1: >4}/{self.epochs: >4}    Loss: {avg_loss: <11.6} Acc: {avg_acc: <8.6}"

            if epoch % 10 == 9: print(disp_str)
            else: print(disp_str, end="\r")
            # print(disp_str)



    def pred(self, dl_test, categorize=False):
        self.model.eval()  # モデルを評価モードにする
        results = []
        with torch.no_grad(): 
            for inputs, img_path in dl_test:
                inputs = inputs.to(self.device)     # GPUが使えるならGPUにデータを送る
                result = self.model(inputs)
                results.append(result)
        results = torch.cat(results).cpu().numpy()
        if categorize: results = np.argmax(results, axis=1)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
        return results


# from torchvision import transforms
# from torchvision.transforms import InterpolationMode
# da = transforms.Compose([
#     transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.NEAREST),
#     transforms.RandomHorizontalFlip(p=0.5), 
#     transforms.ToTensor(), 
#     ])
# inputs = da(inputs)
