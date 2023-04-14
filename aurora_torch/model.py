import torch
import numpy as np

class Model:
    def __init__(self, network, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。
        self.model = network().to(self.device)                                                        # ニューラルネットワークの生成して、GPUにデータを送る
        
        self.criterion = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定（説明省略）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)             # 最適化手法の設定（説明省略）

    def fit(self, dl_train):
        self.model.train()  # モデルを訓練モードにする

        for epoch in range(self.epochs): # 設定したエポック数分、学習する。
            loss_sum = 0
            for inputs, labels in dl_train:
                inputs = inputs.to(self.device)     # GPUが使えるならGPUにデータを送る
                labels = labels.to(self.device)

                self.optimizer.zero_grad()          # optimizerを初期化
                outputs = self.model(inputs)        # ニューラルネットワークの処理を実施

                loss = self.criterion(outputs, labels)  # 損失(出力とラベルとの誤差)の計算
                loss_sum += loss

                loss.backward()                     # 学習
                self.optimizer.step()

            print(f"Epoch: {epoch+1}/{self.epochs}, Loss: {loss_sum.item() / len(dl_train)}")   # 学習状況の表示
            

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

