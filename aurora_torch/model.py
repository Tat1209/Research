import random
import torch
import numpy as np
from time import time
import datetime


class Model:
    def __init__(self, network, epochs, learning_rate):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。
        self.model = network.to(self.device)                                                        # ニューラルネットワークの生成して、GPUにデータを送る

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.loss_func = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定（説明省略）
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate)             # 最適化手法の設定（説明省略）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)


    def fit(self, pr, aug_prob=None, early=None):
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.)

        hist = dict()
        hist["Epoch"] = [i+1 for i in range(self.epochs)]
        start = time()

        for epoch in range(self.epochs):
            if epoch != 0:
                stop = time()
                req_time = (stop-start) / epoch * self.epochs
                left = start + req_time - stop
                eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
                t_hour, t_min = divmod(left//60, 60)
                left = f"{int(t_hour):02d}:{int(t_min):02d}"


            # augmentationの場合分け
            if aug_prob is None: 
                if epoch == 0: dl_train = pr.fetch_train(pr.tr.aug)
            else: 
                if random.random() < aug_prob: dl_train = pr.fetch_train(pr.tr.aug)
                else: dl_train = pr.fetch_train(pr.tr.gen)

            dl_val = pr.fetch_val(pr.tr.aug)

            stats = dict()
            stats["Loss"], stats["Acc"] = self.train_1epoch(dl_train)
            if dl_val is not None: stats["vLoss"], stats["vAcc"] = self.val_1epoch(dl_val)
            
            for key, value in stats.items():
                if epoch == 0: hist[key] = [value]
                else: hist[key].append(value)

            disp_str = f"Epoch: {epoch+1:>4}/{self.epochs:>4}"
            for key, value in stats.items(): disp_str += f"    {key}: {value:<9.7f}"
            if epoch != 0: disp_str += f"    eta: {eta} (left: {left})"

            n = 20
            if epoch % n == (n-1) or epoch == self.epochs: print(disp_str)
            else: print(disp_str, end="\r")
            
            # if early is not None and 
            #     print("\nAchieved the required accuracy.")
            #     break
            
        return hist
            

    def train_1epoch(self, dl):
        self.model.train()  # モデルを訓練モードにする
        stats = {"result":None, "total_loss":0, "total_corr":0.}

        for input_b, label_b in dl:
            loss_b = self.flow(input_b, label_b, stats)
            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                         # 誤差逆伝播 勾配計算
            self.optimizer.step()                   # 重み更新して計算グラフを消す
        self.scheduler.step()
        
        avg_loss = stats["total_loss"] / len(dl.dataset)
        acc = stats["total_corr"] / len(dl.dataset)
        
        return avg_loss, acc
    

    def flow(self, input_b, label_b, stats):
        input_b = input_b.to(self.device)
        output_b = self.model(input_b)
        try:
            label_b = label_b.to(self.device)
            loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
        except: loss_b = None

        if stats["result"] is not None: 
            if stats["result"].size == 0: stats["result"] = output_b.detach().cpu().numpy()
            else: stats["result"] = np.concatenate((stats["result"], output_b.detach().cpu().numpy()), axis=0)
        if stats["total_loss"] is not None: stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
        if stats["total_corr"] is not None:
            _, pred = torch.max(output_b, dim=1)
            stats["total_corr"] += torch.sum(pred == label_b.data).item()
        
        return loss_b
        
    
    def val_1epoch(self, dl):
        self.model.eval()  # モデルを評価モードにする
        stats = {"result":None, "total_loss":0, "total_corr":0.}

        for input_b, label_b in dl:
            _ = self.flow(input_b, label_b, stats)

        avg_loss = stats["total_loss"] / len(dl.dataset)
        acc = stats["total_corr"] / len(dl.dataset)
        return avg_loss, acc


    def pred(self, pr, tr=None, categorize=True):
        stats = {"result":np.empty((0)), "total_loss":None, "total_corr":None}

        if tr is None: dl = pr.fetch_test(pr.tr.rgb)
        else: dl = pr.fetch_test(tr)

        for input_b, label_b in dl:
            _ = self.flow(input_b, label_b, stats)

        if categorize: stats["result"] = np.argmax(stats["result"], axis=1)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
        return stats["result"]


    def pred_tta(self, pr, times, aug_pred=None, aug_ratio=None, categorize=True):
        def pred_custom(value):
            if aug_pred is not None and random.random() < aug_pred  or  aug_ratio is not None and value < aug_ratio:
               return self.pred(pr, tr=pr.tr.rgbaug, categorize=False)
            else: return self.pred(pr, categorize=False)

        total_results = None
        for i in range(times):
            value = i/times
            if i == 0: total_results = pred_custom(value)
            else: total_results += pred_custom(value)
        result = total_results / times
        if categorize: result = np.argmax(result, axis=1)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
        return result


