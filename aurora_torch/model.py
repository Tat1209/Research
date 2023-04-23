import torch
import numpy as np
from time import time
import datetime
import torchvision


class Model:
    def __init__(self, network, epochs, learning_rate):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.model = network.to(self.device)                                                        

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.loss_func = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate)             # 最適化手法の設定
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.)


    def fit(self, pr, fit_aug_ratio=None, mixup_alpha=None):

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
            if fit_aug_ratio is None: 
                if epoch == 0:
                    dl_train = pr.fetch_train(pr.tr.gen)
                    alpha = None
            else: 
                if epoch/self.epochs < fit_aug_ratio:
                    dl_train = pr.fetch_train(pr.tr.aug)
                    alpha = mixup_alpha
                else:
                    dl_train = pr.fetch_train(pr.tr.gen)
                    alpha=None

            dl_val = pr.fetch_val(pr.tr.gen)

            stats = dict()
            stats["Loss"], stats["Acc"] = self.train_1epoch(dl_train, mixup_alpha=alpha)
            if dl_val is not None: stats["vLoss"], stats["vAcc"] = self.val_1epoch(dl_val)

            for key, value in stats.items():
                if epoch == 0: hist[key] = [value]
                else: hist[key].append(value)

            disp_str = f"Epoch: {epoch+1:>4}/{self.epochs:>4}"
            for key, value in stats.items(): disp_str += f"    {key}: {value:<9.7f}"
            if epoch != 0: disp_str += f"    eta: {eta} (left: {left})"

            n = 20
            if (epoch+1) % n == 0 or (epoch+1) == self.epochs: print(disp_str)
            else: print(disp_str, end="\r")

        return hist
    

    def train_1epoch(self, dl, mixup_alpha=None):
        self.model.train()  # モデルを訓練モードにする
        stats = {"result":None, "total_loss":0, "total_corr":0.}

        for input_b, label_b in dl:
            loss_b = self.flow(input_b, label_b, stats, mixup_alpha=mixup_alpha)
            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                         # 誤差逆伝播 勾配計算
            self.optimizer.step()                   # 重み更新して計算グラフを消す
        self.scheduler.step()

        avg_loss = stats["total_loss"] / len(dl.dataset)
        acc = stats["total_corr"] / len(dl.dataset)

        return avg_loss, acc


    def val_1epoch(self, dl):
        self.model.eval()  # モデルを評価モードにする
        stats = {"result":None, "total_loss":0, "total_corr":0.}

        with torch.no_grad():
            for input_b, label_b in dl:
                _ = self.flow(input_b, label_b, stats)

        avg_loss = stats["total_loss"] / len(dl.dataset)
        acc = stats["total_corr"] / len(dl.dataset)
        return avg_loss, acc


    def pred_1iter(self, dl):
        stats = {"result":np.empty((0)), "total_loss":None, "total_corr":None}

        with torch.no_grad():
            for input_b, label_b in dl:
                _ = self.flow(input_b, label_b, stats)

        return stats["result"]


    def pred(self, pr, categorize=True, tta_times=1, tta_aug_ratio=None):

        total_results = None

        for i in range(tta_times):
            if tta_aug_ratio is not None:
                if i/tta_times < tta_aug_ratio: dl = pr.fetch_test(pr.tr.aug)
                else: pr.fetch_test(pr.tr.flip_aug)
            else: dl = pr.fetch_test(pr.tr.gen)

            if i == 0: total_results = self.pred_1iter(dl)
            else: total_results += self.pred_1iter(dl)

        # 平均に対してsoftmaxを適用
        result = torch.nn.functional.softmax(torch.from_numpy(total_results/tta_times), dim=1).numpy()
        if categorize: result = np.argmax(result, axis=1)
        return result


    def flow(self, input_b, label_b, stats, mixup_alpha=None):
        input_b = input_b.to(self.device)
        try:
            label_b = label_b.to(self.device)

            if mixup_alpha is None:
                output_b = self.model(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る

            else: 
                lmd = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(input_b.shape[0]).to(self.device)
                input2_b = input_b[perm, :]
                label2_b = label_b[perm]
                
                input_b = lmd * input_b  +  (1.0 - lmd) * input2_b
                output_b = self.model(input_b)
                loss_b = lmd * self.loss_func(output_b, label_b)  +  (1.0 - lmd) * self.loss_func(output_b, label2_b)

################################### 画像をテスト出力
            # torchvision.transforms.ToPILImage()(input_b[0]).save('test.jpg', quality=100, subsampling=0)

        except: 
            output_b = self.model(input_b)
            loss_b = None   # returnはダメ 結果格納できん test時の挙動

            

        if stats["result"] is not None: 
            if stats["result"].size == 0: stats["result"] = output_b.detach().cpu().numpy()
            else: stats["result"] = np.concatenate((stats["result"], output_b.detach().cpu().numpy()), axis=0)
        if stats["total_loss"] is not None: stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
        if stats["total_corr"] is not None:
            _, pred = torch.max(output_b.detach(), dim=1)
            if mixup_alpha is None:
                stats["total_corr"] += torch.sum(pred == label_b.data).item()
            else: stats["total_corr"] += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).cpu().numpy()


        return loss_b



