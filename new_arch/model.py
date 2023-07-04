import torch
import numpy as np
from time import time
import datetime
import torchvision
import pandas as pd


class Model:
    def __init__(self, pr, network, learning_rate, loss_func, optimizer):
        self.pr = pr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

        self.network = network.to(self.device)                                                        
        self.learning_rate = learning_rate

        self.loss_func = loss_func
        self.optimizer = optimizer

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.hist = dict()
        self.start = None


    def train_1epoch(self, tf, mixup=False, mixup_alpha=0.2):
        dl = self.pr.fetch_train(tf)
        self.network.train()  # モデルを訓練モードにする
        stats = {"total_loss":0., "acc":0.}

        for input_b, label_b in dl:
            input_b = input_b.to(self.device)
            label_b = label_b.to(self.device)

            if mixup:
                lmd = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(input_b.shape[0]).to(self.device)
                input2_b = input_b[perm, :]
                label2_b = label_b[perm]
                
                input_b = lmd * input_b  +  (1.0 - lmd) * input2_b
                output_b = self.network(input_b)
                loss_b = lmd * self.loss_func(output_b, label_b)  +  (1.0 - lmd) * self.loss_func(output_b, label2_b)
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["acc"] += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).cpu().numpy()

            else: 
                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["acc"] += torch.sum(pred == label_b.data).item()


            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                         # 誤差逆伝播 勾配計算
            self.optimizer.step()                   # 重み更新して計算グラフを消す
        # self.scheduler.step()

        stats["total_loss"] /= len(dl.dataset)
        stats["acc"] /= len(dl.dataset)

        return stats


    def val_1epoch(self, tf):
        dl = self.pr.fetch_val(tf)
        self.network.eval()  # モデルを評価モードにする
        stats = {"total_loss":0., "acc":0., "outputs":None}

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["acc"] += torch.sum(pred == label_b.data).item()

                output_b = output_b.detach().cpu().numpy()
                label_b = label_b.detach().cpu().numpy()

                if stats["outputs"] is None: stats["outputs"] = output_b
                else: stats["outputs"] = np.concatenate((stats["outputs"], output_b), axis=0)

        stats["total_loss"] /= len(dl.dataset)
        stats["acc"] /= len(dl.dataset)

        return stats


    def pred_1iter(self, tf, label=False):
        dl = self.pr.fetch_test(tf)
        stats = {"outputs":None, "labels": None}
        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                output_b = self.network(input_b)
                output_b = output_b.detach().cpu().numpy()

                if stats["outputs"] is None: stats["outputs"] = output_b
                else: stats["outputs"] = np.concatenate((stats["outputs"], output_b), axis=0)

                if label:
                    if stats["labels"] is None: stats["labels"] = label_b
                    else: stats["labels"] = np.concatenate((stats["labels"], label_b), axis=0)

        return stats
    
    def printlog(self, e, epochs, train_stats, val_stats=None, log_itv=10):
        if e == 0: 
            self.start = time()
            self.hist["Epoch"] = [i+1 for i in range(epochs)]

        if e != 0:
            stop = time()
            req_time = (stop-self.start) / e * epochs
            left = self.start + req_time - stop
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            t_hour, t_min = divmod(left//60, 60)
            left = f"{int(t_hour):02d}:{int(t_min):02d}"

        summary = dict()
        summary["Loss"], summary["Acc"] = train_stats["total_loss"], train_stats["acc"]
        if val_stats is not None: summary["vLoss"], summary["vAcc"] = val_stats["total_loss"], val_stats["acc"]

        for key, value in summary.items():
            if e == 0: self.hist[key] = [value]
            else: self.hist[key].append(value)

        disp_str = f'Epoch: {e+1:>4}/{epochs:>4}'      # 本当はsummary["Epoch"][epoch]とかがいいけどだるい
        for key, value in summary.items(): disp_str += f"    {key}: {value:<9.7f}"
        if e != 0: disp_str += f"    eta: {eta} (left: {left})"

        if (e+1) % log_itv == 0 or (e+1) == epochs: print(disp_str)
        else: print(disp_str, end="\r")


    def save_model(self, fname=None, fname_head="model_"):
        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.pth"
            else: fname = f"{fname_head}{date}.pth"
        torch.save(self.network, fname)
        
    
    def hist_to_csv(self, fname=None, fname_head="hist_"):
        if isinstance(self.hist, pd.Dataframe): df = self.hist
        else: df = pd.DataFrame(self.hist)

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.to_csv(fname, index=False)
        
    
    def result_to_csv(self, result, fname=None, fname_head="result_"):
        df = pd.DataFrame(result["outputs"], columns=range(result["outputs"].shape[1]))
        df["labels"] = result["labels"]

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.to_csv(fname, index=False)
        

