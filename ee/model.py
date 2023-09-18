import torch
import numpy as np
from time import time
import datetime
import torchvision
import polars as pl
import torch.nn as nn



class Model:
    def __init__(self, network, loss_func, device=None, optimizer=None, scheduler=None):
        if device is None: self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        else: self.device = device

        self.network = network.to(self.device)                                                        

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.hist = None
        self.start_time = None


    def train_1epoch(self, dl, mixup=False, mixup_alpha=0.2):
        self.network.train()  # モデルを訓練モードにする
        loss = 0.0
        acc = 0.0
        
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
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).cpu().numpy()

            else: 
                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += torch.sum(pred == label_b.data).item()


            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                         # 誤差逆伝播 勾配計算
            self.optimizer.step()                   # 重み更新して計算グラフを消す
            
        if self.scheduler is not None: self.scheduler.step()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc


    def val_1epoch(self, dl):
        loss = 0.0
        acc = 0.0

        self.network.eval()  # モデルを評価モードにする

        outputs = None

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += torch.sum(pred == label_b.data).item()

                output_b = output_b.detach().cpu().numpy()
                label_b = label_b.detach().cpu().numpy()

                if outputs is None: outputs = output_b
                else: outputs = np.concatenate((outputs, output_b), axis=0)

        if len(dl.dataset) == 0: return
        loss /= len(dl.dataset)
        acc /= len(dl.dataset)

        return loss, acc


    def pred_1iter(self, dl, label=False):
        self.network.eval()  # モデルを評価モードにする
        
        outputs = None
        labels = None
        
        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                output_b = self.network(input_b)
                output_b = output_b.detach().cpu().numpy()

                if outputs is None: outputs = output_b
                else: outputs = np.concatenate((outputs, output_b), axis=0)

                if label:
                    if labels is None: labels = label_b
                    else: labels = np.concatenate((labels, label_b), axis=0)
                    
                    
    def logging(self, log_dict):
        if self.hist is None: self.hist = pl.DataFrame(log_dict)
        else: self.hist = pl.concat([self.hist, pl.DataFrame(log_dict)], how="diagonal")
        
    
    def get_last_epoch(self):
        if self.hist is None: return 0
        else: return self.hist[-1]['epoch'][0]

    
    def printlog(self, log_dict, e, epochs, log_itv=10):
        if e == 0: self.start_time = time()
        else:
            stop_time = time()
            req_time = (stop_time - self.start_time) / e * epochs
            left = self.start_time + req_time - stop_time
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            t_hour, t_min = divmod(left//60, 60)
            left = f"{int(t_hour):02d}:{int(t_min):02d}"
            
        disp_str = ""
        for key, value in log_dict.items():
            try: 
                if key == "epoch": disp_str += f'Epoch: {value:>4}/{value - e - 1 + epochs:>4}'
                else: disp_str += f"    {key}: {value:<9.7f}"
            except: pass
        if e != 0: disp_str += f"    eta: {eta} (left: {left})"

        if (e+1) % log_itv == 0 or (e+1) == epochs: print(disp_str)
        else: print(disp_str, end="\r")


    def save_ckpt(self, fname=None, fname_head="model_"):
        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.ckpt"
            else: fname = f"{fname_head}{date}.ckpt"

        ckpt = dict()
        ckpt["hist"] = self.hist
        ckpt["network_sd"] = self.network.state_dict()
        ckpt["optimizer_sd"] = self.optimizer.state_dict()
        try: ckpt["scheduler_sd"] = self.scheduler.state_dict()
        except: pass

        torch.save(ckpt, fname)
        
        
    def ret_ckpt(self):
        ckpt = dict()
        ckpt["hist"] = self.hist
        ckpt["network_sd"] = self.network.state_dict()
        ckpt["optimizer_sd"] = self.optimizer.state_dict()
        try: ckpt["scheduler_sd"] = self.scheduler.state_dict()
        except: pass
        
        return ckpt


    def load_ckpt(self, path):
        ckpt = torch.load(path)
        self.hist = ckpt["hist"]
        self.network.load_state_dict(ckpt["network_sd"])
        self.optimizer.load_state_dict(ckpt["optimizer_sd"])
        try: self.scheduler.state_dict(ckpt["scheduler_sd"])
        except: pass


    @classmethod
    def load_ckpt_network(cls, network, path):
        ckpt = torch.load(path)
        network.load_state_dict(ckpt["network_sd"])
        
    
    def hist_to_csv(self, fname=None, fname_head="hist_"):
        if isinstance(self.hist, pl.DataFrame): df = self.hist
        else: df = pl.DataFrame(self.hist)

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.write_csv(fname)
        
    
    def result_to_csv(self, result, fname=None, fname_head="result_"):
        df = pl.DataFrame(result["outputs"], columns=range(result["outputs"].shape[1]))
        df["labels"] = result["labels"]

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.write_csv(fname)
        

