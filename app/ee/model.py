import torch
import numpy as np
from time import time
import datetime
import torchvision
import polars as pl
import torch.nn as nn



class Trainer:
    def __init__(self, device):
        if device is None: self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else: self.device = device
        self.hist = None
        self.start_time = None


    def log_met(self, log_dict):
        if self.hist is None: self.hist = pl.DataFrame(log_dict)
        else: self.hist = pl.concat([self.hist, pl.DataFrame(log_dict)], how="diagonal")
        
    
    def get_last_epoch(self):
        if self.hist is None: return 0
        else: return self.hist[-1]['epoch'][0]


    def get_last_met(self, to_dict=False):
        if self.hist is None: return None
        else:
            met = self.hist[-1]
            if to_dict: met = {k: v[0] for k, v in met.to_dict(as_series=False).items()}
        return met

    
    def printlog(self, log_dict, e, epochs, itv=10):
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

        if (e+1) % itv == 0 or (e+1) == epochs: print(disp_str)
        else: print(disp_str, end="\r")



class Model(Trainer):
    def __init__(self, network=None, loss_func=None, optimizer=None, scheduler=None, device=None):
        super().__init__(device=device)
        self.network = network.to(self.device)                                                        
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train_1epoch(self, dl, mixup=False, mixup_alpha=0.2, sched_iter=False):
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

            if mixup: loss_b = lmd * self.loss_func(output_b, label_b)  +  (1.0 - lmd) * self.loss_func(output_b, label2_b)
            else: loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る

            loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
            _, pred = torch.max(output_b.detach(), dim=1)
                
            if mixup: acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).item()
            else: acc += torch.sum(pred == label_b.data).item()

            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                       # 誤差逆伝播 勾配計算 呼び出すたびに計算グラフを再構築
            self.optimizer.step()                   # 重み更新
            if self.scheduler is not None  and  sched_iter: self.scheduler.step()
        if self.scheduler is not None  and  not sched_iter: self.scheduler.step()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc


    def val_1epoch(self, dl):
        if len(dl.dataset) == 0: return
        loss = 0.0
        acc = 0.0

        self.network.eval()  # モデルを評価モードにする

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += torch.sum(pred == label_b.data).item()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        return loss, acc



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



class Ens(Trainer):
    def __init__(self, models=None, device=None):
        super().__init__(device=device)
        self.models = models


    def me_train_1epoch(self, dl, mixup=False, mixup_alpha=0.2, sched_iter=False):
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

            for model in self.models: model.network.train()
            output_b = [model.network(input_b) for model in self.models]
            output_b = torch.sum(torch.stack(output_b), dim=0) / len(self.models)

            if mixup: loss_b = lmd * model.loss_func(output_b, label_b)  +  (1.0 - lmd) * model.loss_func(output_b, label2_b)
            else: loss_b = model.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る

            loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
            _, pred = torch.max(output_b.detach(), dim=1)

            if mixup: acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).item()
            else: acc += torch.sum(pred == label_b.data).item()

            for model in self.models: model.optimizer.zero_grad()              
            loss_b.backward()                                               
            for model in self.models: model.optimizer.step()               

            if model.scheduler is not None  and  sched_iter: [model.scheduler.step() for model in self.models]
        if model.scheduler is not None  and  not sched_iter: [model.scheduler.step() for model in self.models]

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc
            

    def me_val_1epoch(self, dl):
        if len(dl.dataset) == 0: return
        loss = 0.0
        acc = 0.0
        
        for model in self.models: model.network.eval()  

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = [model.network(input_b) for model in self.models]

                output_b = torch.sum(torch.stack(output_b), dim=0) / len(self.models)
                loss_b = model.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += torch.sum(pred == label_b.data).item()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc
            
        
        
        