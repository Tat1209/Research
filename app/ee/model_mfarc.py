import sys
import io
from time import time

import torch
import numpy as np
import datetime
import polars as pl
from torchinfo import summary


class Trainer:
    def __init__(self, device=None, mlflow_obj=None):
        if device is None: self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else: self.device = device
        self.mlflow_obj = mlflow_obj
        
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

        if e % itv >= itv-1 or (e+1) == epochs: print(disp_str)
        else: print(disp_str, end="\r")
        
        
    def mlflow_save(self):
        
        


class Model(Trainer):
    def __init__(self, network=None, loss_func=None, optimizer=None, sched_tuple=None, device=None, mlflow_obj=None):
        super().__init__(device=device, mlflow_obj=mlflow_obj)
        self.network = network.to(self.device)                                                        
        self.loss_func = loss_func
        self.optimizer = optimizer
        if sched_tuple is not None:
            assert isinstance(sched_tuple, tuple), "sched_tuple must be a tuple"
            assert len(sched_tuple) == 2, "sched_tuple must have two elements"
            assert isinstance(sched_tuple[0], torch.optim.lr_scheduler.LRScheduler), "sched_tuple[0] must be a torch.optim.lr_scheduler type"
            assert isinstance(sched_tuple[1], str), "sched_tuple[1] must be a string"
            assert sched_tuple[1] in ["epoch", "batch"], "sched_tuple[1] must be either 'epoch' or 'batch'"
        self.sched_tuple = sched_tuple

        self.batch_iters = 0
        

    @torch.compile
    def train_1epoch(self, dl, mixup=False, mixup_alpha=0.2, log_batch_lr=False):
        self.network.train()  # モデルを訓練モードにする
        loss = 0.0
        acc = 0.0
        
        for input_b, label_b in dl:
            input_b = input_b.to(self.device)
            label_b = label_b.to(self.device)

            if log_batch_lr:
                self.batch_iters += 1
                self.mlflow_obj.log_metric("lr_iter", self.get_lr(), step=self.batch_iters)

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

            if self.sched_tuple is not None  and  self.sched_tuple[1] == "batch": self.sched_tuple[0].step()
        if self.sched_tuple is not None  and  self.sched_tuple[1] == "epoch": self.sched_tuple[0].step()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc


    @torch.compile
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


    def get_sd(self): return self.network.state_dict()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
    
    def count_params(self):
        return sum(p.numel() for p in self.network.parameters())

    def count_trinable_params(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


    def arc_check(self, out_file=False, file_name='arccheck.txt', dl=None, input_size=(200, 3, 256, 256), verbose=1, col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"], row_settings=["var_names"]):
        if dl is not None:
            input_b, _ = next(iter(dl))
            input_size = input_b.shape
        try:
            temp_out = io.StringIO()
            sys.stdout = temp_out
            summary(model=self.network, input_size=input_size, verbose=verbose, col_names=col_names, row_settings=row_settings)
        finally:
            sys.stdout = sys.__stdout__
        summary_str = temp_out.getvalue()

        if out_file:
            with open(file_name, 'w') as f: f.write(summary_str)
        
        return summary_str
    
    
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
    def __init__(self, models=None, device=None, mlflow_obj=None):
        super().__init__(device=device, mlflow_obj=mlflow_obj)
        self.models = models


    def me_train_1epoch(self, dl, mixup=False, mixup_alpha=0.2):
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
            else: loss_b = model.loss_func(output_b, label_b)

            loss += loss_b.item()*len(input_b)
            _, pred = torch.max(output_b.detach(), dim=1)

            if mixup: acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).item()
            else: acc += torch.sum(pred == label_b.data).item()

            for model in self.models: model.optimizer.zero_grad()              
            loss_b.backward()                                               
            for model in self.models: model.optimizer.step()               


            for model in self.models:
                if model.sched_tuple is not None  and  model.sched_tuple[1] == "batch": model.sched_tuple[0].step()
        for model in self.models:
                if model.sched_tuple is not None  and  model.sched_tuple[1] == "epoch": model.sched_tuple[0].step()

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
        
        
    def get_sd_list(self): return [model.get_sd() for model in self.models]

    def count_params(self):
        return sum(model.count_params() for model in self.models)

    def count_trinable_params(self):
        return sum(model.count_trainable_params() for model in self.models)