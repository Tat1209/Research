import sys
import os
import io
from time import time

import torch
import numpy as np
import datetime
import polars as pl
from torchinfo import summary


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
        if e == 1: self.start_time = time()
        else:
            stop_time = time()
            req_time = (stop_time - self.start_time) / (e-1) * epochs
            left = self.start_time + req_time - stop_time
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            t_hour, t_min = divmod(left//60, 60)
            left = f"{int(t_hour):02d}:{int(t_min):02d}"
            
        disp_str = ""
        for key, value in log_dict.items():
            try: 
                if key == "epoch": disp_str += f'Epoch: {value:>4}/{value - (e-1) - 1 + epochs:>4}'
                else: disp_str += f"    {key}: {value:<9.7f}"
            except: pass
        if (e-1) != 0: disp_str += f"    eta: {eta} (left: {left})"

        if (e-1) % itv >= itv-1 or e == epochs: print(disp_str)
        else: print(disp_str, end="\r")
        
        
    def mlflow_save(self, mlflow_obj, artifact, path):
        if os.path.exists(path): raise FileExistsError(f"Duplicate filenames '{path}' for temporary files. Please specify a different filename.")
        with open(path, "wb") as f:
            torch.save(artifact, f)
            mlflow_obj.log_artifact(f.name)
            os.remove(path)


    # def mlflow_load(self, mlflow_obj, artifact_uri, file_name):
    #     # example : "runs:/500cf58bee2b40a4a82861cc31a617b1/my_model.pkl"
    #     if os.path.exists(file_name): raise FileExistsError(f"Duplicate filenames '{file_name}' for temporary files. Please specify a different filename.")
    #     with open(file_name, "wb") as f:
    #         mlflow_obj.log_artifact(f.name)
    #         os.remove(file_name)



class Model(Trainer):
    def __init__(self, network=None, loss_func=None, optimizer=None, scheduler_t=None, device=None):
        super().__init__(device=device)
        self.network = network.to(self.device)                                                        
        self.loss_func = loss_func
        self.optimizer = optimizer
        if scheduler_t is not None:
            assert isinstance(scheduler_t, tuple), "scheduler_t must be a tuple"
            assert len(scheduler_t) == 2, "scheduler_t must have two elements"
            assert isinstance(scheduler_t[0], torch.optim.lr_scheduler.LRScheduler), "scheduler_t[0] must be a torch.optim.lr_scheduler type"
            assert isinstance(scheduler_t[1], str), "scheduler_t[1] must be a string"
            assert scheduler_t[1] in ["epoch", "batch"], "scheduler_t[1] must be either 'epoch' or 'batch'"
        self.scheduler_t = scheduler_t

        self.batch_iters = 0
        

    @torch.compile
    def train_1epoch(self, dl, mixup=False, mixup_alpha=0.2, mlflow_obj=None, log_batch_lr=False):
        self.network.train()  # モデルを訓練モードにする
        loss = 0.0
        acc = 0.0
        
        for input_b, label_b in dl:
            input_b = input_b.to(self.device)
            label_b = label_b.to(self.device)

            if log_batch_lr:
                self.batch_iters += 1
                mlflow_obj.log_metric("lr_iter", self.get_lr(), step=self.batch_iters)

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

            if self.scheduler_t is not None  and  self.scheduler_t[1] == "batch": self.scheduler_t[0].step()
        if self.scheduler_t is not None  and  self.scheduler_t[1] == "epoch": self.scheduler_t[0].step()

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
    def load_sd(self, sd): self.network.load_state_dict(sd)

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
    def __init__(self, models=None, device=None):
        super().__init__(device=device)
        self.models = models


    # @torch.compile
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

            output_b = torch.mean(torch.stack(output_b), dim=0)

            if mixup: loss_b = lmd * model.loss_func(output_b, label_b)  +  (1.0 - lmd) * model.loss_func(output_b, label2_b)
            else: loss_b = model.loss_func(output_b, label_b)

            loss += loss_b.item() * len(input_b)
            _, pred = torch.max(output_b.detach(), dim=1)

            if mixup: acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).item()
            else: acc += torch.sum(pred == label_b.data).item()

            for model in self.models: model.optimizer.zero_grad()              
            loss_b.backward()                                               
            
            for model in self.models: model.optimizer.step()               


            for model in self.models:
                if model.scheduler_t is not None  and  model.scheduler_t[1] == "batch": model.scheduler_t[0].step()
        for model in self.models:
                if model.scheduler_t is not None  and  model.scheduler_t[1] == "epoch": model.scheduler_t[0].step()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc
            

    # @torch.compile
    def pe_train_1epoch(self, dl, mixup=False, mixup_alpha=0.2):
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

            if mixup: loss_b = [lmd * self.models[m].loss_func(output_b[m], label_b)  +  (1.0 - lmd) * self.models.loss_func(output_b[m], label2_b) for m in range(len(self.models))]
            else: loss_b = [self.models[m].loss_func(output_b[m], label_b) for m in range(len(self.models))]


            output_b = torch.mean(torch.stack(output_b), dim=0)

            loss += sum([loss_b.item() for loss_b in loss_b]) * len(input_b)
            _, pred = torch.max(output_b.detach(), dim=1)

            if mixup: acc += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).item()
            else: acc += torch.sum(pred == label_b.data).item()

            for model in self.models: model.optimizer.zero_grad()              
            for loss_b in loss_b: loss_b.backward()                                               
            for model in self.models: model.optimizer.step()               


            for model in self.models:
                if model.scheduler_t is not None  and  model.scheduler_t[1] == "batch": model.scheduler_t[0].step()
        for model in self.models:
                if model.scheduler_t is not None  and  model.scheduler_t[1] == "epoch": model.scheduler_t[0].step()

        loss /= len(dl.dataset) * len(self.models)
        acc /= len(dl.dataset)
        
        return loss, acc
            

    # @torch.compile
    def val_1epoch(self, dl):
        if len(dl.dataset) == 0: return
        loss = 0.0
        acc = 0.0
        
        for model in self.models: model.network.eval()  

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = [model.network(input_b) for model in self.models]
                output_b = torch.mean(torch.stack(output_b), dim=0)
                
                loss_b = model.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                acc += torch.sum(pred == label_b.data).item()

        loss /= len(dl.dataset)
        acc /= len(dl.dataset)
        
        return loss, acc


    def get_sds(self): return [model.get_sd() for model in self.models]
    def load_sds(self, sd_list): (self.models[i].load_sd(sd) for i, sd in enumerate(sd_list))
        # for i, sd in enumerate(sd_list): self.models[i].load_sd(sd)
    
    def count_params(self):
        return sum(model.count_params() for model in self.models)

    def count_trinable_params(self):
        return sum(model.count_trainable_params() for model in self.models)


    def save_sds(self, path='state_dict_list.pkl'):
        sd_list = self.get_sds()
        torch.save(sd_list, path)


    def load_sds_path(self, path):
        sd_list = torch.load(path)
        self.load_sds(sd_list)


    def mlflow_save_sd(self, mlflow_obj, path='state_dict_list.pkl'):
        sd_list = self.get_sds()
        self.mlflow_save(mlflow_obj, sd_list, path)


    # def mlflow_load_state_dict(self, mlflow_obj, path='state_dict_list.pkl'):
    #     sd_list = self.get_sds()
    #     self.mlflow_save(self, mlflow_obj, sd_list, path)

