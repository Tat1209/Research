import random
import torch
import numpy as np
from scheduler import CosineAnnealingWarmupRestarts

class Model:
    def __init__(self, network, learning_rate):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。
        self.model = network.to(self.device)                                                        # ニューラルネットワークの生成して、GPUにデータを送る

        self.learning_rate = learning_rate
        self.loss_func = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定（説明省略）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)             # 最適化手法の設定（説明省略）
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=50, cycle_mult=1., min_lr=0., warmup_steps=15, gamma=0.8)


    def fit(self, pr, epochs, aug_prob=None, req_acc=None, rgb=False):
        for epoch in range(epochs):
            # augmentationの場合分け
            if aug_prob is None: 
                if epoch == 0: dl_train = pr.fetch_train(aug=False, rgb=rgb)
            else: 
                if random.random() < aug_prob: dl_train = pr.fetch_train(aug=True, rgb=rgb)
                else: dl_train = pr.fetch_train(aug=False, rgb=rgb)

            dl_val = pr.fetch_val(aug=False, rgb=rgb)

            avg_loss, avg_acc = self.train_1epoch(dl_train)

            disp_str = f"Epoch: {epoch+1: >4}/{epochs: >4}   Loss: {avg_loss:<8.6}  Acc: {avg_acc:<8.6}"

            if dl_val is not None:
                val_acc = self.val(dl_val)
                disp_str += f"  val_acc: {val_acc: <8.6}"

            if epoch % 10 == 9: print(disp_str)
            else: print(disp_str, end="\r")
            # print(disp_str)
            
            if req_acc is not None and avg_acc >= req_acc:
                print("\nAchieved the required accuracy.")
                break
            

    def train_1epoch(self, dl_train):
        self.model.train()  # モデルを訓練モードにする
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
            
        self.scheduler.step()
        
        avg_loss = total_loss / len(dl_train.dataset)
        avg_acc = total_acc / len(dl_train.dataset)
        
        return avg_loss, avg_acc
    

    def input_model(self, dl):
        self.model.eval()  # モデルを評価モードにする
        results = []
        labels = []
        with torch.no_grad(): 
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)     # GPUが使えるならGPUにデータを送る
                output_b = self.model(input_b)
                results.append(output_b)
                labels.append(label_b)
        results = torch.cat(results).cpu().numpy()
        try: labels = torch.cat(labels).numpy()
        except: labels = [label for sublist in labels for label in sublist]
        return results, labels
        

    def val(self, dl):
        results, labels = self.input_model(dl)
        results = np.argmax(results, axis=1) 
        acc = (results == labels).sum() / len(dl.dataset)
        return acc


    def pred(self, pr, categorize=True, aug=False, rgb=False):
        dl = pr.fetch_test(aug, rgb)
        results, _ = self.input_model(dl)
        if categorize: results = np.argmax(results, axis=1)  # モデルが予測した画像のクラス (aurora: 0, clearsky: 1, cloud: 2, milkyway: 3)
        return results


    def pred_tta(self, pr, times, aug_pred=None, aug_ratio=None, categorize=True, rgb=False):
        def pred_custom(i, times_i):
            if aug_pred is not None and random.random() < aug_pred  or  aug_ratio is not None and i/times_i < aug_ratio:
               return self.pred(pr, categorize=False, aug=True, rgb=rgb) 
            else: return self.pred(pr, categorize=False, aug=False, rgb=rgb)

        total_results = None
        for i in range(times):
            if i == 0: 
                total_results = pred_custom(i, times)
            else:
                total_results += pred_custom(i, times)
            pass
        results = total_results / times
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
