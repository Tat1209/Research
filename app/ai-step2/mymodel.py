import sys

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import seaborn

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from trainer import Model, MultiTrain

from sklearn.metrics import confusion_matrix, precision_score

# from torchvision.transforms import v2


class MyModel(Model):
    def __init__(self, network, loss_func, optimizer, scheduler_t=None, device=None):
        super().__init__(network, loss_func, optimizer, scheduler_t, device)
        # self.cutmix = v2.CutMix(num_classes=7)
        # inputs, labels = self.cutmix(inputs, labels)

    def train_1epoch(self, dl, mixup=False):
        self.network.train()
        total_loss = 0.0
        total_corr = 0.0

        pred_list = None
        true_list = None

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # lmd = np.random.beta(0.2, 0.2)
            # perm = torch.randperm(inputs.shape[0]).to(self.device)
            # inputs2 = inputs[perm, :]
            # labels2 = labels[perm]
            # inputs = lmd * inputs + (1.0 - lmd) * inputs2

            output = self.network(inputs)
            loss = self.loss_func(output, labels)
            # loss = lmd * self.loss_func(output, labels) + (1.0 - lmd) * self.loss_func(output, labels2)

            _, pred = torch.max(output.detach(), dim=1)
            # corr = (lmd * torch.sum(pred == labels) + (1.0 - lmd) * torch.sum(pred == labels2)).item()
            corr = torch.sum(pred == labels).item()

            if pred_list is None:
                pred_list = pred
            else:
                pred_list = torch.cat([pred_list, pred])

            if true_list is None:
                true_list = labels
            else:
                true_list = torch.cat([true_list, labels])

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler_t is not None and self.scheduler_t[1] == "batch":
                self.scheduler_t[0].step()
        if self.scheduler_t is not None and self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)
        true_list = true_list.cpu()
        pred_list = pred_list.cpu()
        train_f1 = precision_score(y_true=true_list, y_pred=pred_list, average="macro", zero_division=0.0)

        return train_loss, train_acc, train_f1

    def val_1epoch(self, dl):
        if dl is None:
            return None, None, None

        self.network.eval()
        total_loss = 0.0
        total_corr = 0.0

        pred_list = None
        true_list = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.network(inputs)
                loss = self.loss_func(output, labels)

                _, pred = torch.max(output.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_loss += loss.item() * len(inputs)
                total_corr += corr

                if pred_list is None:
                    pred_list = pred
                else:
                    pred_list = torch.cat([pred_list, pred])

                if true_list is None:
                    true_list = labels
                else:
                    true_list = torch.cat([true_list, labels])

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)
        true_list = true_list.cpu()
        pred_list = pred_list.cpu()

        val_f1 = precision_score(y_true=true_list, y_pred=pred_list, average="macro", zero_division=0.0)

        return val_loss, val_acc, val_f1

    def pred_1iter(self, dl, categorize=True):
        self.network.eval()
        total_output = None
        total_label = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                output = self.network(inputs)
                output = output.detach()

                if categorize:
                    _, pred = torch.max(output, dim=1)
                    output = pred

                if total_output is None:
                    total_output = output
                else:
                    total_output = torch.cat((total_output, output), dim=0)

                if total_label is None:
                    total_label = labels
                else:
                    total_label.extend(labels)

        return total_output, total_label
