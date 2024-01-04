class HookManager:
    def __init__(self, layer):
        self.layer = layer

        self.module = None
        self.input = None
        self.output = None

    def hook(self, module, input, output):
        self.output = output.detach().clone()

    def register(self):
        self.handle = self.layer.register_forward_hook(self.hook)

    def remove(self):
        self.handle.remove()

    # def val_1epoch(self, dl, log_layer=None):
    #     if len(dl.dataset) == 0: return
    #     loss = 0.0
    #     acc = 0.0

    #     if log_layer is not None:
    #         features = None
    #         def hook(module, input, output):
    #             nonlocal features
    #             features = output.detach().clone()

    #         handle = log_layer.register_forward_hook(hook)

    #     self.network.eval()  # モデルを評価モードにする

    #     with torch.no_grad():
    #         for input_b, label_b in dl:
    #             input_b = input_b.to(self.device)
    #             label_b = label_b.to(self.device)

    #             output_b = self.network(input_b)
    #             loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
    #             loss += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
    #             _, pred = torch.max(output_b.detach(), dim=1)
    #             acc += torch.sum(pred == label_b.data).item()

    #             if log_layer is not None: print(features.shape)

    #     if log_layer is not None: handle.remove()

    #     loss /= len(dl.dataset)
    #     acc /= len(dl.dataset)

    #     return loss, acc
