import torch
import polars as pl

from model import Model

from myresnet import resnet18 as net
# from torchvision.models import resnet18 as net


# network = net(nb_fils=32, num_classes=100, ee_groups=2)

nb_fils = 16
network = net(nb_fils=nb_fils, num_classes=100)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model = Model(network, loss_func=None, device=device, optimizer=None, scheduler=None)

from torchinfo import summary
import sys
batch_size_tmp = 500

flnm = "arccheck.txt"
with open(flnm, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    summary(model=network, input_size=(batch_size_tmp, 3, 256, 256), verbose=1, col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])
    # summary(model=network, input_size=(batch_size_tmp, 9, 32, 32), verbose=1, col_names=["kernel_size", "input_size", "output_size", "num_params"], row_settings=["var_names"])
    sys.stdout = original_stdout
