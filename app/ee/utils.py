import torch

from torchinfo import summary
import sys

def arc_check(network, batch_size=128, verbose=1, channels=3, input_size=256, file_name='arccheck.txt'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    network.to(device)

    with open(file_name, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        summary(model=network, input_size=(batch_size, channels, input_size, input_size), verbose=verbose, col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])
        sys.stdout = original_stdout
        
# arc_check(network=net, file_name='arccheck.txt', verbose=1)

def sched_repr(scheduler) -> str:
    format_string = scheduler.__class__.__name__ + ' ('
    for attr in dir(scheduler):
        if not attr.startswith("_") and not callable(getattr(scheduler, attr)): # exclude special attributes and methods
            if attr.startswith("optimizer"): value = getattr(scheduler, attr).__class__.__name__ + '()'
            else: value = getattr(scheduler, attr)
            format_string += f"{attr} = {value}\n"
    format_string += ')'
    return format_string
