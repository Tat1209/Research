import io
import random
import sys

import numpy as np
import torch
from torchinfo import summary


def arc_check(
    network,
    out_file=False,
    file_name="arccheck.txt",
    dl=None,
    input_size=(200, 3, 256, 256),
    verbose=1,
    col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"],
    row_settings=["var_names"],
):
    # dl もしくはinput_sizeを指定
    if dl is not None:
        input_b, _ = next(iter(dl))
        input_size = input_b.shape
    try:
        out_tmp = io.StringIO()
        sys.stdout = out_tmp
        summary(
            model=network,
            input_size=input_size,
            verbose=verbose,
            col_names=col_names,
            row_settings=row_settings,
        )
    finally:
        sys.stdout = sys.__stdout__
    summary_str = out_tmp.getvalue()

    if out_file:
        with open(file_name, "w") as f:
            f.write(summary_str)

    return summary_str


def sched_repr(scheduler) -> str:
    format_string = scheduler.__class__.__name__ + " (\n"
    for attr in dir(scheduler):
        if not attr.startswith("_") and not callable(getattr(scheduler, attr)):  # exclude special attributes and methods
            if attr.startswith("optimizer"):
                value = f"{getattr(scheduler, attr).__class__.__name__}()"
            else:
                value = getattr(scheduler, attr)
            format_string += f"{attr} = {value}\n"
    format_string += ")"
    return format_string


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def get_patial_net(network, a_tuple, b_tuple):
    model = network.children()
    num_skip = 0
    for li in a_tuple:
        for _ in range(li):
            try:
                tmp = next(model)
            except Exception:
                break
            num_skip += 1
        try:
            model = next(model).children()
        except Exception:
            break

    layers = []
    model = network.children()
    for li in b_tuple:
        for _ in range(li):
            try:
                tmp = next(model)
                if num_skip <= 0:
                    layers.append(tmp)
            except Exception:
                break
            num_skip -= 1
        try:
            model = next(model).children()
        except Exception:
            break

    return torch.nn.Sequential(*layers)


def copy_params_ee(models, ee_model):
    params_l = [model.network.state_dict() for model in models]
    ee_params = ee_model.network.state_dict()

    for name in params_l[0]:
        if name in ee_params:
            if params_l[0][name].shape == ee_params[name].shape:
                ee_params[name].copy_(params_l[0][name])
            else:
                ee_params[name] = torch.cat([param[name].clone().detach() for param in params_l], dim=0)

    ee_model.network.load_state_dict(ee_params)

    # def __repr__(self) -> str:
    #     format_string = f'{self.__class__.__name__}()\n'
    #     for attr in dir(self):
    #         if not attr.startswith("_") and not callable(getattr(self, attr)): # exclude special attributes and methods
    #             # if attr.startswith("optimizer"): value = f'{getattr(self, attr).__class__.__name__}()'
    #             # else: value = getattr(self, attr)
    #             value = getattr(self, attr)
    #             format_string += f"{attr} = {value}\n"
    #     format_string += ')'
    #     return format_string

    # def __repr__(self) -> str:
    #     format_string = f'{self.__class__.__name__}(\n'
    #     for attr, value in self.kwargs_init.items():
    #             format_string += f"{attr} = {value}\n"
    #     format_string += ')'
    #     return format_string
