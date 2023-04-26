import numpy as np

from model import Model
import post


class Ens:
    def __init__(self):
        pass

    def cross(model_list, categorize=True, fit_aug_ratio=None, tta_times=None, tta_aug_ratio=None, mixup_alpha=None):
        mod_res = {"vAcc":[], "outputs":[]}
        for model in model_list:
            hist = model.fit(fit_aug_ratio=fit_aug_ratio, mixup_alpha=mixup_alpha)

            result = model.pred(categorize=False, tta_times=tta_times, tta_aug_ratio=tta_aug_ratio)
            
            mod_res["vAcc"].append(hist["vAcc"][-1])
            mod_res["results"].append(result)
            
            post.postprocess(result, hist, model)
        vAcc_sum = (np.array(mod_res["results"]) * np.array(mod_res["vAcc"])[:, np.newaxis, np.newaxis]).sum(axis=0)
        ens_res =  vAcc_sum / np.array(mod_res["vAcc"]).sum()
        if categorize: ens_res = np.argmax(ens_res, axis=1)
        post.postprocess(ens_res, None, model)
        