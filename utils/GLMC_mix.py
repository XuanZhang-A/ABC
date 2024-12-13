# code from https://github.com/ynu-yangpeng/GLMC

import torch
import numpy as np


def GLMC_mixed(x_lt1, x_lt2, 
               x_bal1, x_bal2, 
               target_lt, target_bal, 
               alpha=1):
    lam = np.random.beta(alpha, alpha)
    
    # ====mixup====
    # mixup (global)
    x_global = lam * x_lt1 + (1 - lam) * x_bal1
    # label for mixup
    target_g = lam * target_lt + (1 - lam) * target_bal

    # ====cutmix====
    # cutmix(local)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x_lt2.size(), lam)
    x_local = x_lt2
    x_local[:, :, bbx1:bbx2, bby1:bby2] = x_bal2[:, :, bbx1:bbx2, bby1:bby2]
    # label for cutmix
    lam_cutmix = lam  # NOTE It's the same with global
    target_l = lam_cutmix * target_lt + (1 - lam_cutmix) * target_bal

    # ====return====
    mixed={
        "x_mixup": x_global,
        "x_cutmix": x_local,
        "target_mixup": target_g,
        "target_cutmix": target_l,
        "lam": lam
    }
    return mixed


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2