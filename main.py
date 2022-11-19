# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 22:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
from logger import wandb

import torch
import random
import numpy as np

import train

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False
autocast_button = False

random.seed(48)
np.random.seed(48)
torch.manual_seed(48)
torch.cuda.manual_seed_all(48)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


hyper_params = {
    "mode": 'image',
    "ex_number": 'EDSR_3080Ti_Image',
    "raw_size": (3, 512, 512),
    "crop_size": (3, 256, 256),
    "input_size": (3, 256, 256),
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 200,
    "threshold": 28,
    "checkpoint": False,
    "Img_Recon": True,
    "src_path": 'E:/BJM/Motion_Image',
    "check_path": r'F:\BJM\Motion_Image\2022-08-24-14-59-27.160160\save_model\Epoch_10_eval_16.614881643454233.pt'
}

# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = wandb.wandb_init(config=hyper_params, project='', notes='', tags='', group='')


