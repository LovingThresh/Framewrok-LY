# -*- coding: utf-8 -*-
# @Time    : 2022/11/26 15:17
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : image_test.py
# @Software: PyCharm
import shutil

from logger import wandb
from logger.copy import copy_and_upload

import random
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from train import *
from model.model import *
from evaluation.metrics import *
from data.data_loader import get_Image_Mask_Dataset
from utils.visualize import visualize_pair

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False

random.seed(48)
np.random.seed(48)
torch.manual_seed(48)
torch.cuda.manual_seed_all(48)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hyper_params = {
    "mode": 'segmentation',
    "ex_number": 'EDSR_3080Ti_Image',
    "raw_size": (3, 512, 512),
    "crop_size": (3, 256, 256),
    "input_size": (3, 256, 256),
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 2,
    "threshold": 28,
    "checkpoint": False,
    "Img_Recon": True,
    "src_path": 'E:/BJM/Motion_Image',
    "check_path": r''
}

experiment = object
lr = hyper_params['learning_rate']
mode = hyper_params['mode']
Epochs = hyper_params['epochs']
src_path = hyper_params['src_path']
batch_size = hyper_params['batch_size']
raw_size = hyper_params['raw_size'][1:]
crop_size = hyper_params['crop_size'][1:]
input_size = hyper_params['input_size'][1:]
threshold = hyper_params['threshold']
Checkpoint = hyper_params['checkpoint']
Img_Recon = hyper_params['Img_Recon']
check_path = hyper_params['check_path']

# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = wandb.wandb_init(config=hyper_params, project='Image_Enhancement',
                                  notes=None, tags=None, group='Segmentation')

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

MEAN = [0.311, 0.307, 0.307]
STD = [0.165, 0.155, 0.143]

mean = torch.tensor([MEAN[0] * 255, MEAN[1] * 255, MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
std = torch.tensor([STD[0] * 255, STD[1] * 255, STD[2] * 255]).cuda().view(1, 3, 1, 1)

train_loader, val_loader, test_loader = get_Image_Mask_Dataset(re_size=raw_size, batch_size=batch_size)
a = next(iter(train_loader))
image, label = visualize_pair(train_loader, crop_size, mean, std, mode=mode)
