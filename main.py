# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 22:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
import math
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
from evaluation.losses import *
from data.data_loader import get_Image_Mask_Dataset

# from utils.visualize import visualize_pair

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False

random.seed(24)
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hyper_params = {
    "mode": 'segmentation',
    "ex_number": '3090_Segmentation_EarthQuake',
    "raw_size": (3, 512, 512),
    "crop_size": (3, 512, 512),
    "input_size": (3, 512, 512),
    "batch_size": 16,
    "learning_rate": 1e-3,
    "epochs": 200,
    "threshold": 0.7,
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

warm_up_epochs = 5

# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = wandb.wandb_init(config=hyper_params, project='Image_Enhancement',
                                  notes=None, tags=None, group='Segmentation')

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

# UAV_image
# MEAN = [0.382, 0.372, 0.366]
# STD = [0.134, 0.122, 0.111]

# earthquake_crack
MEAN = [0.311, 0.307, 0.307]
STD = [0.165, 0.155, 0.143]

mean = torch.tensor([MEAN[0] * 255, MEAN[1] * 255, MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
std = torch.tensor([STD[0] * 255, STD[1] * 255, STD[2] * 255]).cuda().view(1, 3, 1, 1)

train_loader, val_loader, test_loader = get_Image_Mask_Dataset(re_size=raw_size, batch_size=batch_size)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================

generator = define_G(3, 2, 64, 'resnet_9blocks', learn_residual=False, norm='instance', mode=mode)

eval_function_iou = iou
eval_function_pr = pr
eval_function_re = re
eval_function_f1 = f1
eval_function_acc = torchmetrics.Accuracy()

loss_function = {'loss_seg': Asymmetry_Binary_Loss()}

eval_function = {'eval_iou': eval_function_iou,
                 'eval_pr': eval_function_pr,
                 'eval_re': eval_function_re,
                 'eval_f1': eval_function_f1,
                 'eval_acc': eval_function_acc,
                 }

optimizer_ft = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs \
    else 0.5 * (math.cos((epoch - warm_up_epochs) / (Epochs - warm_up_epochs) * math.pi) + 1)
exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=warm_up_with_cosine_lr)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir = copy_and_upload(hyper_params, src_path)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer = SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer = SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))

# ===============================================================================
# =                                Checkpoint                                   =
# ===============================================================================

if Checkpoint:
    checkpoint = torch.load(check_path)
    generator.load_state_dict(checkpoint)
    print("Load CheckPoint!")

# ===============================================================================
# =                                    Training                                 =
# ===============================================================================

train(generator, optimizer_ft, loss_function, eval_function,
      train_loader, val_loader, Epochs, exp_lr_scheduler,
      threshold, output_dir, train_writer, val_writer, experiment, train_comet, mode=mode)

# ===============================================================================
# =                                    CopyTree                                 =
# ===============================================================================
shutil.copytree('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp),
                '{}/Summary/trainer_{}'.format(os.path.join(src_path), timestamp))
shutil.copytree('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp),
                '{}/Summary/valer_{}'.format(os.path.join(src_path), timestamp))
