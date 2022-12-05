# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 14:22
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : UNet++.py
# @Software: PyCharm
import torch.nn as nn
import segmentation_models_pytorch as smp

Unet_ResNet50_model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=2
)


