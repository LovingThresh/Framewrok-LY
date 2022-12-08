# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 12:41
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_prepare_motion_blur.py
# @Software: PyCharm
import os
import cv2
from albumentations import MotionBlur

train_dir, val_dir, test_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/img'


function = MotionBlur(always_apply=True, p=1.0, blur_limit=(15, 21))
for data_dir in [train_dir, val_dir, test_dir]:
    for path in os.listdir(data_dir):

        img = cv2.imread(os.path.join(data_dir, path))
        blur_img = function(image=img)
        cv2.imwrite(os.path.join(data_dir[:-3] + 'blur_img', path), blur_img['image'])
