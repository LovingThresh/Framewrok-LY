# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 10:40
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_proportion.py
# @Software: PyCharm

import os

import cv2
import numpy as np


def segmentation_proportion(file_path):
    s = []
    for num, i in enumerate(os.listdir(file_path)):
        img = cv2.imread(os.path.join(file_path, i), cv2.IMREAD_GRAYSCALE)
        img = img / 255
        s.append(img.mean())
    return s


file_path = r'L:\crack_segmentation_in_UAV_images\UAV_image\labels'
arr = segmentation_proportion(file_path)



