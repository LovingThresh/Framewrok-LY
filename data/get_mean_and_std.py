# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 13:52
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : get_mean_and_std.py
# @Software: PyCharm
import os
import cv2
import numpy as np

img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = r'L:\crack_segmentation_in_UAV_images\earthquake_crack\train'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0

# 请注意此时的目录层级
for item in imgs_path_list[:1]:
    for file in os.listdir(os.path.join(imgs_path, item)):
        img = cv2.imread(os.path.join(imgs_path, item, file))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))


