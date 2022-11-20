# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 15:08
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_prepare.py
# @Software: PyCharm
import os


train_dir, val_dir, test_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/img'
for txt, data_dir in zip(['train.txt', 'val.txt', 'test.txt'], [train_dir, val_dir, test_dir]):
    with open(txt, 'w') as f:
        for i in os.listdir(data_dir)[:-1]:
            filename = i + ',' + i[:-4] + '.png' + '\n'
            f.write(filename)
        i = os.listdir(data_dir)[-1]
        f.write(i + ',' + i[:-4] + '.png')
