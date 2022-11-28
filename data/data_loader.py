# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 13:46
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import data.copy_and_paste

train_transform = A.Compose([
    A.RandomScale(scale_limit=(1, 1.2), p=0.5),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(p=0.5),
    A.PixelDropout(p=0.5),
    ToTensorV2(True),
])

val_transform = A.Compose([
    A.RandomCrop(512, 512),
    ToTensorV2(True)
])

# UAV_image
# MEAN = [0.382, 0.372, 0.366]
# STD = [0.134, 0.122, 0.111]

# earthquake_crack
MEAN = [0.311, 0.307, 0.307]
STD = [0.165, 0.155, 0.143]


train_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train.txt'
val_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val.txt'
test_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test.txt'

raw_train_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/img/'
raw_train_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/mask/'

raw_val_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/img/'
raw_val_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/mask/'

raw_test_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/img/'
raw_test_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/mask/'


class data_prefetcher:
    def __init__(self, loader: DataLoader, mean, std):
        self.next_input = None
        self.next_target = None
        self.MEAN = mean
        self.STD = std
        self.num = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([self.MEAN[0] * 255, self.MEAN[1] * 255, self.MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([self.STD[0] * 255, self.STD[1] * 255, self.STD[2] * 255]).cuda().view(1, 3, 1, 1)
        self.preload()
        self.range = range(self.num)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            self.next_target = self.next_target.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        it_batch = self.range[item]
        # print(it_batch)
        return self.next()


class Custom_Dataset(Dataset):
    def __init__(self, raw_image_path, raw_mask_path, size, data_txt, transformer=None):
        self.raw_image_path = raw_image_path
        self.raw_mask_path = raw_mask_path
        self.size = size
        self.data_txt = data_txt
        self.transform = transformer
        with open(self.data_txt, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item].split(',')[0]))
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)

        self.raw_mask = cv2.imread(os.path.join(self.raw_mask_path, self.file_list[item].split(',')[1]), cv2.IMREAD_GRAYSCALE)

        self.raw_mask = np.expand_dims(self.raw_mask, axis=-1)  / 255
        self.raw_mask = np.concatenate([1 - self.raw_mask, self.raw_mask], axis=-1)

        self.transformed = self.transform(image=self.raw_image, mask=self.raw_mask)
        self.raw_image, self.raw_mask = self.transformed['image'], self.transformed['mask']

        return self.raw_image, self.raw_mask


def get_Image_Mask_Dataset(re_size, batch_size):
    train_dataset = Custom_Dataset(raw_image_path=raw_train_dir,
                                   raw_mask_path=raw_train_mask_dir,
                                   size=re_size,
                                   data_txt=train_data_txt,
                                   transformer=train_transform)

    val_dataset = Custom_Dataset(raw_image_path=raw_val_dir,
                                 raw_mask_path=raw_val_mask_dir,
                                 size=re_size,
                                 data_txt=val_data_txt,
                                 transformer=val_transform)

    test_dataset = Custom_Dataset(raw_image_path=raw_test_dir,
                                  raw_mask_path=raw_test_mask_dir,
                                  size=re_size,
                                  data_txt=test_data_txt,
                                  transformer=val_transform)

    # when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again

    Train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    Train_loader, Val_loader, Test_loader = data_prefetcher(Train_loader, MEAN, STD), \
                                            data_prefetcher(Val_loader, MEAN, STD), \
                                            data_prefetcher(Test_loader, MEAN, STD)

    return Train_loader, Val_loader, Test_loader

