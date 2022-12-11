# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 13:46
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import os
import cv2
import numpy as np
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

train_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train.txt'
val_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val.txt'
test_data_txt = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test.txt'

raw_train_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/img/'
raw_train_blur_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/blur_img/'
raw_train_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/mask/'

raw_val_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/img/'
raw_val_blur_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/blur_img/'
raw_val_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/mask/'

raw_test_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/img/'
raw_test_blur_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/blur_img/'
raw_test_mask_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/mask/'


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

        self.raw_mask = cv2.imread(os.path.join(self.raw_mask_path, self.file_list[item].split(',')[1]),
                                   cv2.IMREAD_GRAYSCALE)

        self.raw_mask = np.expand_dims(self.raw_mask, axis=-1) / 255
        self.raw_mask = np.concatenate([1 - self.raw_mask, self.raw_mask], axis=-1)

        self.transformed = self.transform(image=self.raw_image, mask=self.raw_mask)
        self.raw_image, self.raw_mask = self.transformed['image'], self.transformed['mask']

        return self.raw_image, self.raw_mask


class Custom_BlurImage_Dataset(Dataset):
    def __init__(self, raw_image_path, raw_blur_path, size, data_txt, transformer=None):
        self.raw_image_path = raw_image_path
        self.raw_blur_path = raw_blur_path
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

        self.blur_image = cv2.imread(os.path.join(self.raw_blur_path, self.file_list[item].split(',')[0]))
        self.blur_image = cv2.cvtColor(self.blur_image, cv2.COLOR_BGR2RGB)

        self.transformed = self.transform(image=self.blur_image, mask=self.raw_image)
        self.blur_image, self.raw_image = self.transformed['image'], self.transformed['mask']
        self.blur_image, self.raw_image = self.blur_image / 255., self.raw_image / 255.
        self.blur_image, self.raw_image = \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.blur_image), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.raw_image)

        return self.blur_image, self.raw_image


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

    return Train_loader, Val_loader, Test_loader


def get_BlurImage_Mask_Dataset(re_size, batch_size):
    train_dataset = Custom_Dataset(raw_image_path=raw_train_blur_dir,
                                   raw_mask_path=raw_train_mask_dir,
                                   size=re_size,
                                   data_txt=train_data_txt,
                                   transformer=train_transform)

    val_dataset = Custom_Dataset(raw_image_path=raw_val_blur_dir,
                                 raw_mask_path=raw_val_mask_dir,
                                 size=re_size,
                                 data_txt=val_data_txt,
                                 transformer=val_transform)

    test_dataset = Custom_Dataset(raw_image_path=raw_test_blur_dir,
                                  raw_mask_path=raw_test_mask_dir,
                                  size=re_size,
                                  data_txt=test_data_txt,
                                  transformer=val_transform)

    # when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again

    Train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return Train_loader, Val_loader, Test_loader


def get_BlurImage_Image_Dataset(re_size, batch_size):
    train_dataset = Custom_BlurImage_Dataset(raw_image_path=raw_train_dir,
                                             raw_blur_path=raw_train_blur_dir,
                                             size=re_size,
                                             data_txt=train_data_txt,
                                             transformer=train_transform)

    val_dataset = Custom_BlurImage_Dataset(raw_image_path=raw_val_dir,
                                           raw_blur_path=raw_val_blur_dir,
                                           size=re_size,
                                           data_txt=val_data_txt,
                                           transformer=val_transform)

    test_dataset = Custom_BlurImage_Dataset(raw_image_path=raw_test_dir,
                                            raw_blur_path=raw_test_blur_dir,
                                            size=re_size,
                                            data_txt=test_data_txt,
                                            transformer=val_transform)

    # when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again

    Train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return Train_loader, Val_loader, Test_loader
