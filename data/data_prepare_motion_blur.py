# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 12:41
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_prepare_motion_blur.py
# @Software: PyCharm
import os
import cv2
import math
import numpy as np
import numpy.fft as fft
from albumentations import MotionBlur

train_dir, val_dir, test_dir = 'L:/crack_segmentation_in_UAV_images/earthquake_crack/train/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/val/img', \
                               'L:/crack_segmentation_in_UAV_images/earthquake_crack/test/img'

# orientation
# 15, 21
# 9, 15
# 3, 7


def straight_motion_psf(image_size: tuple, motion_angle: int, motion_dis: int):
    """
    直线运动模糊
    :param image_size:
    :param motion_angle:
    :param motion_dis:
    :return:
    """
    psf = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        psf[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return psf / psf.sum()


def gaussian_blur_process(input: np.ndarray, degree=21):
    """
    高斯模糊
    :param input:
    :param degree:
    :return:
    """
    blurred = cv2.GaussianBlur(input, ksize=(degree, degree), sigmaX=0, sigmaY=0)
    return blurred


def make_blurred(input: np.ndarray, psf: np.ndarray, eps: float):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def wiener(input: np.ndarray, psf: np.ndarray, eps: float, K=0.01):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    # np.conj是计算共轭值
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def get_motion_blur_image(input: np.ndarray, motion_angle: int, motion_dis: int):
    PSF = straight_motion_psf((input.shape[0], input.shape[1]), motion_angle, motion_dis)
    R, G, B = cv2.split(input)
    blurred_image_R, blurred_image_G, blurred_image_B = \
        np.abs(make_blurred(R, PSF, 1e-3)), np.abs(make_blurred(G, PSF, 1e-3)), np.abs(make_blurred(B, PSF, 1e-3))
    blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])

    return np.uint8(blurred_image)


# function = MotionBlur(always_apply=True, p=1.0, blur_limit=(15, 21))
function = get_motion_blur_image
for data_dir in [train_dir, val_dir, test_dir]:
    for path in os.listdir(data_dir):

        img = cv2.imread(os.path.join(data_dir, path))
        blur_img = function(img, 45, 10)
        cv2.imwrite(os.path.join(data_dir[:-3] + 'blur_img_orientation', path), blur_img)
