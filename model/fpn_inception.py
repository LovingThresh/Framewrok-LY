# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 13:20
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : fpn_inception.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchvision.models.inception import inception_v3
import torch.nn.functional as F


# 构建原则，以model文件中的ResNetGenerator为主
# 将模型分为 Encoder, Extractor, Decoder, Output Head
#

class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pretrained backbone feature extractor
        """

        super().__init__()
        self.inception = inception_v3(pretrained=True)

        self.enc0 = nn.Sequential(self.inception.Conv2d_1a_3x3)

        self.enc1 = nn.Sequential(
            self.inception.Conv2d_2a_3x3,
            self.inception.Conv2d_2b_3x3,
            self.inception.maxpool1,
        )  # 64
        self.enc2 = nn.Sequential(
            self.inception.Conv2d_3b_1x1,
            self.inception.Conv2d_4a_3x3,
            self.inception.maxpool2,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.Mixed_5b,
            self.inception.Mixed_5c,
            self.inception.Mixed_5d,
        )  # 1088
        self.enc4 = nn.Sequential(
            self.inception.Mixed_6a,
            self.inception.Mixed_6b,
            self.inception.Mixed_6c,
            self.inception.Mixed_6d,
            self.inception.Mixed_6e
        )  # 2080

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
                                 norm_layer(num_filters // 2),
                                 nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(768, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(288, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)
        enc0 = nn.ReflectionPad2d((1, 0, 1, 0))(enc0)
        enc1 = self.enc1(enc0)  # 256
        enc1 = nn.ReflectionPad2d((1, 1, 1, 1))(enc1)
        enc2 = self.enc2(enc1)  # 512
        enc2 = nn.ReflectionPad2d((1, 1, 1, 1))(enc2)
        enc3 = self.enc3(enc2)  # 1024
        enc4 = self.enc4(enc3)  # 2048
        enc4 = nn.ReflectionPad2d((1, 0, 1, 0))(enc4)
        # Lateral connections

        # Top-down pathway

        enc4 = self.lateral4(enc4)
        enc3 = self.td1(self.lateral3(enc3) + nn.functional.interpolate(enc4, scale_factor=2, mode="nearest"))
        enc2 = self.td1(self.lateral2(enc2) + enc3)
        enc1 = self.td2(self.lateral1(enc1) + nn.functional.interpolate(enc2, scale_factor=2, mode="nearest"))
        enc0 = self.td3(self.lateral0(enc0) + nn.functional.interpolate(enc1, scale_factor=2, mode="nearest"))

        return enc0, enc1, enc2, enc3, enc4


class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256, mode="image"):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(norm_layer=norm_layer, num_filters=num_filters_fpn)
        self.mode = mode
        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=4, mode="nearest")
        map1 = nn.functional.interpolate(self.head1(map1), scale_factor=2, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        if self.mode == "image":
            res = torch.tanh(final) + x
            res = torch.clamp(res, min=-1, max=1)
        else:
            res = torch.sigmoid(final)

        return res
