# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 20:42
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : metrics.py
# @Software: PyCharm
import torch


def iou(input, target):
    target = (target > 0.5).int()
    input = (input > 0.5).int()

    # input_maxpool = torch.nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=2).cuda()(input.reshape(-1, 1, 512, 512).float())
    # input_maxpool = (input_maxpool.reshape((-1)) > 0.5).int()
    intersection = input * target
    union = (input + target) - intersection
    Iou = (torch.sum(intersection) + torch.tensor(1e-8)) / (torch.sum(union) + torch.tensor(1e-8))
    return Iou


def pr(input, target):
    target = (target > 0.5).int()
    input = (input > 0.5).int()

    # input_maxpool = torch.nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=2).cuda()(
    #     input.reshape(-1, 1, 512, 512).float())
    # input_maxpool = (input_maxpool.reshape((-1)) > 0.5).int()
    tp = torch.sum(target * input)
    pp = torch.sum(input)

    return (tp + torch.tensor(1e-8))  / (pp + torch.tensor(1e-8))


def re(input, target):
    target = (target > 0.5).int()
    input = (input > 0.5).int()
    tp = torch.sum(target * input)
    pp = torch.sum(target)

    return (tp + torch.tensor(1e-8))  / (pp + torch.tensor(1e-8))


def f1(input, target):
    target = (target > 0.5).int()
    input = (input > 0.5).int()
    p = pr(input, target)
    r = re(input, target)

    return 2 * p * r / (p + r + 1e-8)

def correlation(input, target):
    input_vector =  input.reshape((1, -1))
    target_vector = target.reshape((1, -1))
    return torch.corrcoef(torch.cat([input_vector, target_vector], dim=0))[0, 1]
