# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 20:42
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : metrics.py
# @Software: PyCharm
import torch


def iou(input, target):

    intersection = input * target
    union = (input + target) - intersection
    Iou = (torch.sum(intersection) + torch.tensor(1e-8)) / (torch.sum(union) + torch.tensor(1e-8))
    return Iou


def pr(input, target):

    tp = torch.sum(target * input)
    pp = torch.sum(input)

    return (tp + torch.tensor(1e-8))  / (pp + torch.tensor(1e-8))


def re(input, target):

    tp = torch.sum(target * input)
    pp = torch.sum(target)

    return (tp + torch.tensor(1e-8))  / (pp + torch.tensor(1e-8))


def f1(input, target):

    p = pr(input, target)
    r = re(input, target)

    return 2 * p * r / (p + r + 1e-8)
