# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 0:23
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : metrics.py
# @Software: PyCharm
import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = torch.tensor(0, device='cuda', dtype=torch.float32)
        self.avg = torch.tensor(0, device='cuda', dtype=torch.float32)
        self.sum = torch.tensor(0, device='cuda', dtype=torch.float32)
        self.count = torch.tensor(0, device='cuda', dtype=torch.float32)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '[val: ' + str(round(self.val.item(), 4)) + '][' + 'avg: ' + str(round(self.avg.item(), 4)) + ']'