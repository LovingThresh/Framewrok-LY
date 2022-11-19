# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 0:23
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : metrics.py
# @Software: PyCharm
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '[val: ' + str(round(self.val, 4)) + '][' + 'avg: ' + str(round(self.avg, 4)) + ']'
