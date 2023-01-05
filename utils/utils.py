# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 3:35
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : utils.py
# @Software: PyCharm
def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_ = {}
    for k in list(keys)[start:end]:
        c = k
        dict_[c] = adict[k]
    return dict_


def dict_load(adict, cdict):
    # adict source / cdict target
    for i, j in zip(adict, cdict):
        cdict[j] = adict[i]
    return cdict
