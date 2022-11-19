# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 0:07
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : tensorboard.py
# @Software: PyCharm

def write_dict(dict_to_write, writer, step):
    for k, v in dict_to_write.items():
        for i, j in v.items():
            writer.add_scalar('{}/{}'.format(k, i), j, step)


def write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step):
    write_dict(train_dict, train_writer_summary, step)
    write_dict(valid_dict, valid_writer_summary, step)

