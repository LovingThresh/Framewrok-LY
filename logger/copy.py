# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 0:30
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : copy.py
# @Software: PyCharm
import os
import json
import shutil
import datetime


def copy_and_upload(hyper_params, src_path):
    a = str(datetime.datetime.now())
    b = list(a)
    b[10] = '-'
    b[13] = '-'
    b[16] = '-'
    output_dir = ''.join(b)
    output_dir = os.path.join(src_path, output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'summary'))
    os.mkdir(os.path.join(output_dir, 'save_fig'))
    os.mkdir(os.path.join(output_dir, 'save_model'))
    os.mkdir(os.path.join(output_dir, 'checkpoint'))
    hyper_params['output_dir'] = output_dir
    hyper_params['ex_date'] = a[:10]
    shutil.copytree('data', '{}/{}'.format(output_dir, 'data'))
    shutil.copytree('utils', '{}/{}'.format(output_dir, 'utils'))
    shutil.copytree('model', '{}/{}'.format(output_dir, 'model'))
    shutil.copytree('logger', '{}/{}'.format(output_dir, 'logger'))
    shutil.copytree('deployment', '{}/{}'.format(output_dir, 'deployment'))

    # 个人热代码
    shutil.copy('main.py', output_dir)
    shutil.copy('train.py', output_dir)
    shutil.copy('test.py', output_dir)

    hyper_params['output_dir'] = output_dir

    with open('{}/hyper_params.json'.format(output_dir), 'w') as fp:
        json.dump(hyper_params, fp)
    return output_dir
