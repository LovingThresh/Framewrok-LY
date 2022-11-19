# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 22:57
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : wandb.py
# @Software: PyCharm
import wandb


def wandb_init(config: dict, project: str, notes: str = None, tags: str = None, group: str = None, tensorboard: bool = False):
    run = wandb.init(
        project=project,
        notes=notes,
        tags=tags,
        config=config,
        group=group,
        tensorboard=tensorboard
    )

    return run


# 注意对train与validation的区别
def run_define(run, loss_dict, eval_dict):

    for loss in loss_dict:
        run.define_metric(loss, summary='min')

    for evaluation in eval_dict:
        run.define_metric(evaluation, summary='max')

    return run

