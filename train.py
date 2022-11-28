# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 22:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train.py
# @Software: PyCharm
import os
import torch

from accelerate import Accelerator

from logger.metrics import AverageMeter
from logger.tensorboard import write_summary
from utils.visualize import visualize_save_pair

accelerator = Accelerator()
device = accelerator.device

# UAV_image
# MEAN = [0.382, 0.372, 0.366]
# STD = [0.134, 0.122, 0.111]

# earthquake_crack
MEAN = [0.311, 0.307, 0.307]
STD = [0.165, 0.155, 0.143]

mean = torch.tensor([MEAN[0] * 255, MEAN[1] * 255, MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
std = torch.tensor([STD[0] * 255, STD[1] * 255, STD[2] * 255]).cuda().view(1, 3, 1, 1)


def Metrics2Value(dict_Metrics):
    for i in dict_Metrics:
        dict_Metrics[i] = dict_Metrics[i].avg
    return dict_Metrics


def calculate_loss(loss_fn, loss_dict: dict, output, target):
    sum_loss = torch.tensor(0, device='cuda', dtype=torch.float32)
    assert isinstance(loss_fn, dict)
    for loss_name, loss_function in loss_fn.items():
        loss_value = loss_function(output, target)

        loss_dict[loss_name].update(loss_value.item(), output.size(0))
        sum_loss += loss_value

    return sum_loss, loss_dict


def eval_mode(output, target):
    return output[:, 1:, :, :], target[:, 1:, :, :].long()


def calculate_eval(eval_fn, eval_dict: dict, output, target, mode_function=None):
    assert isinstance(eval_fn, dict)

    if mode_function is not None:
        output, target = mode_function(output, target)

    output, target = output.reshape((-1)), target.reshape((-1))
    for eval_name, eval_function in eval_fn.items():
        eval_value = eval_function(output.cpu(), target.cpu())
        eval_dict[eval_name].update(eval_value.item(), output.size(0))

    return eval_dict


def train_epoch(train_model, train_load, loss_fn, eval_fn, optimizer, scheduler, epoch, Epochs):
    it = 0
    train_loss_dict = {}
    train_eval_dict = {}

    for loss_metric in loss_fn:
        train_loss_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn:
        train_eval_dict[eval_metrics] = AverageMeter()

    for batch in train_load:
        it = it + 1

        inputs, target = batch

        optimizer.zero_grad()
        output = train_model(inputs)
        loss, train_loss_dict = \
            calculate_loss(loss_fn, train_loss_dict, output, target)
        accelerator.backward(loss)
        optimizer.step()

        if it % 50 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
            print(train_loss_dict)
            print("-" * 80)

    scheduler.step()

    # evaluate the last batch
    with torch.no_grad():
        train_eval_dict = \
            calculate_eval(eval_fn, train_eval_dict, output, target, mode_function=eval_mode)
    print("Epoch:{}/{}, Last Iter Evaluation:,".format(epoch, Epochs))
    print(train_eval_dict)
    print("-" * 80)

    return train_loss_dict, train_eval_dict


def train_generator_epoch(train_generator, train_discriminator, train_load,
                          loss_fn_generator, loss_fn_generator_extra, loss_fn_discriminator,
                          eval_fn_generator,
                          optimizer_generator, optimizer_discriminator,
                          scheduler_generator, scheduler_discriminator, epoch, Epochs, Device='cuda'):
    it = 0
    train_loss_generator_dict = {}
    train_eval_generator_dict = {}

    train_loss_discriminator_dict = {}

    for loss_metric in loss_fn_generator:
        train_loss_generator_dict[loss_metric] = AverageMeter()

    for loss_metric in loss_fn_generator_extra:
        train_loss_generator_dict[loss_metric] = AverageMeter()

    for loss_metric in loss_fn_discriminator:
        train_loss_discriminator_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn_generator:
        train_eval_generator_dict[eval_metrics] = AverageMeter()

    D_weight = 0.5

    for batch in train_load:
        it = it + 1

        optimizer_discriminator.zero_grad()
        inputs, target = batch
        # ------------------------------------------------------------------------------------- #

        real_predict = train_discriminator(target)
        real_label = torch.ones(real_predict.shape, dtype=torch.float32, device=Device)
        loss_dis_real, train_loss_discriminator_dict = \
            calculate_loss(loss_fn_discriminator, train_loss_discriminator_dict, real_predict, real_label)
        loss_dis_real = loss_dis_real * torch.tensor(D_weight, dtype=torch.float32, device=Device)

        # ------------------------------------------------------------------------------------- #

        fake_output = train_generator(inputs)
        fake_predict = train_discriminator(fake_output.detach().clone())
        fake_label = torch.zeros(fake_predict.shape, dtype=torch.float32, device=Device)
        loss_dis_fake, train_loss_discriminator_dict = \
            calculate_loss(loss_fn_discriminator, train_loss_discriminator_dict, fake_predict, fake_label)
        loss_dis_fake = loss_dis_fake * torch.tensor(D_weight, dtype=torch.float32, device=Device)
        loss_dis = loss_dis_real + loss_dis_fake

        loss_dis.bacward()
        optimizer_discriminator.step()
        # ------------------------------------------------------------------------------------- #

        # ------------------------------------------------------------------------------------- #
        #                                   Generator Training                                  #
        # ------------------------------------------------------------------------------------- #

        optimizer_generator.zero_grad()
        gen_predict = train_generator(fake_output)

        # calculate generator loss
        loss_gen, train_loss_generator_dict = calculate_loss(loss_fn_generator, train_loss_generator_dict, gen_predict,
                                                             torch.ones(gen_predict.shape, dtype=torch.float32, device=Device))

        # calculate generator loss extra
        loss_gen_extra, train_loss_generator_dict = calculate_loss(loss_fn_generator_extra, train_loss_generator_dict,
                                                                   fake_output, target)

        loss_gen_sum = loss_gen + loss_gen_extra
        loss_gen_sum.backward()
        optimizer_generator.step()

        if it % 50 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
            print(train_loss_discriminator_dict)
            print(train_loss_generator_dict)
            print("-" * 80)

    scheduler_generator.step()
    scheduler_discriminator.step()
    # evaluate the last batch

    with torch.no_grad():
        train_eval_generator_dict = \
            calculate_eval(eval_fn_generator, train_eval_generator_dict, fake_output, target, mode_function=eval_mode)
    print("Epoch:{}/{}, Last Iter Evaluation:,".format(epoch, Epochs))
    print(train_eval_generator_dict)
    print("-" * 80)

    return train_loss_generator_dict, train_loss_discriminator_dict, train_eval_generator_dict


def validation_epoch(eval_model, eval_load, loss_fn, eval_fn, epoch, Epochs):
    it = 0
    validation_loss_dict = {}
    validation_eval_dict = {}

    for loss_metric in loss_fn:
        validation_loss_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn:
        validation_eval_dict[eval_metrics] = AverageMeter()

    for batch in eval_load:
        it = it + 1

        inputs, target = batch
        output = eval_model(inputs)
        loss, validation_loss_dict = \
            calculate_loss(loss_fn, validation_loss_dict, output, target)
        validation_eval_dict = \
            calculate_eval(eval_fn, validation_eval_dict, output, target, mode_function=eval_mode)

        if it % 20 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(eval_load)))
            print(validation_loss_dict)
            print(validation_eval_dict)
            print("-" * 80)

    return validation_loss_dict, validation_eval_dict


def train(train_model, optimizer, loss_fn, eval_fn,
          train_load, val_load, epochs, scheduler,
          threshold, output_dir, train_writer_summary, valid_writer_summary,
          experiment, comet=False, init_epoch=1, mode=None):
    train_model, optimizer, train_load, val_load = accelerator.prepare(train_model, optimizer, train_load,
                                                                       val_load)

    def train_process(B_comet, experiment_comet, threshold_value=threshold, init_epoch_num=init_epoch):
        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs}')
            print('-' * 10)

            train_model.train()
            train_loss_dict, train_eval_dict = train_epoch(train_model, train_load, loss_fn, eval_fn,
                                                           optimizer, scheduler, epoch, epochs)
            with torch.no_grad():
                validation_loss_dict, validation_eval_dict = validation_epoch(train_model, val_load, loss_fn, eval_fn,
                                                                              epoch, epochs)
            train_loss_dict, train_eval_dict, \
            validation_loss_dict, validation_eval_dict = Metrics2Value(train_loss_dict), \
                                                         Metrics2Value(train_eval_dict), \
                                                         Metrics2Value(validation_loss_dict), \
                                                         Metrics2Value(validation_eval_dict)

            train_dict, validation_dict = {'loss': train_loss_dict, 'eval': train_eval_dict}, \
                                          {'loss': validation_loss_dict, 'eval': validation_eval_dict}
            write_summary(train_writer_summary, valid_writer_summary, train_dict, validation_dict, step=epoch)

            if B_comet:
                experiment_comet.log({'train/': train_dict, 'valid/': validation_dict, 'lr': optimizer.optimizer.defaults['lr']})

            # 这一部分可以根据任务进行调整
            metric = sorted(validation_eval_dict.items())[2][0]

            if validation_eval_dict[metric] > threshold_value:
                torch.save(train_model.state_dict(),
                           os.path.join(output_dir, 'save_model',
                                        'Epoch_{}_eval_{}.pt'.format(epoch, validation_eval_dict[metric])))
                threshold_value = validation_eval_dict['eval_iou']

            # 验证阶段的结果可视化
            save_path = os.path.join(output_dir, 'save_fig')
            visualize_save_pair(train_model, val_load, mean, std, save_path, epoch, mode=mode)

            if (epoch % 100) == 0:
                save_checkpoint_path = os.path.join(output_dir, 'checkpoint')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": train_model.state_dict(),
                    "loss_fn": loss_fn,
                    "eval_fn": eval_fn,
                    "lr_schedule_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(save_checkpoint_path, str(epoch) + '.pth'))
    train_process(comet, experiment)
