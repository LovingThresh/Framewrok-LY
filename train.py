# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 22:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train.py
# @Software: PyCharm
import copy
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
# MEAN = [0.311, 0.307, 0.307]
# STD = [0.165, 0.155, 0.143]

# Normalize_crack
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

mean = torch.tensor([MEAN[0] * 255, MEAN[1] * 255, MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
std = torch.tensor([STD[0] * 255, STD[1] * 255, STD[2] * 255]).cuda().view(1, 3, 1, 1)


class data_prefetcher:
    def __init__(self, loader, Mean, Std, mode):
        self.next_input = None
        self.next_target = None
        self.MEAN = Mean
        self.STD = Std
        self.mode = mode
        self.num = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([self.MEAN[0] * 255, self.MEAN[1] * 255, self.MEAN[2] * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([self.STD[0] * 255, self.STD[1] * 255, self.STD[2] * 255]).cuda().view(1, 3, 1, 1)
        self.preload()
        self.range = range(self.num)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

            self.next_target = self.next_target.float()
            if self.mode == 'image':
                self.next_target = self.next_target.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        _ = self.range[item]
        return self.next()


def Metrics2Value(dict_Metrics):
    for i in dict_Metrics:
        dict_Metrics[i] = dict_Metrics[i].avg
    return dict_Metrics


def calculate_loss(loss_fn, loss_dict: dict, output, target):
    sum_loss = torch.tensor(0, device='cuda', dtype=torch.float32)
    assert isinstance(loss_fn, dict)
    for loss_name, loss_function in loss_fn.items():
        loss_value = loss_function(output, target)
        loss_dict[loss_name].update(loss_value, output.size(0))
        sum_loss += loss_value

    return sum_loss, loss_dict


def eval_mode(output, target):
    return output[:, 1:, :, :], target[:, 1:, :, :].long()


def eval_gan_mode(output, target):
    gan_output, gan_target = copy.deepcopy(output), copy.deepcopy(target)
    gan_output, gan_target = gan_output.mul_(std).add_(mean), gan_target.mul_(std).add_(mean)
    return gan_output, gan_target


def calculate_eval(eval_fn, eval_dict: dict, output, target, mode='image'):
    assert isinstance(eval_fn, dict)

    if mode == 'segmentation':
        mode_function = eval_mode
        output, target = mode_function(output, target)
        output, target = output.reshape((-1)), target.reshape((-1))

    elif mode == 'image':
        mode_function = eval_gan_mode
        output, target = mode_function(output, target)

    for eval_name, eval_function in eval_fn.items():
        eval_value = eval_function(output, target)
        eval_dict[eval_name].update(eval_value, output.size(0))

    return eval_dict


def train_epoch(train_model, train_load, loss_fn, eval_fn, optimizer, scheduler, epoch, Epochs, mode='image'):
    it = 0
    train_loss_dict = {}
    train_eval_dict = {}
    output, target = None, None
    for loss_metric in loss_fn:
        train_loss_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn:
        train_eval_dict[eval_metrics] = AverageMeter()

    train_load = data_prefetcher(train_load, MEAN, STD, mode)

    for batch in train_load:
        it = it + 1

        inputs, target = batch

        optimizer.zero_grad()
        output = train_model(inputs)
        loss, train_loss_dict = \
            calculate_loss(loss_fn, train_loss_dict, output, target)
        loss.backward()
        optimizer.step()

        if it % 200 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
            print(train_loss_dict)
            print("-" * 80)

            with torch.no_grad():
                train_eval_dict = \
                    calculate_eval(eval_fn, train_eval_dict, output, target, mode=mode)

    scheduler.step()

    # evaluate the last batch
    with torch.no_grad():
        train_eval_dict = \
            calculate_eval(eval_fn, train_eval_dict, output, target, mode=mode)

    print("Epoch:{}/{}, Last Iter Evaluation:,".format(epoch, Epochs))
    print(train_eval_dict)
    print("-" * 80)

    return train_loss_dict, train_eval_dict


def train_generator_epoch(train_generator, train_discriminator, train_load,
                          loss_fn_generator, loss_fn_generator_extra, loss_fn_discriminator,
                          eval_fn_generator,
                          optimizer_generator, optimizer_discriminator,
                          scheduler_generator, scheduler_discriminator, epoch, Epochs, Device='cuda', mode='image'):
    it = 0
    train_loss_generator_extra_dict = {}
    train_loss_generator_dict = {}
    train_eval_generator_dict = {}

    train_loss_discriminator_dict = {}

    for loss_metric in loss_fn_generator:
        train_loss_generator_dict[loss_metric] = AverageMeter()

    for loss_metric in loss_fn_generator_extra:
        train_loss_generator_extra_dict[loss_metric] = AverageMeter()

    for loss_metric in loss_fn_discriminator:
        train_loss_discriminator_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn_generator:
        train_eval_generator_dict[eval_metrics] = AverageMeter()

    D_weight = torch.tensor(0.5, dtype=torch.float32, device=Device, requires_grad=True)
    fake_gen_output, target = None, None
    # train_load = data_prefetcher(train_load, MEAN, STD, mode=mode)

    for batch in train_load:
        it = it + 1

        input, target = batch
        # ------------------------------------------------------------------------------------- #
        #                               optimize generator                                      #
        # ------------------------------------------------------------------------------------- #
        for p in train_discriminator.parameters():
            p.requires_grad = False

        optimizer_generator.zero_grad()
        fake_gen_output = train_generator(input)
        # loss_fn_generator_extra includes pixel loss and perceptual loss
        loss_gen_extra, train_loss_generator_extra_dict = \
            calculate_loss(loss_fn_generator_extra, train_loss_generator_extra_dict, fake_gen_output, target)

        # gan generator loss
        fake_gen_pred = train_discriminator(fake_gen_output)
        loss_gen_real, train_loss_generator_dict = \
            calculate_loss(loss_fn_generator, train_loss_generator_dict,
                           fake_gen_pred, torch.ones(fake_gen_pred.shape, dtype=torch.float32, device=Device))

        loss_gen_total = loss_gen_extra + loss_gen_real
        loss_gen_total.backward()
        optimizer_generator.step()

        # ------------------------------------------------------------------------------------- #
        #                            optimize discriminator                                     #
        # ------------------------------------------------------------------------------------- #
        for p in train_discriminator.parameters():
            p.requires_grad = True

        optimizer_discriminator.zero_grad()
        # real loss
        real_dis_pred = train_discriminator(target)
        loss_dis_real, train_loss_discriminator_dict = \
            calculate_loss(loss_fn_discriminator, train_loss_discriminator_dict,
                           real_dis_pred, torch.ones(real_dis_pred.shape, dtype=torch.float32, device=Device))

        # fake loss
        fake_dis_pred = train_discriminator(fake_gen_output.detach())
        loss_dis_fake, train_loss_discriminator_dict = \
            calculate_loss(loss_fn_discriminator, train_loss_discriminator_dict,
                           fake_dis_pred, torch.zeros(fake_dis_pred.shape, dtype=torch.float32, device=Device))
        loss_dis_total = D_weight * (loss_dis_real + loss_dis_fake)
        loss_dis_total.backward()
        optimizer_discriminator.step()

        # ------------------------------------------------------------------------------------- #
        #                             training status indicator                                 #
        # ------------------------------------------------------------------------------------- #
        if it % 200 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
            print(train_loss_discriminator_dict)
            print(train_loss_generator_dict)
            print("-" * 80)

            with torch.no_grad():
                train_eval_generator_dict = \
                    calculate_eval(eval_fn_generator, train_eval_generator_dict,
                                   fake_gen_output, target, mode=mode)

    scheduler_generator.step()
    scheduler_discriminator.step()

    # evaluate the last batch
    with torch.no_grad():
        train_eval_generator_dict = \
            calculate_eval(eval_fn_generator, train_eval_generator_dict,
                           fake_gen_output, target, mode=mode)

    print("Epoch:{}/{}, Last Iter Evaluation:,".format(epoch, Epochs))
    print(train_eval_generator_dict)
    print("-" * 80)

    return train_loss_generator_dict, train_loss_generator_extra_dict, \
        train_loss_discriminator_dict, train_eval_generator_dict


def validation_epoch(eval_model, eval_load, loss_fn, eval_fn, epoch, Epochs, mode='image'):
    it = 0
    validation_loss_dict = {}
    validation_eval_dict = {}

    for loss_metric in loss_fn:
        validation_loss_dict[loss_metric] = AverageMeter()

    for eval_metrics in eval_fn:
        validation_eval_dict[eval_metrics] = AverageMeter()

    eval_load = data_prefetcher(eval_load, MEAN, STD, mode)

    for batch in eval_load:
        it = it + 1

        inputs, target = batch
        output = eval_model(inputs)
        loss, validation_loss_dict = \
            calculate_loss(loss_fn, validation_loss_dict, output, target)
        validation_eval_dict = \
            calculate_eval(eval_fn, validation_eval_dict, output, target, mode=mode)

        if it % 50 == 0:
            print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(eval_load)))
            print(validation_loss_dict)
            print(validation_eval_dict)
            print("-" * 80)

    return validation_loss_dict, validation_eval_dict


def train(train_model, optimizer, loss_fn, eval_fn,
          train_load, val_load, epochs, scheduler,
          threshold, output_dir, train_writer_summary, valid_writer_summary,
          experiment, comet=False, init_epoch=1, mode=None, metrix_number=2):
    train_model, optimizer, scheduler, train_load, val_load = accelerator.prepare(train_model, optimizer, scheduler,
                                                                                  train_load, val_load)

    def train_process(B_comet, experiment_comet, threshold_value=threshold,
                      init_epoch_num=init_epoch, save_number=metrix_number):
        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs}')
            print('-' * 10)

            train_model.train()
            train_loss_dict, train_eval_dict = train_epoch(train_model, train_load, loss_fn, eval_fn,
                                                           optimizer, scheduler, epoch, epochs, mode=mode)
            with torch.no_grad():
                validation_loss_dict, validation_eval_dict = validation_epoch(train_model, val_load, loss_fn, eval_fn,
                                                                              epoch, epochs, mode=mode)
            train_loss_dict, train_eval_dict, \
                validation_loss_dict, validation_eval_dict = Metrics2Value(train_loss_dict), \
                Metrics2Value(train_eval_dict), \
                Metrics2Value(validation_loss_dict), \
                Metrics2Value(validation_eval_dict)

            train_dict, validation_dict = {'loss': train_loss_dict, 'eval': train_eval_dict,
                                           'lr': {'lr': optimizer.state_dict()['param_groups'][0]['lr']}}, \
                {'loss': validation_loss_dict, 'eval': validation_eval_dict}
            write_summary(train_writer_summary, valid_writer_summary, train_dict, validation_dict, step=epoch)

            # 这一部分可以根据任务进行调整
            metric = sorted(validation_eval_dict.items())[save_number][0]

            if validation_eval_dict[metric] > threshold_value:
                torch.save(train_model.state_dict(),
                           os.path.join(output_dir, 'save_model',
                                        'Epoch_{}_eval_{}.pt'.format(epoch, validation_eval_dict[metric])))
                threshold_value = validation_eval_dict['eval_iou']

            # 验证阶段的结果可视化
            save_path = os.path.join(output_dir, 'save_fig')
            val_loader = data_prefetcher(val_load, MEAN, STD, mode=mode)
            visualize_save_pair(train_model, val_loader, mean, std, save_path, epoch, mode=mode)

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


def train_gen_dis(train_model_G, train_model_D,
                  optimizer_G, optimizer_D,
                  loss_fn_generator, loss_gan_g, loss_gan_d,
                  eval_fn_generator,
                  train_load, val_load, epochs,
                  scheduler_generator, scheduler_discriminator,
                  threshold, output_dir, train_writer_summary, valid_writer_summary,
                  experiment, comet=False, init_epoch=1, mode=None, metrix_number=2):
    train_model_G, train_model_D, optimizer_G, optimizer_D, \
        scheduler_generator, scheduler_discriminator, train_load, val_load = \
        accelerator.prepare(train_model_G, train_model_D, optimizer_G, optimizer_D,
                            scheduler_generator, scheduler_discriminator, train_load, val_load)

    def train_process(B_comet, experiment_comet, threshold_value=threshold,
                      init_epoch_num=init_epoch, save_number=metrix_number):
        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs}')
            print('-' * 10)

            train_model_G.train(True)
            train_model_D.train(True)

            train_loss_generator_dict, train_loss_generator_extra_dict, \
                train_loss_discriminator_dict, train_eval_generator_dict = \
                train_generator_epoch(train_model_G, train_model_D, train_load,
                                      loss_gan_g, loss_fn_generator,
                                      loss_gan_d,
                                      eval_fn_generator, optimizer_G, optimizer_D,
                                      scheduler_generator, scheduler_discriminator, epoch, epochs, mode=mode)

            with torch.no_grad():
                validation_loss_dict, validation_eval_dict = validation_epoch(train_model_G, val_load,
                                                                              loss_fn_generator, eval_fn_generator,
                                                                              epoch, epochs, mode=mode)

            train_loss_generator_dict, train_loss_generator_extra_dict, train_loss_discriminator_dict, \
                train_eval_generator_dict, validation_loss_dict, validation_eval_dict = \
                Metrics2Value(train_loss_generator_dict), \
                Metrics2Value(train_loss_generator_extra_dict), \
                Metrics2Value(train_loss_discriminator_dict), \
                Metrics2Value(train_eval_generator_dict), \
                Metrics2Value(validation_loss_dict), \
                Metrics2Value(validation_eval_dict)

            train_dict, validation_dict = {'loss_g': train_loss_generator_dict,
                                           'loss_g_extra': train_loss_generator_extra_dict,
                                           'loss_d': train_loss_discriminator_dict,
                                           'eval': train_eval_generator_dict,
                                           'lr': {'lr_g': optimizer_G.state_dict()['param_groups'][0]['lr'],
                                                  'lr_d': optimizer_D.state_dict()['param_groups'][0]['lr']}}, \
                {'loss_g_extra': validation_loss_dict, 'eval': validation_eval_dict}
            write_summary(train_writer_summary, valid_writer_summary, train_dict, validation_dict, step=epoch)

            # 这一部分可以根据任务进行调整
            metric = sorted(validation_eval_dict.items())[save_number][0]

            if validation_eval_dict[metric] > threshold_value:
                torch.save(train_model_G.state_dict(),
                           os.path.join(output_dir, 'save_model',
                                        'Epoch_{}_eval_{}.pt'.format(epoch, validation_eval_dict[metric])))
                threshold_value = validation_eval_dict['eval_iou']

            # 验证阶段的结果可视化
            save_path = os.path.join(output_dir, 'save_fig')
            # val_loader = data_prefetcher(val_load, MEAN, STD, mode=mode)
            val_loader = val_load
            visualize_save_pair(train_model_G, val_loader, mean, std, save_path, epoch, mode=mode)

            if (epoch % 100) == 0:
                save_checkpoint_path = os.path.join(output_dir, 'checkpoint')
                torch.save({
                    "epoch": epoch,
                    "model_g_state_dict": train_model_G.state_dict(),
                    "model_d_state_dict": train_model_D.state_dict(),
                    "loss_fn_generator": loss_fn_generator,
                    "loss_gn_gan_g": loss_gan_g,
                    "loss_fn_gan_d": loss_gan_d,
                    "eval_fn_generator": eval_fn_generator,
                    "lr_schedule_g_state_dict": scheduler_generator.state_dict(),
                    "lr_schedule_d_state_dict": scheduler_discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_G.state_dict(),
                    "optimizer_d_state_dict": optimizer_D.state_dict(),
                }, os.path.join(save_checkpoint_path, str(epoch) + '.pth'))

    train_process(comet, experiment)
