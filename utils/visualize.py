# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 23:09
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : visualize.py
# @Software: PyCharm
import cv2
# import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt


def dim_to_numpy(x: torch.Tensor or np.array):
    if (x.ndim == 4) & (torch.is_tensor(x)):
        x = x.squeeze(0)
        x = x.cpu().numpy()
        x = x.transpose(1, 2, 0)
    elif (x.ndim == 3) & (torch.is_tensor(x)):
        x = x.cpu().numpy()
    return np.uint8(x)


def plot(x: np.array or torch.Tensor, size=(10, 10)):
    x = dim_to_numpy(x)
    assert x.ndim == 3 or (x.ndim == 2)
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(x)
    plt.show()


def visualize_model(model: torch.nn.Module, image, image_pair=False):
    model.eval()
    assert image.ndim == 4
    prediction = model(image)
    if image_pair:
        image, prediction = dim_to_numpy(image), dim_to_numpy(prediction)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(prediction)
        plt.show()
    else:
        plot(prediction)


def visualize_pair(train_loader, input_size, mean, std, plot_switch=True, mode='image'):

    a = next(iter(train_loader))
    input_tensor_numpy = a[0][0:1].mul_(std).add_(mean).cpu().numpy()
    input_tensor_numpy = input_tensor_numpy.transpose(0, 2, 3, 1)
    input_tensor_numpy = input_tensor_numpy.reshape(input_size[0], input_size[1], 3)
    input_tensor_numpy = np.uint8(input_tensor_numpy)
    if plot_switch:
        plot(input_tensor_numpy)

    if mode == 'image':
        output_tensor_numpy = a[1][0:1].mul_(std).add_(mean).cpu().numpy()
    else:
        output_tensor_numpy = a[1][0:1].cpu().numpy()

    output_tensor_numpy = output_tensor_numpy.transpose(0, 2, 3, 1)

    if output_tensor_numpy.shape[-1] == 2:
        output_tensor_numpy = output_tensor_numpy[0, :, :, 0:1].repeat(3, axis=-1)

    if mode == 'image':
        output_tensor_numpy = np.uint8(output_tensor_numpy)
    else:
        output_tensor_numpy = np.uint8(output_tensor_numpy * 255)

    if plot_switch:
        plot(output_tensor_numpy)

    return input_tensor_numpy, output_tensor_numpy


def visualize_save_pair(val_model: torch.nn.Module, val_loader, mean, std, save_path, epoch, num=0, mode='image'):

    a = next(iter(val_loader))
    i = 1
    input_tensor_numpy = a[0][0 + i: 1 + i].mul_(std).add_(mean).cpu().numpy()
    input_tensor_numpy = input_tensor_numpy.transpose(0, 2, 3, 1)
    input_tensor_numpy = input_tensor_numpy.reshape(input_tensor_numpy.shape[1],input_tensor_numpy.shape[2], 3)
    input_tensor_numpy = np.uint8(input_tensor_numpy)

    if mode == 'image':
        output_tensor_numpy = a[1][0 + i:1 + i].mul_(std).add_(mean).cpu().numpy()
    else:
        output_tensor_numpy = a[1][0 + i:1 + i].cpu().numpy()

    cv2.imwrite('{}/{}_input.jpg'.format(save_path, epoch + num), input_tensor_numpy)

    output_tensor_numpy = output_tensor_numpy.transpose(0, 2, 3, 1)

    if mode == 'image':
        output_tensor_numpy = output_tensor_numpy.reshape(output_tensor_numpy.shape[1], output_tensor_numpy.shape[2], 3)
        output_tensor_numpy = cv2.cvtColor(output_tensor_numpy, cv2.COLOR_BGR2RGB)

    else:
        output_tensor_numpy = output_tensor_numpy.reshape(output_tensor_numpy.shape[1], output_tensor_numpy.shape[2], 2)
        output_tensor_numpy = output_tensor_numpy[:, :, 1]

    cv2.imwrite('{}/{}_output.jpg'.format(save_path, epoch + num), np.uint8(output_tensor_numpy * 255))

    val_model.train(True)
    with torch.no_grad():
        predict_tensor = val_model(a[0][0 + i: 1 + i])
    predict_tensor_numpy = predict_tensor.detach().cpu().numpy()
    predict_tensor_numpy = predict_tensor_numpy.transpose(0, 2, 3, 1)

    if predict_tensor_numpy.shape[-1] == 2:
        predict_tensor_numpy = predict_tensor_numpy[:, :, :, 1:].repeat(3, axis=-1)

    predict_tensor_numpy = predict_tensor_numpy.reshape(predict_tensor_numpy.shape[1], predict_tensor_numpy.shape[2], 3)

    if mode == 'image':
        predict_tensor_numpy = torch.tensor(predict_tensor_numpy).mul_(std).add_(mean).numpy()
        predict_tensor_numpy = cv2.cvtColor(predict_tensor_numpy, cv2.COLOR_BGR2RGB)
    else:
        predict_tensor_numpy = predict_tensor_numpy * 255

    cv2.imwrite('{}/{}_predict.jpg'.format(save_path, epoch + num), np.uint8(predict_tensor_numpy))


def image2tensor(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.transpose(0, 2)
    image = image.transpose(1, 2)

    return image.unsqueeze(0)


def tensor2array(tensor):

    tensor = tensor.squeeze(0)
    tensor = tensor.transpose(0, 2)
    tensor = tensor.transpose(0, 1)
    array = tensor.numpy()

    return array
