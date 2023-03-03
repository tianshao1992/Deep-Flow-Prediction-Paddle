# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/27 20:05
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：utils.py
@File ：utils.py
"""

import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm
import paddle.nn as nn

activation_dict = \
    {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leakyrelu': nn.LeakyReLU(0.2), 'elu': nn.ELU()}


def calculate_fan_in_and_fan_out(shape):
    # dimensions = tensor.dim()
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def params_initial(initialization, shape, scale=1.0, gain=1.0):
    if initialization == 'constant':
        Weight = gain * np.ones(shape).astype('float32')
    elif initialization == 'normal':
        Weight = gain * np.random.normal(loc=0., scale=scale, size=shape).astype('float32')
    elif initialization == 'xavier_Glorot_normal':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        Weight = gain * np.random.normal(loc=0., scale=scale, size=shape) / np.sqrt(in_dim).astype('float32')
    elif initialization == 'xavier_normal':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        std = np.sqrt(2. / (in_dim + out_dim))
        Weight = gain * np.random.normal(loc=0., scale=std, size=shape).astype('float32')
    elif initialization == 'uniform':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        a = np.sqrt(1. / in_dim)
        Weight = gain * np.random.uniform(low=-a, high=a, size=shape).astype('float32')
    elif initialization == 'xavier_uniform':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        a = np.sqrt(6. / (in_dim + out_dim))
        Weight = gain * np.random.uniform(low=-a, high=a, size=shape).astype('float32')
    else:
        print("initialization error!")
        exit(1)
    return Weight


# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint: print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()


# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


# image output
def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)

    s = outputs.shape[1]  # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ((s + 10) * 3, s * 3), color=(255, 255, 255))
        # BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else:  # from -1,1 to 0,1
            outputs[i] -= -1.
            targets[i] -= -1.
            outputs[i] /= 2.
            targets[i] /= 2.

        if not saveMontage:
            suffix = ""
            if i == 0:
                suffix = "_pressure"
            elif i == 1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.RdBu_r(outputs[i], bytes=True))
            im = im.resize((512, 512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.RdBu_r(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512, 512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            im = Image.fromarray(cm.RdBu_r(targets[i], bytes=True))
            new_im.paste(im, ((s + 10) * i, s * 0))
            im = Image.fromarray(cm.RdBu_r(outputs[i], bytes=True))
            new_im.paste(im, ((s + 10) * i, s * 1))
            imE = Image.fromarray(np.abs(targets[i] - outputs[i]) * 10. * 256.)
            new_im.paste(imE, ((s + 10) * i, s * 2))

            # im = Image.fromarray(targets[i] * 256.)
            # BW_im.paste(im, ( (s+10)*i, s*0))
            # im = Image.fromarray(outputs[i] * 256.)
            # BW_im.paste(im, ( (s+10)*i, s*1))
            # imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            # BW_im.paste(imE, ( (s+10)*i, s*2))

    if saveMontage:
        new_im.save(filename + ".png")
        # BW_im.save( filename + "_bw.png")


# save single image
def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.RdBu_r(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)


# read data split from command line
def readProportions():
    flag = True
    while flag:
        input_proportions = input(
            "Enter total numer for training files and proportions for training (normal, superimposed, sheared respectively) seperated by a comma such that they add up to 1: ")
        input_p = input_proportions.split(",")
        prop = [float(x) for x in input_p]
        if prop[1] + prop[2] + prop[3] == 1:
            flag = False
        else:
            print("Error: poportions don't sum to 1")
            print("##################################")
    return (prop)


# helper from data/utils
def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)
