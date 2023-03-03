# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/27 15:04
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : UNet_model.py
"""
import math
import paddle
import paddle.nn as nn
import numpy as np
from utils import activation_dict, params_initial


def blockUNet(in_c, out_c, name, transposed=False, bn=True, activation='gelu', size=4, pad=1, dropout=0.):
    block = nn.Sequential()

    activation = activation_dict[activation]

    block.add_sublayer('%s_activation' % name, activation)

    if not transposed:
        block.add_sublayer('%s_conv' % name, nn.Conv2D(in_c, out_c, kernel_size=size, stride=2, padding=pad))
    else:
        block.add_sublayer('%s_upsam' % name,
                           nn.Upsample(scale_factor=2, mode='bilinear'))  # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_sublayer('%s_tconv' % name,
                           nn.Conv2D(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad))
    if bn:
        block.add_sublayer('%s_bn' % name, nn.BatchNorm2D(out_c))
    if dropout > 0.:
        block.add_sublayer('%s_dropout' % name, nn.Dropout2D(dropout))
    return block


class UNet2d(nn.Layer):
    def __init__(self, channelExponent=6, dropout=0.):
        super(UNet2d, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_sublayer('layer1_conv', nn.Conv2D(3, channels, 4, 2, 1, bias_attr=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True,
                                activation='leakyrelu', dropout=dropout)
        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True,
                                 activation='leakyrelu', dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True,
                                activation='leakyrelu', dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels * 4, channels * 8, 'layer4', transposed=False, bn=True,
                                activation='leakyrelu', dropout=dropout, size=4)  # note, size 4!
        self.layer5 = blockUNet(channels * 8, channels * 8, 'layer5', transposed=False, bn=True,
                                activation='leakyrelu', dropout=dropout, size=2, pad=0)
        self.layer6 = blockUNet(channels * 8, channels * 8, 'layer6', transposed=False, bn=False,
                                activation='leakyrelu', dropout=dropout, size=2, pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels * 8, channels * 8, 'dlayer6', transposed=True, bn=True,
                                 activation='relu', dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, 'dlayer5', transposed=True, bn=True,
                                 activation='relu', dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, 'dlayer4', transposed=True, bn=True,
                                 activation='relu', dropout=dropout)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True,
                                 activation='relu', dropout=dropout)
        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True,
                                  activation='relu', dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True,
                                 activation='relu', dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_sublayer('dlayer1_relu', nn.ReLU())
        self.dlayer1.add_sublayer('dlayer1_tconv', nn.Conv2DTranspose(channels * 2, 3, 4, 2, 1))

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = params_initial('normal', shape=m.weight.shape, scale=0.02, gain=1.0)
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                v = params_initial('normal', shape=m.weight.shape, scale=0.02, gain=1.0)
                m.weight.set_value(v)
                v = params_initial('constant', shape=m.bias.shape, gain=0.0)
                m.bias.set_value(v)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b = self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = paddle.concat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = paddle.concat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = paddle.concat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = paddle.concat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = paddle.concat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = paddle.concat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


if __name__ == '__main__':
    Net_model = UNet2d()
    x = paddle.ones([64, 3, 128, 128], dtype=paddle.float32)
    y = Net_model(x)
    print(y.shape)
