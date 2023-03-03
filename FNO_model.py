# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/3/2 20:40
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：FNO_model.py
@File ：FNO_model.py
"""
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from utils import activation_dict, params_initial
from functools import partial


class SpectralConv2d(nn.Layer):
    '''
    2维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,  # number of fourier modes
                 dropout=0.0,
                 norm='ortho',
                 activation='gelu',
                 return_freq=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_dict[activation]
        self.return_freq = return_freq
        self.linear = nn.Conv2D(self.in_dim, self.out_dim, 1)  # for residual

        self.weights1 = paddle.create_parameter([in_dim, out_dim, self.modes1, self.modes2, 2], dtype=paddle.float32)
        w1 = params_initial('xavier_normal', shape=self.weights1.shape,
                            gain=1 / (in_dim * out_dim) * np.sqrt(in_dim + out_dim))
        self.weights1.set_value(w1)
        self.weights2 = paddle.create_parameter([in_dim, out_dim, self.modes1, self.modes2, 2], dtype=paddle.float32)
        w2 = params_initial('xavier_normal', shape=self.weights2.shape,
                            gain=1 / (in_dim * out_dim) * np.sqrt(in_dim + out_dim))
        self.weights2.set_value(w2)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = partial(paddle.einsum, "bixy,ioxy->boxy")
        return paddle.stack([
            op(input[..., 0], weights[..., 0]) - op(input[..., 1], weights[..., 1]),
            op(input[..., 1], weights[..., 0]) + op(input[..., 0], weights[..., 1])
        ], axis=-1)

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.linear(x)
        x = self.dropout(x)
        x_ft = paddle.fft.rfft2(x, norm=self.norm)
        x_ft = paddle.stack([x_ft.real(), x_ft.imag()], axis=-1)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros((batch_size, self.out_dim, x.shape[-2], x.shape[-1] // 2 + 1, 2), dtype=paddle.float32)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = paddle.complex(out_ft[..., 0], out_ft[..., 1])

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class FNO2d(nn.Layer):
    """
        2维FNO网络
    """

    def __init__(self, in_dim, out_dim, modes=(8, 8), width=32, depth=4, steps=1, padding=2, activation='gelu',
                 dropout=0.0):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.padding = padding  # pad the domain if input is non-periodic
        self.activation = activation
        self.dropout = dropout
        self.fc0 = nn.Linear(steps * in_dim, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.LayerList()
        for i in range(self.depth):
            self.convs.append(
                SpectralConv2d(self.width, self.width, self.modes, activation=self.activation, dropout=self.dropout))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        forward computation
        """
        # x dim = [b, t*v+2, x1, x2]
        x = x.transpose((0, 2, 3, 1))
        # x = paddle.concat((x, grid), axis=-1)
        x = self.fc0(x)
        x = x.transpose((0, 3, 1, 2))

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        for i in range(self.depth):
            x = self.convs[i](x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.transpose((0, 2, 3, 1))  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.transpose((0, 3, 1, 2))


if __name__ == '__main__':
    layer = SpectralConv2d(1, 5, modes=(12, 12))
    x = paddle.ones([10, 1, 64, 64])
    x = layer(x)
    print(x.shape)

    x = paddle.ones([10, 3, 128, 128])
    g = paddle.ones([10, 2, 128, 128])
    layer = FNO2d(in_dim=3, out_dim=3, modes=(16, 16), width=32, depth=4, steps=1, padding=3, activation='gelu')
    y = layer(x)
    print(y.shape)
