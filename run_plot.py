# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/28 0:12
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_plot.py
@File ：run_plot.py
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, rcParams

config = {'font.family': 'Times New Roman', 'font.size': 25, 'mathtext.fontset': 'stix', 'font.serif': ['SimSun'], }
rcParams.update(config)
# Figure 10

val_loss = []
net = 'UNet'
expo = 6
params = [[1.0, 0, 0.0], [0.0, 0, 1.0], [0.5, 0, 0.5]]

plt.figure(100, figsize=(25, 10))
for p in params:
    resfile = os.path.join('work', net, str(expo) + '-' + str(p) + '-res.txt')
    resdata = np.loadtxt(resfile)
    x = np.arange(resdata.shape[0])
    plt.subplot(1, 2, 1)
    plt.errorbar(x=x, y=resdata[:, 1], yerr=resdata[:, 2] / 6, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Validation Loss')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

    plt.subplot(1, 2, 2)
    plt.errorbar(x=x, y=resdata[:, 3], yerr=resdata[:, 4] / 3, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Test data, relative error')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

plt.subplot(1, 2, 1)
plt.legend(['Regular', 'Sheared', 'Mixed'])
plt.subplot(1, 2, 2)
plt.legend(['Regular', 'Sheared', 'Mixed'])

plt.show()

# Figure 11

val_loss = []
net = 'UNet'
expo = 7
params = [[1.0, 0, 0.0], [0.5, 0, 0.5]]

plt.figure(100, figsize=(25, 10))
for p in params:
    resfile = os.path.join('work', net, str(expo) + '-' + str(p) + '-res.txt')
    resdata = np.loadtxt(resfile)
    x = np.arange(resdata.shape[0])
    plt.subplot(1, 2, 1)
    plt.errorbar(x=x, y=resdata[:, 1], yerr=resdata[:, 2] / 6, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Validation Loss')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

    plt.subplot(1, 2, 2)
    plt.errorbar(x=x, y=resdata[:, 3], yerr=resdata[:, 4] / 3, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Test data, relative error')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

plt.subplot(1, 2, 1)
plt.legend(['Regular', 'Mixed'])
plt.subplot(1, 2, 2)
plt.legend(['Regular', 'Mixed'])

plt.show()
