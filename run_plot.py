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

net = 'UNet'
expo = 6
params = [[1.0, 0, 0.0], [0.0, 0, 1.0], [0.5, 0, 0.5]]

plt.figure(100, figsize=(15, 25))
for p in params:
    resfile = os.path.join('work', net, str(expo) + '-' + str(p) + '-res.txt')
    resdata = np.loadtxt(resfile)
    x = np.arange(resdata.shape[0])
    plt.subplot(2, 1, 1)
    plt.errorbar(x=x, y=resdata[:, 1], yerr=resdata[:, 2] / 6, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Validation Loss')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

    plt.subplot(2, 1, 2)
    plt.errorbar(x=x, y=resdata[:, 3], yerr=resdata[:, 4] / 3, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Test data, relative error')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

plt.subplot(2, 1, 1)
plt.legend(['Regular', 'Sheared', 'Mixed'])
plt.subplot(2, 1, 2)
plt.legend(['Regular', 'Sheared', 'Mixed'])
plt.savefig('work/Fig10.jpg')
plt.show()

# Figure 11

net = 'UNet'
expo = 7
params = [[1.0, 0, 0.0], [0.5, 0, 0.5]]

plt.figure(100, figsize=(15, 25))
for p in params:
    resfile = os.path.join('work', net, str(expo) + '-' + str(p) + '-res.txt')
    resdata = np.loadtxt(resfile)
    x = np.arange(resdata.shape[0])
    plt.subplot(2, 1, 1)
    plt.errorbar(x=x, y=resdata[:, 1], yerr=resdata[:, 2] / 6, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Validation Loss')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

    plt.subplot(2, 1, 2)
    plt.errorbar(x=x, y=resdata[:, 3], yerr=resdata[:, 4] / 3, linewidth=3.0, elinewidth=10.0)
    plt.ylabel('Test data, relative error')
    plt.xlabel('Trainging data (regular)')
    plt.grid('on')
    x_labels = [str(int(res)) for res in resdata[:, 0]]
    plt.xticks(x, x_labels, rotation=40)

plt.subplot(2, 1, 1)
plt.legend(['Regular', 'Mixed'])
plt.subplot(2, 1, 2)
plt.legend(['Regular', 'Mixed'])
plt.savefig('work/Fig11.jpg')
plt.show()


# 绘制条形图

nets = ['UNet', 'FNO', 'Transformer']
expo = [7, 6, 6]

resdata = []
for p, e in zip(nets, expo):
    resfile = os.path.join('work', str(p), str(e) + '-' + str([1.0, 0, 0.0]) + '-res.txt')
    resdata.append(np.loadtxt(resfile))

resdata[0] = resdata[0][(0, 3, 6), :]
resdata = np.array(resdata)

fig, ax = plt.subplots(1, 2, num=100, figsize=(16, 8))
index = np.arange(len(nets))
bar_width = 0.3

opacity = 0.6
error_config = {'ecolor': '0.3', 'elinewidth': 5.0}

for i in range(len(nets)):
    rects = ax[0].bar(index+i*bar_width, resdata[i, :, 1], bar_width,
                    alpha=opacity, yerr=resdata[0, :, 2],
                    error_kw=error_config,
                    label=nets[i])

ax[0].set_xlabel('Trainging data (regular)')
ax[0].set_ylabel('Validation Loss')
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(('400', '3200', '25600'))
ax[0].legend()

for i in range(len(nets)):
    rects = ax[1].bar(index+i*bar_width, resdata[i, :, 3], bar_width,
                    alpha=opacity, yerr=resdata[0, :, 4],
                    error_kw=error_config,
                    label=nets[i])

ax[1].set_xlabel('Trainging data (regular)')
ax[1].set_ylabel('Test data, relative error')
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(('400', '3200', '25600'))
ax[1].legend()
plt.savefig('work/comparison.jpg')
fig.tight_layout()
plt.show()
