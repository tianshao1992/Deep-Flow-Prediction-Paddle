# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/28 0:12
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_statistics.py
@File ：run_statistics.py
"""

import os, sys, random, math
import numpy as np

import utils
from utils import log

dropout = 0.0
prop = [10000, 1.0, 0, 0.0]
net = 'FNO'
expo = 6
# statistics number
sta_number = 7

statistics_res = []

# for p in ((100, 200, 400, 1600, 3200, 6400, 12800)):  #for fig.10
# for p in ((400, 800, 1600, 3200, 6400, 12800, 25600, 51200)):  #for fig.11
for p in ((400, 3200, 25600)):  # for Transformer and FNO
    try:
        prop[0] = p
        test_results = os.path.join('work', net, "prop-" + str(prop), "expo-" + str(expo)) + "/testout.txt"
        with open(test_results, 'r') as f:
            lines = f.readlines()

            last_lines = lines[-1]

        res_str = last_lines.split(':')[-1].split(',')
        test_avg = float(res_str[0])
        test_std = float(res_str[1])

        valid_loss = []
        for s in range(sta_number):
            work_path = os.path.join('work', net, "prop-" + str(prop), "expo-" + str(expo), "statistics-" + str(s + 1),
                                     'L1val.txt')
            valid_loss.append(np.loadtxt(work_path)[-1])

        valid_avg = np.mean(valid_loss)
        valid_std = np.std(valid_loss)

        statistics_res.append([p, valid_avg, valid_std, test_avg, test_std])
    except:
        pass

statistics_res = np.array(statistics_res)
np.savetxt(os.path.join('work', net, str(expo) + "-" + str(prop[1:]) + '-res.txt'), statistics_res)
