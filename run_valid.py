# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/28 0:12
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_valid.py
@File ：run_valid.py
"""

import os, sys, random, math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

from read_data import TurbDataset
from net_model import TurbNetG
import utils
from utils import log

suffix = ""  # customize loading & output if necessary
prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

expo = 5
data_path = os.path.join('H:\\', 'PythonProject', 'Deep-Flow-Prediction', 'data')
train_path = os.path.join(data_path, 'train\\')
valid_path = os.path.join(data_path, 'test\\')
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDir=train_path, dataDirTest=valid_path)
# dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDirTest="../data/test/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

netG = TurbNetG(channelExponent=expo)
lf = "./" + prefix + "testout{}.txt".format(suffix)
utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.
losses = []
models = []

for si in range(25):
    s = chr(96 + si)
    if (si == 0):
        s = ""  # check modelG, and modelG + char
    modelFn = "./" + prefix + "modelG{}{}".format(suffix, s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn)
    netG.set_state_dict(paddle.load(modelFn))
    log(lf, "Loaded " + modelFn)
    # netG.cuda()

    criterionL1 = nn.L1Loss()
    # criterionL1.cuda()
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_p_accum = 0
    lossPer_v_accum = 0
    lossPer_accum = 0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs, targets = data

        outputs = netG(inputs)
        outputs = outputs[0]
        targets = targets[0]

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        outputs = np.array(outputs)
        targets = np.array(targets)

        # precentage loss by ratio of means which is same as the ratio of the sum
        lossPer_p = np.sum(np.abs(outputs[0] - targets[0])) / np.sum(np.abs(targets[0]))
        lossPer_v = (np.sum(np.abs(outputs[1] - targets[1])) + np.sum(np.abs(outputs[2] - targets[2]))) \
                    / (np.sum(np.abs(targets[1])) + np.sum(np.abs(targets[2])))
        lossPer = np.sum(np.abs(outputs - targets)) / np.sum(np.abs(targets))
        lossPer_p_accum += lossPer_p.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_accum += lossPer.item()

        log(lf, "Test sample %d" % i)
        log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs[0] - targets[0])),
                                                                     lossPer_p.item()))
        log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs[1] - targets[1])) +
                                                                     np.sum(np.abs(outputs[2] - targets[2])),
                                                                     lossPer_v.item()))
        log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs - targets)),
                                                                     lossPer.item()))

        # Calculate the norm
        input_ndarray = inputs.numpy()[0]
        v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :])) ** 2) ** 0.5

        outputs_denormalized = dataset.denormalize(outputs, v_norm)
        targets_denormalized = dataset.denormalize(targets, v_norm)

        # denormalized error
        outputs_denormalized_comp = np.array([outputs_denormalized])
        # outputs_denormalized_comp=torch.from_numpy(outputs_denormalized_comp)
        targets_denormalized_comp = np.array([targets_denormalized])
        # targets_denormalized_comp=torch.from_numpy(targets_denormalized_comp)

        # targets_denormalized_comp, outputs_denormalized_comp = targets_denormalized_comp.float().cuda(), outputs_denormalized_comp.float().cuda()

        # outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
        # targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

        outputs_dn = paddle.to_tensor(outputs_denormalized_comp)
        targets_dn = paddle.to_tensor(targets_denormalized_comp)

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()

        # write output image, note - this is currently overwritten for multiple models
        os.chdir("./results_test/")
        utils.imageOut("%04d" % (i), outputs, targets, normalize=False, saveMontage=True)  # write normalized with error
        os.chdir("../")

    log(lf, "\n")
    L1val_accum /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    lossPer_v_accum /= len(testLoader)
    lossPer_accum /= len(testLoader)
    L1val_dn_accum /= len(testLoader)
    log(lf, "Loss percentage (p, v, combined): %f %%    %f %%    %f %% " %
        (lossPer_p_accum * 100, lossPer_v_accum * 100, lossPer_accum * 100))
    log(lf, "L1 error: %f" % (L1val_accum))
    log(lf, "Denormalized error: %f" % (L1val_dn_accum))
    log(lf, "\n")

    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

if len(losses) > 1:
    avgLoss /= len(losses)
    lossStdErr = np.std(losses) / math.sqrt(len(losses))
    log(lf, "Averaged relative error and std dev across models:   %f , %f " % (avgLoss, lossStdErr))
