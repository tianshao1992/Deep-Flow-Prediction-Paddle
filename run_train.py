# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/27 20:07
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_train.py
@File ：run_train.py
"""

import os, sys, random
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.optimizer as optim

from Unet_model import UNet2d
from FNO_model import FNO2d
from Trans_model import FourierTransformer2D
import read_data
import utils

######## Settings ########

# number of training iterations
iterations = 100000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0005
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 7
# data set config
# prop = None  # by default, use all from "../data/train"
prop = [10000, 0.5, 0, 0.5]  # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = True
# model type
net = 'FNO'

##########################

for p in ((400, 800, 1600, 3200, 6400, 12800, 25600, 51200)):
    prop[0] = p

    work_path = os.path.join('work', net, str(prop))
    data_path = os.path.join('data')

    prefix = work_path + '-expo-' + str(expo) + "/"
    print("Output prefix: {}".format(prefix))

    dropout = 0.  # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
    doLoad = ""  # optional, path to pre-trained model
    print("net: {}".format(net))
    print("LR: {}".format(lrG))
    print("LR decay: {}".format(decayLr))
    print("Iterations: {}".format(iterations))
    print("Dropout: {}".format(dropout))

    ##########################

    seed = random.randint(0, 2 ** 32 - 1)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # paddle.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic=True # warning, slower

    # create pytorch data object with dfp dataset
    train_path = os.path.join(data_path, 'train/')
    valid_path = os.path.join(data_path, 'test/')
    data = read_data.TurbDataset(prop, shuffle=1, dataDir=train_path, dataDirTest=valid_path)
    trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    print("Training batches: {}".format(len(trainLoader)))
    dataValidation = read_data.ValiDataset(data)
    validLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
    print("Validation batches: {}".format(len(validLoader)))

    # setup training
    epochs = int(iterations / len(trainLoader) + 0.5)
    if 'UNet' in net:
        net_model = UNet2d(channelExponent=expo, dropout=dropout)
    elif 'FNO' in net:
        net_model = FNO2d(in_dim=3, out_dim=3, modes=(32, 32), width=32, depth=4, steps=1, padding=4, activation='gelu')
    elif 'Transformer' in net:
        import yaml

        with open(os.path.join('transformer_config.yml')) as f:
            config = yaml.full_load(f)
        config = config['Transformer']
        net_model = FourierTransformer2D(**config)

    print(net_model)  # print full net
    model_parameters = filter(lambda p: ~p.stop_gradient, net_model.parameters())
    params = sum([np.prod(p.shape) for p in model_parameters])
    print("Initialized TurbNet with {} trainable params ".format(params))

    if len(doLoad) > 0:
        net_model.load_state_dict(paddle.load(doLoad))
        print("Loaded model " + doLoad)

    criterionL1 = nn.L1Loss()
    optimizerG = optim.Adam(parameters=net_model.parameters(), learning_rate=lrG, beta1=0.5, beta2=0.999,
                            weight_decay=0.0)

    ##########################

    for epoch in range(epochs):
        print("Starting epoch {} / {}".format((epoch + 1), epochs))

        net_model.train()
        L1_accum = 0.0
        for i, traindata in enumerate(trainLoader, 0):
            inputs, targets = traindata

            # compute LR decay
            if decayLr:
                currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
                optimizerG.set_lr(currLr)

            optimizerG.clear_grad()
            gen_out = net_model(inputs)

            lossL1 = criterionL1(gen_out, targets)
            lossL1.backward()

            optimizerG.step()

            lossL1viz = lossL1.item()
            L1_accum += lossL1viz

        # validation
        net_model.eval()
        L1val_accum = 0.0
        for i, validata in enumerate(validLoader, 0):
            inputs, targets = validata
            with paddle.no_grad():
                outputs = net_model(inputs)
                lossL1 = criterionL1(outputs, targets)
                L1val_accum += lossL1.item()

        if epoch % 10 == 0:
            input_ndarray = inputs.numpy()[0]
            v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :])) ** 2) ** 0.5

            outputs_denormalized = data.denormalize(outputs.numpy()[0], v_norm)
            targets_denormalized = data.denormalize(targets.numpy()[0], v_norm)
            utils.makeDirs([prefix + "results_train"])
            utils.imageOut(prefix + "results_train/epoch{}_{}".format(epoch, i), outputs_denormalized,
                           targets_denormalized,
                           saveTargets=True)

        # data for graph plotting
        L1_accum /= len(trainLoader)
        L1val_accum /= len(validLoader)
        if saveL1:
            if epoch == 0:
                utils.resetLog(prefix + "L1.txt")
                utils.resetLog(prefix + "L1val.txt")
            utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
            utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)

        logline = "Epoch: {}, batch-idx: {}, L1: {}, L1_val {}\n".format(epoch, i, L1_accum, L1val_accum)
        print(logline)

    paddle.save(net_model.state_dict(), prefix + "net_model")
