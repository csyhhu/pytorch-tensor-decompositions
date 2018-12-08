"""
This code decomposes CIFAR9 model using Tucker, and implement dual path
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import numpy as np
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain

from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from models_CIFAR9STL9.CIFARNet import CIFARNet
from utils.dataset import get_dataloader
from utils.miscellaneous import progress_bar
from utils.train import validate


def L21decompose_fc()

# --------------------------------------
use_cp = True
decompose_fc = True
# --------------------------------------

pretrain_param = torch.load('./checkpoint/CIFARNet9_CIFAR9.pth')
model = CIFARNet()
model.load_state_dict(pretrain_param)
model.eval()
model.cpu()

ranks_dict = {
    '0': [8, 8],
    '3': [22, 22],
    '6': [48, 48]
}

'''
N = len(model.features._modules.keys())
for i, key in enumerate(model.features._modules.keys()):

    # if i >= N - 2:
    #     break
    if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
        conv_layer = model.features._modules[key]
        if use_cp:
            rank = max(conv_layer.weight.data.numpy().shape)//2
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)
        else:
            decomposed = tucker_decomposition_conv_layer(conv_layer, None)

        model.features._modules[key] = decomposed
'''

for i, key in enumerate(model.classifier._modules.keys()):

    if isinstance(model.classifier._modules[key], nn.Linear):
        fc_linear = model.classifier._modules[key]


# torch.save(model, './checkpoint/%s_CIFARNet9.p' %('cp' if use_cp else 'tucker'))
test_loader = get_dataloader('CIFAR9', 'test', 128)
model.cuda()
validate(model, test_loader)