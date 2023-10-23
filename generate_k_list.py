import os
import time
import argparse
import shutil
import math
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import MobileNet
from tensorboardX import SummaryWriter
import json
from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model
from model.model import *
from pruning import *
from tran import tran 
from torch_receptive_field import receptive_field, receptive_field_for_unit

from lib.utils import * 
import matplotlib.pyplot as plt


idx = 1

def new_forward(m):
    def lambda_forward(x):
        m.input_feat = x.clone()
        measure_layer_for_pruning(m, x)
        y = m.old_forward(x)
        m.output_feat = y.clone()
        return y

    return lambda_forward

use_cuda=1

def get_model(model,datasets,kernel,kernel_list,undersample):
    if datasets == 'cifar100':
        n_class = 100
    elif datasets == 'cifar10':
        n_class = 10
    elif datasets == 'imagenet-100':
        n_class =100
    else :#tiny-imagenet
        n_class = 200
    if not kernel_list:
        print('=> Building model..')
        if model == 'mobilenetv1':
            net = MobileNet(n_class=n_class,kernel_size=kernel,undersample=undersample)
        elif model == 'mobilenetv2':
            net = MobileNetV2(n_class=n_class,kernel_size=kernel,undersample=undersample)
        elif model == 'resnet18':# resnet18
            net = resnet18(n_class=n_class)
        else:
            net =None
        return net.cuda() if use_cuda else net
    else:
        print('=> Building model..')
        if model == 'mobilenetv1':
            net = MobileNet(n_class=n_class,kernel_list=kernel_list,undersample=undersample)
        elif model == 'mobilenetv2':
            net = MobileNetV2(n_class=n_class,kernel_list=kernel_list,undersample=undersample)
        elif model == 'resnet18':# resnet18
            net = resnet18(n_class=n_class)
        else:
            net =None
        return net.cuda() if use_cuda else net

dataset = 'cifar10'
model = 'mobilenetv2'
kernel = 3
kernel_list =None
undersample=False

net = get_model(model,dataset,kernel,kernel_list = kernel_list,undersample=undersample)

receptive_field_dict = receptive_field(net, (3, 32, 32))

