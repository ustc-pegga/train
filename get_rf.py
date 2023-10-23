import torch.nn as nn
import torch
import random
from measure import measure_model
from torchstat import stat
from torchsummary import summary
import os
import sys
import csv
import random
from model.op import * 
from measure import measure_model
from model.model import * 
from pruning import *
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

def get_intensity(layer,kernel):
    in_h = layer.in_h
    in_w = layer.in_h
    out_h = int((in_h + 2 * layer.padding[0] - kernel) /
                layer.stride[0] + 1)
    out_w = int((in_h + 2 * layer.padding[1] - kernel) /
                layer.stride[1] + 1)
    flops = layer.in_channels * layer.out_channels * kernel *  \
                kernel * out_h * out_w / layer.groups * 1
    params = kernel * kernel * layer.in_channels * layer.out_channels / layer.groups
    macs  = layer.in_channels * in_h * in_w + layer.out_channels * out_h * out_w + \
        params
    params *= 4
    macs *= 4

    flops += 3 * out_h * out_h * layer.out_channels
    return float(flops/macs)

def get_macs(layer,kernel):
    in_h = layer.in_h
    in_w = layer.in_h
    out_h = int((in_h + 2 * layer.padding[0] - kernel) /
                layer.stride[0] + 1)
    out_w = int((in_h + 2 * layer.padding[1] - kernel) /
                layer.stride[1] + 1)
    flops = layer.in_channels * layer.out_channels * kernel *  \
                kernel * out_h * out_w / layer.groups * 1
    params = kernel * kernel * layer.in_channels * layer.out_channels / layer.groups
    macs  = layer.in_channels * in_h * in_w + layer.out_channels * out_h * out_w + \
        params
    macs *= 4

    return macs

if __name__ == '__main__':

    input_size = 32
    n_class = 10
    net = MobileNetV2(n_class=n_class,input_size=32,undersample=True,kernel_size=11)
    model = list(net.modules())
    idx = 1
    for idx in range(len(model)):
        m = model[idx]
        m.old_forward = m.forward
        # print(type(new_forward(m)))
        m.forward = new_forward(m)

    # stat(net,(3,32,32))
    input = torch.randn(1,3,input_size,input_size)
    net(input)
    kernel_list = []
    DW = []
    conv = []
    all = []
    idx = 0
    s=1
    rf=1
    for i in range(len(model)):
        m = model[i]
        if type(m) == nn.Conv2d or type(m)==nn.AvgPool2d:
            print(s)
            rf+=(m.kernel_size[0]-1)*s
            s*=m.stride[0]
        else:
            pass
    print(rf)
