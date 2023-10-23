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
from ms_export import onnx_export
from model.model import * 
from pruning import *
from utils import * 
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

# def get_model


if __name__ == '__main__':

    devices_intensity = 20
    net = MobileNet(n_class=10)
    model = list(net.modules())
    idx = 1
    for idx in range(len(model)):
        m = model[idx]
        m.old_forward = m.forward
        # print(type(new_forward(m)))
        m.forward = new_forward(m)

    input = torch.randn(1,3,32,32)
    net(input)

    kernel_list = []
    DW = []
    conv = []
    all = []
    idx = 0
    for i in range(len(model)):
        m = model[i]
        if type(m) == nn.Conv2d and m.groups == m.in_channels:
            DW.append([idx,m.flops/m.macs])
            all.append(m.flops/m.macs)
            print("DW",m.flops/m.macs)
            idx+=1
        elif type(m) == nn.Conv2d:
            conv.append([idx,m.flops/m.macs])
            all.append(m.flops/m.macs)
            print("Conv",m.flops/m.macs)
            idx+=1
    layer = range(len(all))
    plt.plot(layer,all,label='32x32')
    a = [i[0] for i in DW]
    b = [i[1] for i in DW] 
    plt.scatter(a,b,marker="x")
    a = [i[0] for i in conv]
    b = [i[1] for i in conv] 
    plt.scatter(a,b,marker="*")

    # net = MobileNet(n_class=10)
    # # ,kernel_list=[5, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3]

    # model = list(net.modules())

    # idx = 1
    # for idx in range(len(model)):
    #     m = model[idx]
    #     m.old_forward = m.forward
    #     # print(type(new_forward(m)))
    #     m.forward = new_forward(m)

    # input = torch.randn(1,3,64,64)
    # net(input)

    # kernel_list = []
    # DW = []
    # conv = []
    # all = []
    # idx = 0
    # kernel_list = []
    # for i in range(len(model)):
    #     m = model[i]
    #     if type(m) == nn.Conv2d and m.groups == m.in_channels:
    #         DW.append([idx,m.flops/m.macs])
    #         all.append(m.flops/m.macs)
    #         print("DW",m.flops/m.macs)

    #         idx+=1
    #     elif type(m) == nn.Conv2d:
    #         conv.append([idx,m.flops/m.macs])
    #         all.append(m.flops/m.macs)
    #         print("Conv",m.flops/m.macs)
    #         idx+=1
    # layer = range(len(all))
    plt.plot(layer,all,label="64x64")
    a = [i[0] for i in DW]
    b = [i[1] for i in DW] 
    plt.scatter(a,b,marker="x")
    a = [i[0] for i in conv]
    b = [i[1] for i in conv] 
    plt.scatter(a,b,marker="*")

    plt.xlabel("layer")
    plt.ylabel('intensity')
    plt.legend()
    plt.savefig('/home/hujie/Desktop/fig/mbv1-kernel.png',dpi=300)




    # print(kernel_list)

    # kernel_list = [3, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3]

    # net = MobileNetV2(n_class=25,kernel_list=kernel_list)
    # # stat(net,(3,224,224))
    # model = list(net.modules())

    # idx = 1
    # for idx in range(len(model)):
    #     m = model[idx]
    #     m.old_forward = m.forward
    #     # print(type(new_forward(m)))
    #     m.forward = new_forward(m)

    # input = torch.randn(1,3,224,224)
    # net(input)

    # kernel_list = []
    # for i in range(len(model)):
    #     m = model[i]
    #     if type(m) == nn.Conv2d and m.groups == m.in_channels:
    #         if m.flops/m.macs > 3:
    #             kernel_list.append(3)
    #         else:
    #             kernel_list.append(0)
    # # print(net)
    # # print(kernel_list)


    # # kernel_list = [3, 5, 3, 5, 3, 5, 3, 3, 3, 3, 3, 5, 3]
    # # print()
    # net = MobileNetV2()
    # print(stat(net,(3,32,32)))

