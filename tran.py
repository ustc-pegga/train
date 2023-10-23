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
# from ms_export import onnx_export
from model.model import * 
from pruning import *

def tran(net1,net2):



    # ckpt_path = "/home/hujie/code/motivation/checkpoint/mobilenet_k3_imagenet25_300-run7/ckpt.best.pth.tar"
    # # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run81/ckpt.best.pth.tar"
    # # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run88/ckpt.best.pth.tar"
    # if ckpt_path is not None:  # assigned checkpoint path to resume from
    #     print('=> Resuming from checkpoint..')
    #     checkpoint = torch.load(ckpt_path)
    #     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    #     new_dict = {k: v for k, v in sd.items() if net1.state_dict()[k].numel() == v.numel()}
    #     # sd.update(new_dict)
    #     missing_keys, unexpected_keys = net1.load_state_dict(new_dict, strict=False)

    model1 = list(net1.modules())
    model2 = list(net2.modules())

    idx = 1
    for i in range(len(model1)):
        if type(model1[i]) == nn.Conv2d and model1[i].groups == model1[i].in_channels and model1[i].kernel_size[0] == model2[i].kernel_size[0]:
            model2[i].weight.data = model1[i].weight.data
            print(idx)
            idx += 1
            # # model2[i].weight.data = np.zeros(model2[i].weight.data.shape)
            # print(model1[i].weight.data.shape)
        elif type(model1[i]) == nn.Conv2d and model1[i].groups == model1[i].in_channels and model1[i].kernel_size[0] < model2[i].kernel_size[0]:
            padding = (model2[i].kernel_size[0]-model1[i].kernel_size[0])//2
            model2[i].weight.data = F.pad(model1[i].weight.data, (padding,padding,padding,padding),'constant', 0)
            idx += 1
        elif type(model1[i]) in [nn.Conv2d,nn.Linear]:
            model2[i].weight = model1[i].weight
            idx += 1
        elif type(model1[i]) == nn.BatchNorm2d:
            model2[i].weight = model1[i].weight
            model2[i].bias = model1[i].bias
            model2[i].running_mean = model1[i].running_mean
            model2[i].running_var = model1[i].running_var
            model2[i].num_batches_tracked = model1[i].num_batches_tracked



    # print(idx)
    # for i,v in net1.state_dict().items():
    #     print(i,v)
    return net2

# ckpt_path = "/home/hujie/code/motivation/checkpoint/mobilenet_k3_imagenet25_300-run7/ckpt.best.pth.tar"
# # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run81/ckpt.best.pth.tar"
# # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run88/ckpt.best.pth.tar"
# if ckpt_path is not None:  # assigned checkpoint path to resume from
#     print('=> Resuming from checkpoint..')
#     checkpoint = torch.load(ckpt_path)
#     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#     new_dict = {k: v for k, v in sd.items() if net1.state_dict()[k].numel() == v.numel()}
#     # sd.update(new_dict)
#     missing_keys, unexpected_keys = net1.load_state_dict(new_dict, strict=False)

# kernel_list = []
# net = MobileNet()
# tran()