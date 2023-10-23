
from torchstat import stat
from torchsummary import summary
import os
import sys
import csv
import random
from model.op import * 
from measure import measure_model
import argparse


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
from model.model import *
from pruning import *
# from ms_export import onnx_export
# from tran import tran
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--kernel', default='3', type=int, help='kernel_size')
    parser.add_argument('--kernel_list', default=None, type=str, help='kernel_size')
    parser.add_argument('--rate', default=None, type=str, help='pruning')
    parser.add_argument('--name', default="Origin", type=str, help='pruning')
    parser.add_argument('--type', default=None, type=str, help='pruning')
    parser.add_argument('--data_root', default=None, type=str, help='root')
    return parser.parse_args()


def get_model(model,datasets,kernel,kernel_list):
    if datasets == 'cifar100':
        n_class = 100
    elif datasets == 'cifar10':
        n_class = 10
    else :#tiny-imagenet
        n_class = 200
    if not kernel_list:
        print('=> Building model..')
        if model == 'mobilenetv1':
            net = MobileNet(n_class=n_class)
        elif model == 'mobilenetv2':
            net = MobileNetV2(n_class=n_class)
        elif model == 'resnet18':# resnet18
            net = resnet18(n_class=n_class)
        else:
            net =None
        return net
    else:
        print('=> Building model..')
        if model == 'mobilenetv1':
            net = MobileNet(n_class=n_class,kernel_list=kernel_list)
        elif model == 'mobilenetv2':
            net = MobileNetV2(n_class=n_class,kernel_list=kernel_list)
        elif model == 'resnet18':# resnet18
            net = resnet18(n_class=n_class)
        else:
            net =None
        return net

def get_input(dataset):
    if "cifar" in dataset:
        return (32,32)
    elif dataset == 'imagenet':
        return (224,224)
    else:
        return(64,64)

def get_input2(dataset):
    if "cifar" in dataset:
        return torch.randn(1,3,32,32)
    elif dataset == 'imagenet':
        return torch.randn(1,3,224,224)
    else:
        return torch.randn(1,3,64,64)

def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    model = args.model
    dataset = args.dataset
    kernel = args.kernel # args.kernel
    kernel_list = args.kernel_list
    # data_root = "/datasets/tiny-imagenet-200"
    data_root = args.data_root
    print('=> Preparing data..')

    if kernel_list != None:
        kernel_list = [int(i) for i in kernel_list.split(",")]
    net = get_model(model,dataset,kernel,kernel_list = kernel_list)

    if args.rate != None:
        rate = [float(i) for i in args.rate.split(",")]
        input = get_input(dataset)
        if model == 'mobilenet' or model == 'mobilenetv1':

            net = mbv1_pruning(net.to('cpu'),rate,input=input)
        elif model == 'mobilenetv2':
            net = mbv2_pruning(net.to('cpu'),rate,input=input)
        else:
            pass
    export_path = os.path.join(data_root,args.type)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    input = get_input2(dataset)
    print(export_path)
    # onnx_export(net, input ,export_path, '{}_{}_{}'.format(model,dataset,args.name))
    flops,macs,params = measure_model(net,(3,32,32))
    print(model,dataset,args.name,flops/(1024*1024),params/(1024*1024),macs/(1024*1024))
    # print(net)


