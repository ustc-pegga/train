import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np 
import time
from torchstat import stat
from torchsummary import summary
from model.model import MobileNetV2
from lib.net_measure import measure_model
import os

def width_prune(model,rate,n_class,step=8):
    example_inputs = torch.randn(1, 3, 224, 224)

    # 1. Importance criterion
    imp = tp.importance.MagnitudeImportance(p=1) # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == n_class:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=rate, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # ch_sparsity_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized sparsity for layers or blocks
        ignored_layers=ignored_layers,
        round_to = step
    )
    pruner.step()
    return model

def global_prune(model,global_rate,n_class,step=8):
    example_inputs = torch.randn(1, 3, 224, 224)

    # 1. Importance criterion
    imp = tp.importance.MagnitudeImportance(p=1) # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == n_class:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=global_rate, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # ch_sparsity_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized sparsity for layers or blocks
        ignored_layers=ignored_layers,
        global_pruning=True,
        round_to = step,
    )
    pruner.step()
    return model

if __name__ == '__main__':
    net = MobileNetV2(n_class=100,kernel_list=[3, 5, 7, 9, 3, 7, 3, 9, 5, 9, 7, 3, 9, 9, 3, 3, 3])
    ckpt_path = '/home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_300_origin-run8/ckpt.best.pth.tar'

    if ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(ckpt_path,map_location=torch.device('cuda'))
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = {k: v for k, v in sd.items() if net.state_dict()[k].numel() == v.numel()}

        missing_keys, unexpected_keys = net.load_state_dict(new_dict, strict=False)
    model = width_prune(net,0.35,100)
    print(model)
    FLOPs,p,m = measure_model(model,(3,224,224))
    print((FLOPs / 1024 /1024))