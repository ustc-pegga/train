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
from model.model import *


# def measure_inference_time(net, input, repeat=100):
#    # torch.cuda.synchronize()   # if use cuda uncomment it
#     start = time.perf_counter()
#     for _ in range(repeat):
#         model(input)
#         #torch.cuda.synchronize() # if use cuda uncomment it
#     end = time.perf_counter()
#     return (end-start) / repeat

# def prune_model(model, round_to=1):
#     model.cpu()
#     DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
#     def prune_conv(conv, amount=0.2, round_to=1):
#         #weight = conv.weight.detach().cpu().numpy()
#         #out_channels = weight.shape[0]
#         #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
#         #num_pruned = int(out_channels * pruned_prob)
#         #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
#         strategy = tp.strategy.L1Strategy()
#         pruning_index = strategy(conv.weight, amount=amount, round_to=round_to)
#         plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
#         plan.exec()
    
#     block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
#     blk_id = 0
#     for m in model.modules():
#         if isinstance( m, BasicBlock ):
#             prune_conv( m.conv1, block_prune_probs[blk_id], round_to )
#             prune_conv( m.conv2, block_prune_probs[blk_id], round_to )
#             blk_id+=1
#     return model 
 
device = torch.device('cpu')  #torch.device('cuda') # or torch.device('cpu')


# before pruning
cfg = [3, 32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
rate = [1.0, 0.75, 0.75, 0.6875, 0.75, 0.71875, 0.71875, 0.703125, 0.703125, 0.6875, 0.65625, 0.640625, 0.765625, 0.7890625, 0.5]
model = MobileNet(n_class=25)


def mbv1_pruning(model,rate,input):
    m_list = list(model.modules())
    idx = 1
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, input[0], input[1]) )
    for m in m_list:
        if isinstance(m,nn.Sequential):
            if(len(m)==3):
                # print(m[0].groups)
                # strategy = tp.strategy.L1Strategy()
                # DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 224, 224) )
                # pruning_index = strategy(m[1].weight, amount=0.2, round_to=4)
                weight = m[0].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                idx+=1
                print(num_pruned)
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                print(pruning_index)
                plan = DG.get_pruning_plan(m[0], tp.prune_conv_out_channels, pruning_index)
                plan.exec()
            elif len(m)==6:
                weight = m[3].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                idx+=1
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(m[3], tp.prune_conv_out_channels, pruning_index)
                plan.exec()
    return model



def mbv2_pruning(model,rate,input):
    m_list = list(model.modules())
    idx = 1
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, input[0], input[1]) )
    for m in m_list:
        if isinstance(m,nn.Sequential):
            if(len(m)==3):# first 
                # print(m[0].groups)
                # strategy = tp.strategy.L1Strategy()
                # DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 224, 224) )
                # pruning_index = strategy(m[1].weight, amount=0.2, round_to=4)
                weight = m[0].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                num_pruned = num_pruned & ~3
                if num_pruned == 0:
                    continue
                idx+=1
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(m[0], tp.prune_conv_out_channels, pruning_index)
                plan.exec()
            elif len(m)==5:# first invert
                weight = m[3].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                num_pruned = num_pruned & ~3
                idx+=1
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(m[3], tp.prune_conv_out_channels, pruning_index)
                plan.exec()     
            elif len(m)==8:
                weight = m[0].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                num_pruned = num_pruned & ~3
                idx+=1
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(m[3], tp.prune_conv_out_channels, pruning_index)
                plan.exec()

                weight = m[6].weight.detach().cpu().numpy()
                out_channels = weight.shape[0]
                L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
                num_pruned = int(out_channels * (1-rate[idx]))
                num_pruned = num_pruned & ~3
                idx+=1
                pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(m[3], tp.prune_conv_out_channels, pruning_index)
                plan.exec()
    return model

# model = MobileNetV2(n_class=25)
# m_list = list(model.modules())
# rate =  [1.0, 0.625, 0.75, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6944444444444444, 0.75, 0.7083333333333334, 0.75, 0.6666666666666666, 0.75, 0.6875, 0.875, 0.7083333333333334, 0.875, 0.5833333333333334, 0.875, 0.7395833333333334, 0.875, 0.6770833333333334, 0.875, 0.6319444444444444, 0.875, 0.5833333333333334, 0.875, 0.7083333333333334, 0.85, 0.6291666666666667, 0.75, 0.625, 0.875, 0.7208333333333333, 0.4375, 0.2]
# net = mbv2_pruning(model,rate,[224,224])
# print(net)

