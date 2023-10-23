import numpy as np
from thop import profile
from thop import clever_format
import torch
from torchstat import stat
import torch.onnx
from summary import summary


def measure_model(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):

    x = torch.randn(1, input_size[0], input_size[1], input_size[2])
    flops, params = profile(model, inputs= (x,))
    model.to(device)
    total_size , macs, params = summary(model,input_size)
    return flops, params, total_size
    