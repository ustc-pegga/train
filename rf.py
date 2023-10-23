import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os
from model.model import *
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
use_cuda=1
#这里随机拿100张测试集图像，放到一个文件夹中，img_dir是文件夹路径
img_dir = "./cifar10_test_images"
images=os.listdir(img_dir)
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


model = get_model('mobilenetv2','cifar10',3,kernel_list = None,undersample=False)
print(model)
model = model.eval()
#定义输入图像的长宽，这里需要保证每张图像都要相同
input_H, input_W = 56,56
#生成一个和输入图像大小相同的0矩阵，用于更新梯度
heatmap = np.zeros([input_H, input_W])
#打印一下模型，选择其中的一个层
print(model)

#这里选择骨干网络的最后一个模块
layer = model.get_last_conv_layer
print(layer)
def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
    
#为了简单，这里直接一张一张图来算，遍历文件夹中所有图像  
for img in images:
    read_img = os.path.join(img_dir,img)
    image = Image.open(read_img)
    
    #图像预处理，将图像缩放到固定分辨率，并进行标准化
    image = image.resize((input_H, input_W))
    image = np.float32(image) / 255
    input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])(image)
    
    #添加batch维度
    input_tensor = input_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
    #输入张量需要计算梯度
    input_tensor.requires_grad = True
    fmap_block = list()
    input_block = list()
    
    #对指定层获取特征图
    layer.register_forward_hook(farward_hook)
    
    #进行一次正向传播
    output = model(input_tensor)
    
    #特征图的channel维度算均值且去掉batch维度，得到二维张量
    feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()
    
    #对二维张量中心点（标量）进行backwar
    print(feature_map)
    feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

    #对输入层的梯度求绝对值
    grad = torch.abs(input_tensor.grad)
    
    #梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
    grad = grad.mean(dim=1,keepdim=False).squeeze()
    
    #累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
    heatmap = heatmap + grad.cpu().numpy()
    
    
cam = heatmap

#对累加的梯度进行归一化
cam = cam / cam.max()

#可视化，蓝色值小，红色值大
cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
Image.fromarray(cam)