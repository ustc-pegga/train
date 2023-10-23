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
from my_pruning_test import width_prune 
# from tran import tran
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--kernel', default='3', type=int, help='kernel_size')
    parser.add_argument('--kernel_list', default=None, type=str, help='kernel_size')
    parser.add_argument('--rate', default=None, type=str, help='pruning')
    parser.add_argument('--name', default=None, type=str, help='pruning')
    parser.add_argument('--width_rate',default=None,type=float)
    parser.add_argument('--trans',default=False)
    return parser.parse_args()


def get_model(model,datasets,kernel,kernel_list,undersample):
    if datasets == 'cifar100':
        n_class = 100
    elif datasets == 'cifar10':
        n_class = 10
    elif datasets == 'imagenet-100':
        n_class =100
    elif datasets == 'imagenet':
        n_class = 1000
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


def train(epoch, train_loader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    num_idx = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer,epoch,batch_idx,num_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    num_idx = len(train_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            lr = adjust_learning_rate(optimizer,epoch,batch_idx,num_idx)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)
    return epoch, best_acc,loss


def adjust_learning_rate(optimizer, epoch,step,num_iter):
    lr_type = "cos"
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch-10) / (n_epoch-10)))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = lr
    else:
        raise NotImplementedError
    if epoch < 10:
            lr = lr * float(1 + step + epoch * num_iter) / (10. * num_iter)
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_input(dataset):
    if "cifar" in dataset:
        return (32,32)
    elif dataset == 'imagenet' or dataset == 'imagenet-100':
        return (224,224)
    else:
        return(64,64)

def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()
    # gpus = [0, 1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[1]))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gc.collect()
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    n_workers = args.n_worker
    kernel = args.kernel # args.kernel
    kernel_list = args.kernel_list
    # data_root = "/datasets/tiny-imagenet-200"
    data_root = args.data_root
    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(dataset, batch_size, n_workers,
                                                    data_root)
    print(len(train_loader))

    if kernel_list != None:
        kernel_list = [int(i) for i in kernel_list.split(",")]
        print(kernel_list)
    # kernel_list = [5, 7, 5, 9, 5, 5, 3, 3, 3, 3, 3, 3, 3]
    # net2 = get_model(model,dataset,kernel,kernel_list)
    undersample = True
    # if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'tiny-imagenet':
    #     undersample = False
    # else:
    #     undersample = True
    if args.trans:
        net = get_model(model,dataset,kernel,kernel_list=None,undersample=undersample)
    else:
        net = get_model(model,dataset,kernel,kernel_list=kernel_list,undersample=undersample)
    # kernel_list = [3, 5, 3, 5, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3]
    # net = MobileNetV2(n_class=25,kernel_list=kernel_list)
    ckpt_path = args.ckpt_path

    if ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(ckpt_path,map_location=torch.device('cuda'))
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = {k: v for k, v in sd.items() if net.state_dict()[k].numel() == v.numel()}

        missing_keys, unexpected_keys = net.load_state_dict(new_dict, strict=False)
        # for param in net.features.parameters():
        #     param.requires_grad = False
    # rate = [1.0, 0.5, 0.75, 0.5833333333333334, 0.5, 0.5, 0.6666666666666666, 0.5277777777777778, 0.75, 0.5625, 0.75, 0.5625, 0.75, 0.625, 0.625, 0.65625, 0.75, 0.6666666666666666, 0.75, 0.59375, 0.75, 0.6979166666666666, 0.75, 0.6875, 0.875, 0.7638888888888888, 0.7916666666666666, 0.75, 0.775, 0.6958333333333333, 0.775, 0.6916666666666667, 0.75, 0.7125, 0.65, 0.553125]
    # net = mbv2_pruning(net,rate,input=(224,224))
    # net = tran()
    # net = MobileNet(n_class=25,kernel_list=kernel_list)
    # rate = [1.0, 0.5, 0.625, 0.625, 0.65625, 0.640625, 0.65625, 0.6640625, 0.671875, 0.671875, 0.671875, 0.671875, 0.6640625, 0.421875, 0.203125]
    # net = mbv1_pruning(net,rate,input=(224,224))
    if kernel_list:
        net2 = get_model(model,dataset,kernel,kernel_list=kernel_list,undersample=undersample)
        net = tran(net,net2)
        # print(net)
    if args.rate != None:
        rate = [float(i) for i in args.rate.split(",")]
        input = get_input(dataset)
        if model == 'mobilenet' or model == 'mobilenetv1':

            net = mbv1_pruning(net.to('cpu'),rate,input=input)
        elif model == 'mobilenetv2':
            net = mbv2_pruning(net.to('cpu'),rate,input=input)
        else:
            pass

    
    # net = nn.DataParallel(net,device_ids=[0,1],output_device=0)
    # net.to("cuda:0")
    net.to('cpu')
    if args.width_rate:
        net = width_prune(net,rate=args.width_rate,n_class=100)
    # print(net)
    
    FLOPs ,p,m = measure_model(net,(3,224,224))
    print(FLOPs)
    net.cuda()
    wd = 4e-5
    lr = 0.1
    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # optimizer = optim.Adam(net.parameters(),lr=0.1)
    eval = args.eval

    if eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(model, dataset))
        log_dir = get_output_folder('./checkpoint_1', '{}_{}_{}_{}'.format(model, dataset,n_epoch,args.name))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)
        # img = torch.rand([1, 3, 64, 64], dtype=torch.float32)
        # writer.add_graph(net, input_to_model=img)  # 类似于TensorFlow 1.x 中的fed
        acc = dict()
        f = open('{}/acc.txt'.format(log_dir), 'w')
        for epoch in range(start_epoch, start_epoch + n_epoch):
            # lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader,optimizer)
            epoch, best_acc, loss = test(epoch, val_loader)
            acc[epoch] = best_acc
            print(epoch, best_acc)
            f.write('epoch: {}, best_acc: {}, loss: {}\n'.format(epoch, best_acc,loss))

        

        writer.close()
        # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M, best top-1 acc: {}%'.format(n_params / 1e6, n_flops / 1e6, best_acc))

