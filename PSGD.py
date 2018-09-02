import os
import shutil
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

def coordinate(rank, world_size):
    output = open("DPSGD_output.txt", "w")
    #print('Start coordinate  Total: %3d'%(world_size))
    args = parser.parse_args()
    model = resnet20()
    #model = alexnet()
    model = model.cuda()
    model_flat = flatten_all(model)
    dist.broadcast(model_flat, world_size)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = datasets.CIFAR10(root='./data', train=True,download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,pin_memory=True,shuffle=False, num_workers=2)

    valset = datasets.CIFAR10(root='./data', train=False,download=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100,pin_memory=True,shuffle=False, num_workers=2)
   
    time_cost = 0
    for epoch in range(args.epochs):
        dist.barrier()
        t1 = time.time()
        dist.barrier()
        t2 = time.time() 
        time_cost += t2 - t1
        model_flat.zero_()
        loss = torch.FloatTensor([0])
        dist.reduce(loss, world_size, op=dist.reduce_op.SUM)
        loss.div_(world_size)
        dist.reduce(model_flat, world_size, op=dist.reduce_op.SUM)
        model_flat.div_(world_size)
        unflatten_all(model, model_flat)
        # evaluate on validation set
        _ ,prec1 = validate(val_loader, model, criterion)
        output.write('%d %3f %3f %3f\n'%(epoch,time_cost,loss.item(),prec1))
        output.flush()
    
    output.close()

def run(rank, world_size):
    print('Start node: %d  Total: %3d'%(rank,world_size))
    args = parser.parse_args()
    current_lr = args.lr
    adjust = [80,120]


    model = resnet20()
    #model = alexnet()
    model = model.cuda()
    model_flat = flatten_all(model)
    dist.broadcast(model_flat, world_size)
    unflatten_all(model, model_flat)
    model_l = flatten(model)
    model_r = flatten(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, weight_decay=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9 , weight_decay=0.0001)

    cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = datasets.CIFAR10(root='./data', train=True,download=False, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128//world_size,pin_memory=True,shuffle=False, num_workers=2, sampler=train_sampler)
    
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    valset = datasets.CIFAR10(root='./data', train=False,download=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100,pin_memory=True,shuffle=False, num_workers=2)
  
    for epoch in range(args.epochs):
        dist.barrier()
        # adjust learning rate 

        if epoch in adjust: 
            current_lr = current_lr * 0.1    
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # train for one epoch
        train_sampler.set_epoch(epoch)	
        loss = train(train_loader, model, criterion, optimizer, epoch, rank, world_size, model_l, model_r)
        dist.barrier()
        model_flat = flatten_all(model)
        
        dist.reduce(torch.FloatTensor([loss]), world_size, op=dist.reduce_op.SUM)
        dist.reduce(model_flat, world_size, op=dist.reduce_op.SUM)

        #output.write('Epoch: %d  Time: %3f  Train_loss: %3f  Val_acc: %3f\n'%(epoch,time_cost,loss,prec1))

def train(train_loader, model, criterion, optimizer, epoch, rank, world_size, model_l, model_r):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.add_(0.0001,param.data)   
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        
        # communicate
        model_flat = flatten(model)
        broadcast(model_flat, rank, world_size, model_l, model_r)
        model_flat.add_(model_l)
        model_flat.add_(model_r)
        model_flat.div_(3)
        unflatten(model, model_flat)

        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #model.train()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

    return losses.avg, top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def broadcast(data, rank, world_size, recv_buff_l, recv_buff_r):
    left = ((rank - 1) + world_size) % world_size
    right = (rank + 1) % world_size
    send_req_l = dist.isend(data, dst=left)
    recv_req_r = dist.irecv(recv_buff_r, src=right)
    recv_req_r.wait()
    send_req_l.wait()
    send_req_r = dist.isend(data, dst=right)
    recv_req_l = dist.irecv(recv_buff_l, src=left)
    recv_req_l.wait()
    send_req_r.wait()

def flatten_all(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    for b in model._all_buffers():
        vec.append(b.data.view(-1))
    return torch.cat(vec)

def unflatten_all(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param
    for b in model._all_buffers():
        num_param = torch.prod(torch.LongTensor(list(b.size())))
        b.data = vec[pointer:pointer + num_param].view(b.size())
        pointer += num_param

def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)

def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

if __name__ == '__main__':
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    #run(rank, world_size)
    if rank == world_size - 1:
        coordinate(rank, world_size - 1)
    else:
        run(rank, world_size - 1)
