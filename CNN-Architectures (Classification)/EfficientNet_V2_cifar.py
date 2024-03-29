#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:53:59 2023

@author: achref
"""
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, act=True, bias=False):
        super().__init__()
        """ If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2. """
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.silu(self.bn(self.c(x)))
        return x

# Squeeze-and-Excitation Block
class SeBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ x shape: [N, C, H, W]. """ 
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.silu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]
        """ f shape: [N, C, 1, 1] """ 

        scale = x * f
        return scale

# MBConv
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp, r):
        super().__init__()
        exp_channels = in_channels * exp
        self.add = in_channels == out_channels and stride == 1
        self.c1 = ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity()
        self.c2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, exp_channels)
        self.se = SeBlock(exp_channels, r)
        self.c3 = ConvBlock(exp_channels, out_channels, 1, 1, act=False)

        " Stochastic Depth module with default survival probability 0.5. "
       # self.sd = StochasticDepth()

    def forward(self, x):
        f = self.c1(x)
        f = self.c2(f)
        f = self.se(f)
        f = self.c3(f)

        if self.add:
            f = x + f

       # f = self.sd(f)

        return f

# Classfier
class Classifier(nn.Module):
    """ Last stage with Average Pooling and Fully-Connected layer. """ 
    def __init__(self, in_channels, classes, p):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, classes)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(self.pool(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

# EfficientNet
class EfficientNet(nn.Module):
    def __init__(
        self, 
        model_config : str,
        in_channels : int = 3, 
        classes : int = 1000,
        show : str = False
        ):
        super().__init__()
        self.show = show # if True print the output dim between layers 
        config = Config()
        stages = config.stages
        phis = config.phis[model_config]

        """ Parameters. """
        phi, res, p = phis
        self._calculate_coef(phi)

        """ Network. """
        self.net = nn.ModuleList([])
        self.channels = []

        """ First stage Conv3x3. """
        f, c, l, k, s, exp = stages[0]
        self._add_layer(3, f, c, l, k, s)

        """ 2-8 stages with MBConvs. """
        for i in range(1, len(stages)-1):
            if i == 1:
                r = 4
            else:
                r = 24

            f, c, l, k, s, exp = stages[i]
            self._add_layer(self.channels[-1], f, c, l, k, s, exp, r)

        """ Last stage Conv1x1 + Classifier. """
        f, c, l, k, s, exp = stages[-1]
        self._add_layer(self.channels[-1], f, c, l, k, s)
        self.net.append(Classifier(self.channels[-1], classes, p))

    def forward(self, x):
        ''' Fancy way to print shapes of certain stages. '''
        i = 1
        for F in self.net:
            in_feat, h, w = x.shape[1:]

            x = F(x)
            if in_feat != x.shape[1] and i < 10:
                if self.show : print("Stage {} -> ".format(i), [x.shape[1], h, w])
                i += 1
        return x

    def _add_layer(self, in_channels, f, c, l, k, s, *args):
        c, l = self._update_feat(c, l)
        if l == 1:
            self.net.append(f(in_channels, c, k, s, *args))
        else:
            """ First layer with stride 1. """
            self.net.append(f(in_channels, c, k, 1, *args))
            
            """ Another layers with stride 1. """
            for _ in range(l-2):
                self.net.append(f(c, c, k, 1, *args))                
        
            """ Final layer with stride s(1 or 2). """
            self.net.append(f(c, c, k, s, *args))

        self.channels.append(c)
                
    def _calculate_coef(self, phi, alpha=1.2, beta=1.1):
        self.d = alpha**phi
        self.w = beta**phi

    def _update_feat(self, c, l):
        return int(c * self.w), int(l * self.d)

class Config:
    stages = [
            # [Operator(f), Channels(c), Layers(l), Kernel(k), Stride(s), Expansion(exp)]
            [ConvBlock, 32, 1, 3, 2, 1], 
            [MBConv, 16, 1, 3, 1, 1],
            [MBConv, 24, 2, 3, 2, 6],
            [MBConv, 40, 2, 5, 2, 6],
            [MBConv, 80, 3, 3, 2, 6],
            [MBConv, 112, 3, 5, 1, 6],
            [MBConv, 192, 4, 5, 2, 6],
            [MBConv, 320, 1, 3, 1, 6],
            [ConvBlock, 1280, 1, 1, 1, 0]
    ]

    phis = {
            # BX : (phi, resolution, dropout) 
            "B0" : (0, 224, 0.2), 
            "B1" : (0.5, 240, 0.2),
            "B2" : (1, 260, 0.3),
            "B3" : (2, 300, 0.3),
            "B4" : (3, 380, 0.4),
            "B5" : (4, 456, 0.4),
            "B6" : (5, 528, 0.5),
            "B7" : (6, 600, 0.5)
    }

            
 
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

in_channels=3
num_classes=10
lr=0.001
bach_size= 10
epoches= 5
load_model=False




train_dataset= datasets.CIFAR10(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.CIFAR10(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)

device='cpu'
version= 'b0'

model_config = "B0"
model = EfficientNet(model_config)

 

print(model)
model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr)

 

for epoch in range(epoches):
    losses=[]
    
 
    
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        
        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        print(f'Loss of ech was {loss: .5f}')

        #gradient descent
        optimizer.step()
    mean_loss= sum(losses)/len(losses)
    print(f'Loss of ech {epoch} was {mean_loss: .5f}')
    
#check acc

def check_acc(loader,model):
    
    if loader.dataset.train :
        print('checking acc on training data')
    else:
        print('checking acc on test data')
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader :
            x=x.to(device=device)
            y=y.to(device=device)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct+=(preds==y).sum()
            num_samples+=preds.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    
    model.train()
 

check_acc(train_loader, model)
check_acc(test_loader, model)


