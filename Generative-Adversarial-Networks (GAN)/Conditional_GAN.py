#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:12:24 2023

@author: achref
"""

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
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,num_classes,img_size):
        super(Discriminator,self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._bloc(features_d, features_d*2, 4, 2, 1),
            self._bloc(features_d*2, features_d*4, 4, 2, 1),
            self._bloc(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8,1, 4, 2, 0),
            )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
        
    def _bloc(self, in_channels, out_channels, kernel_size, stride, panding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, panding,bias = False),
            nn.InstanceNorm2d(out_channels,affine = True),
            nn.LeakyReLU(0.2)
            )
    
    def forward(self,x,labels):
        embedding = self.embed(labels).view(labels.shape[0], 1 , self.img_size, self.img_size)
        x = torch.cat([x,embedding], dim =1)
        return self.disc(x)
    
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g,num_classes, img_size, embed_size):
        super(Generator,self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            # input: N * z_dim * 1 * 1
            self._bloc(z_dim+embed_size, features_g*16, 4, 1, 0), # N * f_g * 16 * 4 * 4
            self._bloc(features_g*16, features_g*8, 4, 2, 1), # 8*8
            self._bloc(features_g*8, features_g*4, 4, 2, 1), # 16*16
            self._bloc(features_g*4, features_g*2, 4, 2, 1), # 32*32
            nn.ConvTranspose2d(features_g*2, channels_img, 4,2,1),
            nn.Tanh()
            
            )
        self.embed = nn.Embedding(num_classes, embed_size)
        
        
        
    def _bloc(self, in_channels, out_channels, kernel_size, stride, panding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, panding,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
    def forward(self,x,labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding],dim=1)
        return self.gen(x)
