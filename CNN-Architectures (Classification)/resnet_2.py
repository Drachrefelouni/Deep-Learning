#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:27:14 2023

@author: achref
"""
import torch
import torch.nn as nn

# ConvBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c(x))


# Botleneck ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        res_channels = in_channels//4
        stride = 1
        self.projection = in_channels != out_channels

        if self.projection:
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            print("la 1 "+str(in_channels)+"   "+str(out_channels))
            stride = 2
            res_channels = in_channels // 2
        else:
            print('la 1 pas de projection')
        if first:
            self.p = ConvBlock(in_channels, out_channels, 1, 1, 0)
            print("la 2 " + str(in_channels)+"   "+str(out_channels))
            stride = 1
            res_channels = in_channels

        else:
            print('la 1 pas de first')
            
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0) 
        print("c1   "+str(in_channels)+"   "+str(res_channels))

        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1)
        print("c2  "+str(res_channels)+"   "+str(res_channels))

        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        print("c3   "+str(res_channels)+"   "+str(out_channels))

        self.relu = nn.ReLU()
        print("********************************************")

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h

# ResNetx


class ResNet(nn.Module):
    def __init__(self, no_blocks, in_channels=3, classes=1000):
        super().__init__()
        out_features = [256, 512, 1024, 2048]
        print('ici')

        self.blocks = nn.ModuleList([ResidualBlock(64, 256, True)])

        for i in range(len(out_features)):
            print('*************')
            if i > 0:
                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i]))
                print( 'i > 0 ' + str(out_features[i-1],)+"   "+str(out_features[i]))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(ResidualBlock(out_features[i], out_features[i]))
                print('___' + str(out_features[i],) +"   "+str(out_features[i],))
                print('i ' + str(i))

            print("******************** new ************************")

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


no_blocks = [3, 4, 6, 3]
res = ResNet(no_blocks)

# print(res)
