#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:57:06 2023

@author: achref
"""
import torch 
import torch.nn as nn

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))

# Bottleneck ResidualBlock 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        res_channels = in_channels // 4
        print(res_channels)
        stride = 1

        self.projection = in_channels!=out_channels
        if self.projection:
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2
            #print("c1 projection  "+str(in_channels)+"   "+str(out_channels))
        else:
            print('pas de projection')
        if first:
            self.p = ConvBlock(in_channels, out_channels, 1, 1, 0)
            print("c1 first "+str(in_channels)+"   "+str(out_channels))
            stride = 1
            res_channels = in_channels
        else:
            print('pas de first')

        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0) 
        print("c1   "+str(in_channels)+"   "+str(res_channels))

        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1)
        print("c2  "+str(res_channels)+"   "+str(res_channels))

        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        print("c3   "+str(res_channels)+"   "+str(out_channels))

        self.relu = nn.ReLU()
        print('*************************************************')
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
    def __init__(
        self, 
        config_name : int, 
        in_channels=3, 
        classes=1000
        ):
        super().__init__()

        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }

        no_blocks = configurations[config_name]

        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResidualBlock(64, 256, True)])

        for i in range(len(out_features)):
            if i > 0:
                print( 'i > 0 ' + str(out_features[i-1])+"   "+str(out_features[i]))

                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i]))
            for _ in range(no_blocks[i]-1):
                print(str(i)+ '___' + str(out_features[i],) +"   "+str(out_features[i],))

                self.blocks.append(ResidualBlock(out_features[i], out_features[i]))
            print("******************** new ************************")

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            print(block)
            print('*********************')
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


if __name__ == "__main__":
    config_name = 50 # 50-layer
    resnet50 = ResNet(50)
    image = torch.rand(1, 3, 224, 224)
    print(resnet50(image).shape)