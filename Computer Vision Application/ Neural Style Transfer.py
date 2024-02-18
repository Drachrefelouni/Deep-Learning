#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:46:46 2023

@author: achref
"""
import torch
import torch.nn as nn
import torch.optim  as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image


model = models.vgg19(pretrained=True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        
        self.chosen_features = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features
        
        
    def forward(self,x):
            features = []
            
            for layer_num, layer in enumerate(self.model):
                    x = layer(x)
                    if str(layer_num) in self.chosen_features:
                        features.append(x)
            return features
            
        

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

image_size = 356
loader = transforms.Compose(
    [
     transforms.Resize((image_size,image_size)),
     transforms.ToTensor()
     ]
    )


orig_img = load_image("/home/achref/Bureau/CNN/content.png")
style_img = load_image("/home/achref/Bureau/CNN/style.png")
model = VGG().to(device)
generated = orig_img.clone().requires_grad_(True)

total_steps = 6000
lr = 0.001
alpha = 1
beta  = 0.01

optimizer = optim.Adam([generated], lr=lr)


for step in range(total_steps):
    print(step)
    generated_features = model(generated)
    original_img_features = model(orig_img)
    style_features = model(style_img)
    
    style_loss = original_loss = 0
    
    for gen_features,orig_feature,  style_feature in zip(
        generated_features, original_img_features,style_features):
        
        batch_size, channel , height , width = gen_features.shape
        original_loss += torch.mean((gen_features - orig_feature)**2)
        # Compute Gram Matrix
        
        G = gen_features.view(channel, height*width).mm(
            gen_features.view(channel, height*width).t()
            )
        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
            )
        style_loss += torch.mean((G-A)**2)

      

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step%10 == 0:
        print(total_loss)
        save_image(generated,"generated.png")
    
    







orig_img = load_image("/home/achref/Bureau/CNN/content.png")
style_img = load_image("/home/achref/Bureau/CNN/style.png")
model = VGG().to(device)
generated_1 = orig_img.clone().requires_grad_(True)

total_steps = 6000
lr = 0.001
alpha = 1
beta  = 0.01

optimizer = optim.Adam([generated_1], lr=lr)


for step in range(total_steps):
    print(step)
    generated_features = model(generated_1)
    original_img_features = model(orig_img)
    style_features = model(style_img)
    
    style_loss = original_loss = 0
    
    for gen_features,orig_feature,  style_feature in zip(
        generated_features, original_img_features,style_features):
        
        batch_size, channel , height , width = gen_features.shape

        original_loss += torch.mean((gen_features - orig_feature)**2)
        # Compute Gram Matrix
        

        style_loss += torch.mean((gen_features-style_feature)**2)

      

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step%10 == 0:
        print(total_loss)
        save_image(generated_1,"generated_1.png")












