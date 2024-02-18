#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:15:09 2023

@author: achref
"""
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns

mu1 , sig1 = [-5., 5.], 1.
mu2 , sig2 = [-5., 5.], 1.
mu3 , sig3 = [-5., 5.], 1.

Noise_dim = 2

def get_uniform(batch_size):
    return torch.rand(batch_size,Noise_dim)

def plot_2d_density(xy):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.kdeplot(x=xy[:,0], y=xy[:,1], cut = 5, label = 'pdata')
    

def sample_real_data(batch_size):
    n1,n2 = int(batch_size/3), int(batch_size/3)
    n3 = batch_size - n1 - n2
    
    data = np.vstack([
        np.random.normal(mu1, sig1,(n1,2)),
        np.random.normal(mu2, sig2,(n2,2)),
        np.random.normal(mu3, sig3,(n3,2)),
        ])
    return torch.FloatTensor(data)


def get_normal(batch_size):
    return torch.randn(batch_size,1) + 5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=Noise_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(16, 2)
            )
        
    def forward(self,x):
        return self.main(x)
    
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        return self.main(x)   
    
def update_D(batch_size):
    real_samples = sample_real_data(batch_size)
    prob_real_is_real = disc(real_samples)
    
    noise = get_uniform(batch_size)
    generated_samples = gen(noise)
    prob_fake_is_fake = disc(generated_samples)
    
    loss = (-torch.log(prob_real_is_real) - torch.log(1-prob_fake_is_fake)).mean()
    print(loss)

    discriminator_opt.zero_grad()
    loss.backward()
    discriminator_opt.step()
    
    return loss.data.numpy().item()

def update_G(batch_size):
    
    noise = get_uniform(batch_size)
    generated_samples = gen(noise)
    prob_fake_is_real = disc(generated_samples)
    
    loss = (-torch.log(prob_fake_is_real)).mean()
    print(loss)
    
    generator_opt.zero_grad()
    loss.backward()
    generator_opt.step()
    
    return loss.data.numpy().item()

gen = Generator()
disc = Discriminator()
lr= 1e-3
generator_opt = torch.optim.SGD(gen.parameters(), lr=lr)
discriminator_opt = torch.optim.SGD(disc.parameters(), lr=lr)

N_DISC_OPERATIONS = 10
BATCH_SIZE = 64

for i in range(10000):
    for _ in range(N_DISC_OPERATIONS):
        update_D(BATCH_SIZE)
        
    update_G(BATCH_SIZE)
    
    if i%100 == 0:
        clear_output(True)
        plt.figure(figsize=[14,6])
        plt.subplot(1,2,1)
        plt.title("Real data")
        plot_2d_density(sample_real_data(1000).data.numpy())
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.title("Generated data")
        
        plot_2d_density(get_uniform(1000).data.numpy())
        
        axes = plt.gca()
        axes.set_ylim([-15,15])
        axes.set_xlim([-15,15])
        
        plt.show()
    
    
    
    
    
    
    
    
    
    
    























