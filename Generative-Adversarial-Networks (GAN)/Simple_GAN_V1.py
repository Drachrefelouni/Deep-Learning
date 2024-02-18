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

def get_uniform(batch_size):
    return torch.rand(batch_size,1)+5

def get_normal(batch_size):
    return torch.randn(batch_size,1) + 5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )
        
    def forward(self,x):
        return self.main(x)
    
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        return self.main(x)   
    
def update_D(batch_size):
    real_samples = get_normal(batch_size)
    prob_real_is_real = disc(real_samples)
    
    noise = get_uniform(batch_size)
    generated_samples = gen(noise)
    prob_fake_is_fake = disc(generated_samples)
    
    loss = (-torch.log(prob_real_is_real) - torch.log(1-prob_fake_is_fake)).mean()
    
    discriminator_opt.zero_grad()
    loss.backward()
    discriminator_opt.step()
    
    return loss.data.numpy().item()

def update_G(batch_size):
    
    noise = get_uniform(batch_size)
    generated_samples = gen(noise)
    prob_fake_is_real = disc(generated_samples)
    
    loss = (-torch.log(prob_fake_is_real)).mean()
    
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
        plt.figure(figsize=[10,10])
        
        sns.kdeplot(gen(get_uniform(1000)).data.numpy()[:,0], label='generated distrubition')
        sns.kdeplot((get_normal(1000)).data.numpy()[:,0], label='Real distrubition')
        
        x = np.linspace(0,10, dtype ='float32')
        p_real = disc(torch.from_numpy(x[:,None]))
        plt.plot(x,p_real.data.numpy(), label='P(real)')
        plt.legend()
        plt.show()
    
    
    
    
    
    
    
    
    
    
    























