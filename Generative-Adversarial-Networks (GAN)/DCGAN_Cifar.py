#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:13:13 2023

@author: achref
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
image_channels = 3
batch_size = 64
epochs = 100
lr = 0.0002

# Create generator and discriminator
generator = Generator(latent_dim, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# DataLoader for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataloader = DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=4
)

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Update discriminator
        real_images, _ = data
        real_images = real_images.to(device)
        real_labels = torch.full((batch_size,), 1, device=device)

        # Forward pass real images through discriminator
        output_real = discriminator(real_images).view(-1)
        errD_real = criterion(output_real,  torch.zeros_like(output_real))

        # Generate fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size,), 0, device=device)

        # Forward pass fake images through discriminator
        output_fake = discriminator(fake_images.detach()).view(-1)
        errD_fake = criterion(output_fake, torch.zeros_like(output_fake))
        
        total = (errD_real+errD_fake)/2
        
        
        discriminator.zero_grad()
        total.backward()
        optimizer_D.step()

        # Update generator
        output_fake_2 = discriminator(fake_images).view(-1)
        errG = criterion(output_fake_2, torch.zeros_like(output_fake_2))
        
        generator.zero_grad()
        errG.backward()
        # Update generator
        optimizer_G.step()

        # Print statistics
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, epochs, i, len(dataloader), errD_real.item() + errD_fake.item(), errG.item()))

    # Save generated images at the end of each epoch
    with torch.no_grad():
        fake = generator(torch.randn(64, latent_dim, 1, 1, device=device))
        save_image(fake.detach(), 'fake_samples_epoch_%03d.png' % epoch, normalize=True)

print("Training finished.")
