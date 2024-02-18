#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:25:39 2023

@author: achref
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision


class CNN(nn.Module):
    
    def __init__(self,in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool= nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2=nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1= nn.Linear(16*7*7, num_classes)
        
        
    def forward(self,x):
        x = F.relu_(self.conv1(x))
        x = self.pool(x)
        x = F.relu_(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x) 
        return x

def save_checkpoint(state, filename="my_checkepoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

in_channels=1
num_classes=10
lr=0.001
bach_size= 64
epoches= 2
load_model=False


train_dataset= datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)




batch_sizes = [256]
learning_rates = [0.001]
classes= ['0','1','2','3','4','5','6','7','8','9']
for batch_size in batch_sizes:
    for learning_rate in learning_rates:

        step=0
        model=CNN().to(device)
        criterion=nn.CrossEntropyLoss()
        train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)
        optimizer=optim.Adam(model.parameters(),lr, weight_decay=0.0)
        writer= SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')

        
        
       # if load_model:
       #     load_checkpoint(torch.load("my_checkepoint.pth.tar"))
        
        for epoch in range(epoches):
            losses=[]
            accuracies=[]
           # if epoch%3 ==0:
           #     checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
           #     save_checkpoint(checkpoint)
            
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
                #gradient descent
                optimizer.step()
                
                
                # Calculate  'runing ' traning accuracy

                features=data.reshape(data.shape[0],-1)
                img_grid = torchvision.utils.make_grid(data)
                _ , predicitons = scores.max(1)
                num_correct = (predicitons== targets).sum()
                running_train_acc= float(num_correct)/float(data.shape[0])
                accuracies.append(running_train_acc)

                

                
                class_labels = [classes[label] for label in predicitons]
                writer.add_image('mnist_images', img_grid)
                writer.add_histogram('fc1', model.fc1.weight)
                writer.add_scalar('Training Loss', loss,global_step=step)
                writer.add_scalar('Training Accuracy', running_train_acc,global_step=step)
                
                if batch_idx==230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx,
                        )
                
                
                step+=1
            writer.add_hparams({'LR' :  learning_rate, 'bsize' : batch_size},
                               {'accuracy' : sum(accuracies)/len(accuracies),
                               'Loss': sum(losses)/len(losses)})
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        