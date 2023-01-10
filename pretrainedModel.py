import os
import os.path as osp
import time

%matplotlib inline
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from torchvision import datasets

class PreTrainedResNet(nn.Module):
  def __init__(self, num_classes, feature_extracting):
    super(PreTrainedResNet, self).__init__()
    
    #TODO1: Load pre-trained ResNet Model
    #self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    self.resnet18 = models.resnet18(pretrained=True)

    #Set gradients to false
    if feature_extracting:
      for param in self.resnet18.parameters():
          param.requires_grad = False
    
    #Replace last fc layer
    num_feats = self.resnet18.fc.in_features
    
    
    #TODO2: Replace fc layer in resnet to a linear layer of size (num_feats, num_classes)
    self.resnet18.fc = nn.Linear(num_feats, num_classes)
    steps = list(self.resnet18.children())    
    self.sequence = nn.Sequential(*steps[:-1])    
    
    
  def forward(self, x):
    #TODO3: Forward pass x through the model
    #print("Original", x.shape)
    x = self.sequence(x)    
    #x = self.resnet18(x)
    #print("Pretrained Output", x.shape)
    #x = self.model_label(x)
    #print("Fully Connected", x.shape)
    #x = x.view(-1, 112)
    x = x.view(x.size(0), -1)
    x = self.resnet18.fc(x)
    return x