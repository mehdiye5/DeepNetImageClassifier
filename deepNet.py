from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import csv
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from dataLoader import *

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)

import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
        # <<TODO#3>> Add more conv layers with increasing 
        # output channels
        # <<TODO#4>> Add normalization layers after conv
        # layers (nn.BatchNorm2d)

        # Also experiment with kernel size in conv2d layers (say 3
        # inspired from VGGNet)
        # To keep it simple, keep the same kernel size
        # (right now set to 5) in all conv layers.
        # Do not have a maxpool layer after every conv layer in your
        # deeper network as it leads to too much loss of information.

        self.conv0 = nn.Conv2d(3, 32, 3,padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(64, 96, 3,padding=1)
        self.bn3_2 = nn.BatchNorm2d(96)
        self.conv5_2 = nn.Conv2d(256, 384, 3,padding=1)
        self.bn5_2 = nn.BatchNorm2d(384)
        #self.conv6_2 = nn.Conv2d(512, 768, 3,padding=1)
        #self.bn6_2 = nn.BatchNorm2d(768)



        self.conv1 = nn.Conv2d(32, 48, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(48, 64, 3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(96, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=1)        
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(384, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)        
        self.bn6 = nn.BatchNorm2d(1024)

        #self.dropout = nn.Dropout(0.2)

        # <<TODO#3>> Add more linear (fc) layers
        # <<TODO#4>> Add normalization layers after linear and
        # experiment inserting them before or after ReLU (nn.BatchNorm1d)
        # More on nn.sequential:
        # http://pytorch.org/docs/master/nn.html#torch.nn.Sequential
        
        self.fc_net = nn.Sequential(
            #nn.Flatten(),
            #nn.Dropout(0.2),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
            #nn.Linear(1024, 256),
            #nn.BatchNorm1d(256),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
            #nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, 128),
            #nn.BatchNorm1d(128),            
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.2),            
            nn.Linear(1024, TOTAL_CLASSES)
            
        )

    def forward(self, x):

        # <<TODO#3&#4>> Based on the above edits, you'll have
        # to edit the forward pass description here.
        
        x = F.relu(self.conv0(x))
        x = self.bn0(x)
        
        #print("1 Original", x.shape)
        #x = self.pool(F.relu(self.conv1(x)))        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)        
        # Output size = 28//2 x 28//2 = 14 x 14
        #print("2 Conv1", x.shape)

        x = F.relu(self.conv2(x))
        #print("3 Conv2", x.shape)
        x = self.pool(x)
        #print("4 Pool2", x.shape)
        x = self.bn2(x)
        # Output size = 10//2 x 10//2 = 5 x 5
        
        #print(x.shape)
        x = F.relu(self.conv3_2(x))
        x = self.bn3_2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)        
        #print(x.shape)
        #print("5 Conv3", x.shape)
        
        x = F.relu(self.conv4(x))
        #print("6 Conv4", x.shape)
        x = self.pool(x)
        #print("7 Pool4", x.shape)
        x = self.bn4(x)

        #========testing===========
        x = F.relu(self.conv5_2(x))
        x = self.bn5_2(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)        
        #print(x.shape)
        #print("8 Conv5", x.shape)
        

        #x = F.relu(self.conv6_2(x))        
        #x = self.pool(x)        
        #x = self.bn6_2(x)
        x = F.relu(self.conv6(x))        
        x = self.pool(x)        
        x = self.bn6(x)
        #==========================        
        
        
        
        x = F.relu(x)
        
        #print("8", x.shape)

        # See the CS231 link to understand why this is 16*5*5!
        # This will help you design your own deeper network
        x = x.view(-1, 1024 * 4 * 4)
        
        #print("8", x.shape)
        x = self.fc_net(x)
        #print("9", x.shape)
        #print("11 FC Out", x.shape)

        # No softmax is needed as the loss function in step 3
        # takes care of that
        
        return x