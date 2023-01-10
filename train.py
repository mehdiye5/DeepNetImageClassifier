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
import torch.optim as optim

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from dataLoader import *
from deepNet import *

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)


def calculate_val_accuracy(valloader, is_gpu):
    """ Util function to calculate val set accuracy,
    both overall and per class accuracy
    Args:
        valloader (torch.utils.data.DataLoader): val set 
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """    
    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for i in range(TOTAL_CLASSES))
    class_total = list(0. for i in range(TOTAL_CLASSES))

    for data in valloader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        c = (predicted == labels).squeeze()
        #print(c.is_cuda)
        c = c.cpu()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    #print("class_correct", class_correct.is_cuda)
    #print("class_total", class_total.is_cuda)
    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100*correct/total, class_accuracy

if __name__ == '__main__':
    EPOCHS = 25
    # ---------

    IS_GPU = True
    TEST_BS = 256
    #TEST_BS = 64
    TOTAL_CLASSES = 100
    #TRAIN_BS = 64
    TRAIN_BS = 16
    PATH_TO_CIFAR100_SFU_CV = "/data/"


    train_transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.RandomRotation(30),           
     transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
     transforms.RandomHorizontalFlip(),     
     #transforms.RandomHorizontalFlip(0.5),
     #transforms.Normalize((0, 0, 0), (1, 1, 1))
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # ---------------------

    trainset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="train",
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BS,
                                            shuffle=True, num_workers=2)
    print("Train set size: "+str(len(trainset)))

    valset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="val",
                                        download=True, transform=test_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=2)
    print("Val set size: "+str(len(valset)))

    testset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="test",
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=2)
    print("Test set size: "+str(len(testset)))

    # The 100 classes for CIFAR100
    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


    # Create an instance of the nn.module class defined above:
    net = BaseNet()

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if IS_GPU:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()

    # Tune the learning rate.
    # See whether the momentum is useful or not
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    plt.ioff()
    fig = plt.figure()
    train_loss_over_epochs = []
    val_accuracy_over_epochs = []

    EPOCHS = 20
    start_epoch = 300
    best =73
    for epoch in range(start_epoch,start_epoch + EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if IS_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            #nn.utils.clip_grad_value_(net.parameters(), 0.01)

            optimizer.step()
            #optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
        
        # Normalizing the loss by the total number of train batches
        running_loss/=len(trainloader)
        print('[%d] loss: %.3f' %
            (epoch + 1, running_loss))

        # Scale of 0.0 to 100.0
        # Calculate validation set accuracy of the existing model
        val_accuracy, val_classwise_accuracy = \
            calculate_val_accuracy(valloader, IS_GPU)
            #calculate_val_accuracy(valloader, IS_GPU)
            
        print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

        if int(val_accuracy) > best:
        best = int(val_accuracy)
        save_model(epoch + 1, net, optimizer, 'checkpoint.pt')
        

        # # Optionally print classwise accuracies
        # for c_i in range(TOTAL_CLASSES):
        #     print('Accuracy of %5s : %2d %%' % (
        #         classes[c_i], 100 * val_classwise_accuracy[c_i]))

        train_loss_over_epochs.append(running_loss)
        val_accuracy_over_epochs.append(val_accuracy)