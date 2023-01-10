# DeepNet Image Classifier

## Project Overview

The goal of this project is to design and train deep convolutional neural networks using PyTorch. I will design a deep net architecture to classify (small) images into 100 categories and evaluate the performance of the architecture by uploading the predictions to the Kaggle competition. 

## Dataset

For this project I was working with the [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. This dataset consists of 60K 32x32 color images from 100 classes, with 600 images per class. There are 50K training images and 10K test images. The images in CIFAR100 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels. I have modified the standard dataset to create our own CIFAR100 dataset which consists of 45K training images (450 of each class), 5K validation images (50 of each class), and 10K test images (100 of each class). The training and val datasets have labels while all the labels in the test set are set to 0.

## Model Architecture
The model consists of series of convolutional, normalization and pooling layers. The outlined architechure is described below.

![archit](./images/archit.JPG)

## Performance Overview

Fully trained model has reached 92% accuracy. The improvements in performance was achieved through
- Data normalization: where normalizing input data makes training easier and more robust
- Data augmentation: By attempting different image transforms provied our model more variations of the images for our model to learn from. The augmentation that was attemptend was RandomHorizontalFlip and RandomCrop.
- Deeper network: deeper network have increased the model capacity there by improved it accuracy. With that being said there was a rist of overfitting. 
- Normalization layers: the normalization layer helped to reduce the posibility for overfitting and improve training of the model. 

I should note that this model was trained from scratch. For model selection I also trained a classifier using transfer learning using resnet18. The model using transfer learning has achieved 94% accuracy. 