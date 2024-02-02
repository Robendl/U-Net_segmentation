import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import *
from unet import *
import matplotlib.pyplot as plt
from matplotlib import colors


def binarize( gray_image , threshold ):
    return 1 * ( gray_image > threshold )

def otsuThresholdB(image):
    #Get size of image
    rows, cols =  image.shape
    #ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Plotting image histogram
    plt.figure()
    #We are interested on H (histogram), other values that plt.hist returns will be ignored here
    H, binEdges, patches = plt.hist(image.ravel(),256)
    # Getting relative histogram (pdf)
    pdf = H /(rows*cols)
    # Get cdf for all gray levels
    cdf = np.cumsum(pdf)
    #Initialization
    othresh = 1
    maxVarB = 0
    for t in range(1,255):
        #gray levels belongs to background 
        bg = np.arange(0,t)
        #object gray levels
        obj = np.arange(t, 256)
        #Calculation of mean gray level for object and background
        mBg    = sum(bg*pdf[0:t])/cdf[t]
        mObj   = sum(obj*pdf[t:256])/(1-cdf[t])
        # Calculate between class variance
        varB = cdf[t] * (1-cdf[t]) *(mObj - mBg)**2
        #Pick up max variance and corresponding threshold
        if varB > maxVarB:
            maxVarB= varB
            othresh = t
    return othresh

def thresholding(loss_function=combined_loss):
    train_indices = list(range(0,10))
    valid_indices = list(range(0,307))

    image_indices = list(range(0,10))
    random.shuffle(image_indices)

    train_image_indices = image_indices[0:10]
    valid_image_indices = image_indices[2450:2757]

    best_valid_loss = np.Inf

    train_dataset = ImageDataset(train_indices, train_image_indices, True)

    valid_dataset = ImageDataset(valid_indices, valid_image_indices)
    total_loss = 0.0

    for (images, labels) in train_dataset:
        hsvImage = colors.rgb_to_hsv(images)
        myIm = hsvImage[...,2]

        plt.figure()
        plt.imshow(myIm)
        plt.set_cmap("gray")
        oTb = otsuThresholdB(myIm)

        binaryIm = binarize(myIm, oTb)
        plt.figure()
        plt.imshow(binaryIm)
        plt.show()

        #loss = loss_function(images, labels)
        #total_loss += loss.item()



    #otsuThresholdB(image)
    image = cv2.imread('data/thresholdingTs/1.png', 0)

    # Optional: Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu's thresholding
    return

if __name__ == '__main__':
    thresholding()
