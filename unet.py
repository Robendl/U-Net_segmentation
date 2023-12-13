import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import *

class ImageDataset(Dataset):
    def __init__(self, indices, image_indices, transform=False):
        self.indices = indices
        self.transform = transform
        self.image_indices = image_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        random_img_number = self.image_indices[idx]

        #Load image
        image_path = "data/train/tumor_images/image" + str(random_img_number) + ".png"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        #Load label
        label_path = "data/train/tumor_labels/image" + str(random_img_number) + ".png"
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 

        if self.transform:
            image, label = perform_augmentations(image, label)

        image = image.astype(np.float32) / 255.0 
        label = label.astype(np.float32) / 255.0  
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        return image, label

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_pool = self.pool(x)
        return x, x_pool

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DownConv(in_channels,64)
        self.down2 = DownConv(64,128)
        self.down3 = DownConv(128,256)
        self.down4 = DownConv(256,512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpConv(1024,512)
        self.up2 = UpConv(512,256)
        self.up3 = UpConv(256,128)
        self.up4 = UpConv(128,64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)

        x = self.out(x)
        return x


class SupConUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, mode="mlp"):
        super(SupConUnet, self).__init__()

        self.encoder = UNet(n_channels, bilinear=False)
        if mode == 'mlp':
            self.head = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1),
                                      nn.Conv2d(256, n_classes, kernel_size=1))

        elif mode == "cls":
            self.head = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        y = self.encoder(x)
        output = self.head(y)
        # output = F.normalize(self.head(y), dim=1)



def dice_loss(y_pred, y_true):
    smooth = 1e-5
    y_true = y_true.view(-1, 512, 512)
    y_pred = y_pred.view(-1, 512, 512)
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    dice = (2. * intersection + smooth) / (sum_ + smooth)
    return 1. - dice

def validate(dataloader_valset, model):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    # validation steps
    with torch.no_grad():

        valid_loss = 0.0
        counter = 0

        for (images, labels) in dataloader_valset:
            images = images.unsqueeze(1)
            output = model(images.float().to('cuda'))
            output_sigmoid = torch.sigmoid(output)
            labels = labels.float().to('cuda')
                
            loss = dice_loss(output_sigmoid.squeeze(), labels) + bce_loss(output.squeeze(), labels)
            valid_loss += loss.item()
            counter += 1
            
        valid_loss /= counter
    
    return valid_loss


def save_model(model, save_file):
    torch.save(model.state_dict(), save_file)


def main():
    model = UNet(in_channels=1, out_channels=1)
    model = model.cuda()

    train_indices = list(range(0,1388))
    valid_indices = list(range(0,200))

    image_indices = list(range(0,1588))
    random.shuffle(image_indices)

    train_image_indices = image_indices[0:1388]
    valid_image_indices = image_indices[1388:1588]

    num_epochs = 50
    batch_size = 8
    learning_rate = 0.00001
    best_valid_loss = np.Inf
    bce_loss = nn.BCEWithLogitsLoss()

    train_dataset = ImageDataset(train_indices, train_image_indices, True)
    dataloader_trainset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = ImageDataset(valid_indices, valid_image_indices)
    dataloader_valset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_trainset), eta_min=0,
                                                                            last_epoch=-1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_counter = 1

        for (images, labels) in dataloader_trainset:
            optimizer.zero_grad()

            images = images.unsqueeze(1)
            output = model(images.float().to('cuda'))
            labels = labels.float().to('cuda')
            output_sigmoid = torch.sigmoid(output)
                
            loss = dice_loss(output_sigmoid.squeeze(), labels) + bce_loss(output.squeeze(), labels)

            loss.backward()
            optimizer.step()
            batch_counter += 1
            total_loss += loss.item()


        valid_loss = validate(dataloader_valset, model)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_file = "results/unet.pth" 
            save_model(model, save_file)

        total_loss /= batch_counter

        print("train loss", total_loss)
        print("valid loss", valid_loss)


main()