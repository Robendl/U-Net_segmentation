import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from augmentations import *
from network import *

class ImageDataset(Dataset):
    def __init__(self, indices, image_indices, transform=False):
        self.indices = indices
        self.image_indices = image_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        random_img_number = self.image_indices[idx]

        #Load image
        image_path = "brain_tumour/train/images/" + str(random_img_number) + ".png"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (512, 512))

        image1, image2 = perform_augmentations_simclr(image, image)

        image1 = image1.astype(np.float32) / 255.0 
        image1 = torch.from_numpy(image1)

        image2 = image2.astype(np.float32) / 255.0 
        image2 = torch.from_numpy(image2)
 
        return image1, image2

# lass DownConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = self.conv(x)
#         x_pool = self.pool(x)
#         return x, x_pool

# class UpConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpConv, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, mode='simclr'):
#         super(UNet, self).__init__()
#         self.mode = mode
#         self.down1 = DownConv(in_channels,64)
#         self.down2 = DownConv(64,128)
#         self.down3 = DownConv(128,256)
#         self.down4 = DownConv(256,512)

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         self.up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.up1 = UpConv(1024,512)
#         self.up2 = UpConv(512,256)
#         self.up3 = UpConv(256,128)
#         self.up4 = UpConv(128,64)

#         # self.out = nn.Conv2d(64, out_channels, kernel_size=1)


#     def forward(self, x):
#         x1, x = self.down1(x)
#         x2, x = self.down2(x)
#         x3, x = self.down3(x)
#         x4, x = self.down4(x)

#         x = self.bottleneck(x)

#         if self.mode == 'simclr':
#             return self.up(x)

#         x = self.up1(x,x4)
#         x = self.up2(x,x3)
#         x = self.up3(x,x2)
#         x = self.up4(x,x1)

#         # x = self.out(x)
#         return x


# class UnetWithHeader(nn.Module):
#     def __init__(self, n_channels, n_classes, mode="simclr"):
#         super(UnetWithHeader, self).__init__()

#         self.unet = UNet(in_channels=n_channels, out_channels=n_classes, mode="simclr")
#         self.mode = mode

#         if self.mode == 'mlp':
#             self.head = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1),
#                                       nn.Conv2d(256, n_classes, kernel_size=1))

#         elif self.mode == "cls":
#             self.head = nn.Conv2d(64, n_classes, kernel_size=1)

#         elif self.mode == "simclr":
#             self.gap = nn.AdaptiveAvgPool2d(1)
#             self.head = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 128))


#     def forward(self, x):
#         y = self.unet(x)

#         if self.mode == 'simclr':
#             y = self.gap(y)
#             output = self.head(y.squeeze())

#         else:
#             output = self.head(y)

#         return output


def contrastive_loss(zis, zjs, batch_size, temperature):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    zijs = torch.cat((zis, zjs), dim=0)
    
    results = torch.zeros((2*batch_size, 2*batch_size)).to('cuda')
    labels = torch.zeros(2 * batch_size).to(device='cuda', dtype=torch.int64)

    for i in range(zijs.shape[0]):
        for j in range(zijs.shape[0]):
            # Calculate dot product similarity between each pair of vectors
            results[i, j] = torch.dot(zijs[i], zijs[j])

    #For filtering out positives
    diag = np.eye(2 * batch_size)
    pos1 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    pos2 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    mask = diag + pos1 + pos2
    mask = np.logical_not(mask).astype(int)
    mask = torch.from_numpy(mask)
    boolean_mask = mask != 0
    boolean_mask = boolean_mask.to('cuda')

    l_pos = torch.diag(results, batch_size)
    r_pos = torch.diag(results, -batch_size)

    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    negatives = results[boolean_mask].view(2 * batch_size, -1)


    logits = torch.cat((positives, negatives), dim=1)
    logits = logits / temperature

    loss = criterion(logits, labels) 
    return loss / (2 * batch_size)


def validate(dataloader_valset, model, batch_size):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    # validation steps
    with torch.no_grad():

        valid_loss = 0.0
        counter = 0

        for (xis, xjs) in dataloader_valset:
            xis = xis.permute(0, 3, 1, 2)
            xjs = xjs.permute(0, 3, 1, 2)

            xis = xis.float().to('cuda')
            xjs = xjs.float().to('cuda')

            zis = model(xis)
            zjs = model(xjs)

            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)

            loss = contrastive_loss(zis, zjs, batch_size, 0.5)
            valid_loss += loss.item()
            counter += 1
            
        valid_loss /= counter
    return valid_loss


def save_model(model, save_file):
    torch.save(model.state_dict(), save_file)

def load_path(model, path):
    print('path:\t', path)
    state_dict = torch.load(path, map_location=torch.device('cuda:0'))
    
    print("model before", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    model.load_state_dict(state_dict, strict=False)

    print("model after", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    return model

def main():
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="simclr")
    model = model.cuda()

    #model = load_path(model, "results/simclr.pth")

    train_indices = list(range(0,2450))
    valid_indices = list(range(0,307))

    image_indices = list(range(0,2757))
    random.shuffle(image_indices)

    train_image_indices = image_indices[0:2450]
    valid_image_indices = image_indices[2450:2757]

    num_epochs = 100
    batch_size = 8
    learning_rate = 0.00001
    best_valid_loss = np.Inf

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

        for (xis, xjs) in dataloader_trainset:
            optimizer.zero_grad()

            xis = xis.permute(0, 3, 1, 2)
            xjs = xjs.permute(0, 3, 1, 2)


            xis = xis.float().to('cuda')
            xjs = xjs.float().to('cuda')

            zis = model(xis)
            zjs = model(xjs)

            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)

            loss = contrastive_loss(zis, zjs, batch_size, 0.5)

            loss.backward()
            optimizer.step()
            batch_counter += 1
            total_loss += loss.item()


        valid_loss = validate(dataloader_valset, model, batch_size)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_file = "results/simclr.pth" 
            save_model(model, save_file)

        total_loss /= batch_counter

        print("EPOCH: ", int(epoch))
        print("train loss", total_loss)
        print("valid loss", valid_loss)


main()