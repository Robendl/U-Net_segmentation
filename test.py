import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #Load image
        image_path = "data/test/tumor_images/image" + str(idx) + ".png"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0

        #Load label
        label_path = "data/test/tumor_labels/image" + str(idx) + ".png"
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label / 255.0   

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


def main():
    model = UNet(in_channels=1, out_channels=1)
    model = model.cuda()

    state_dict = torch.load("results/unet.pth", map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict, strict=False)
    image_indices = list(range(0,50))

    test = ImageDataset(image_indices)
    test_set = DataLoader(test, batch_size=1, shuffle=True, num_workers=4)

    for (image, label) in test_set:
        with torch.no_grad():

            image = image.unsqueeze(1)
            output = model(image.float().to('cuda'))
            label = label.float().to('cuda')
            
            output_binary = (output > 0.5).float()

            # print(org_image.shape)
            # org_image = org_image.detach().cpu().numpy()
            # org_image = pil_im.fromarray((org_image * 255).astype(np.uint8))
            # org_image = org_image.convert("L")
            # org_image.save('org_image' + str(idx) + '.png')

            output_binary = output_binary.squeeze()  
            output_binary = output_binary.detach().cpu().numpy()
            output_image = Image.fromarray((output_binary * 255).astype(np.uint8))
            output_image = output_image.convert("L")
            output_image.save('tensor_image.png')


if __name__ == '__main__':
    main()
