import torch
import torch.nn as nn


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
    def __init__(self, in_channels, out_channels, mode):
        super(UNet, self).__init__()
        self.mode = mode
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

        self.up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = UpConv(1024,512)
        self.up2 = UpConv(512,256)
        self.up3 = UpConv(256,128)
        self.up4 = UpConv(128,64)

        # self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.bottleneck(x)

        if self.mode == 'simclr':
            return self.up(x)

        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)

        # x = self.out(x)
        return x


class UnetWithHeader(nn.Module):
    def __init__(self, n_channels, n_classes, mode):
        super(UnetWithHeader, self).__init__()

        self.unet = UNet(in_channels=n_channels, out_channels=n_classes, mode=mode)
        self.mode = mode

        if self.mode == 'mlp':
            self.head = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1),
                                      nn.Conv2d(256, n_classes, kernel_size=1))

        elif self.mode == "cls":
            self.head = nn.Conv2d(64, n_classes, kernel_size=1)

        elif self.mode == "simclr":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 128))

    def forward(self, x):
        y = self.unet(x)

        if self.mode == 'simclr':
            y = self.gap(y)
            output = self.head(y.squeeze())
            return output
        
        output = self.head(y)
        return output
