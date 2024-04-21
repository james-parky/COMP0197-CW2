from torch import nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.down1 = CBR(3, 64)
        self.down2 = CBR(64, 128)
        self.down3 = CBR(128, 256)
        self.down4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up_c1 = CBR(512, 256)
        self.up_c2 = CBR(256, 128)
        self.up_c3 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)

        u1 = self.up1(d4)
        u1 = torch.cat([u1, d3], 1)
        u1 = self.up_c1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], 1)
        u2 = self.up_c2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], 1)
        u3 = self.up_c3(u3)

        return self.final(u3)
