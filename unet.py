# @title unet me
# https://github.com/milesial/Pytorch-UNet/blob/master/unet
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, bias=False):
        super().__init__()
        if mid_ch==None: mid_ch = out_ch
        act = nn.ReLU(inplace=True) # ReLU GELU SiLU ELU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=bias), nn.BatchNorm2d(mid_ch), act,
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=bias), nn.BatchNorm2d(out_ch), act,
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch),)

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=1),)
        else: self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)#, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1) # [c,h,w]
        diffX, diffY = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()
        # self.in_ch, self.out_ch, self.bilinear = in_ch, out_ch, bilinear
        # self.inc = DoubleConv(in_ch, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

        # https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
        wf=6 # width factor, 2^6 = 64 -> 1024
        depth=4
        self.inc = DoubleConv(in_ch, 2 ** wf)
        self.down_list = nn.ModuleList([Down(2 ** (wf + i), 2 ** (wf + i+1)) for i in range(depth)])
        self.up_list = nn.ModuleList([Up(2 ** (wf + i+1), 2 ** (wf + i)) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(2 ** wf, out_ch, kernel_size=1)

    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     print(x5.shape, x4.shape)
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     logits = self.outc(x)
    #     return logits

    def forward(self, x):
        blocks = []
        x = self.inc(x)
        for i, down in enumerate(self.down_list):
            blocks.append(x)
            x = down(x)
        for i, up in enumerate(self.up_list):
            x = up(x, blocks[-i - 1])
        return self.outc(x)


unet = UNet(3, 5)
x=torch.rand(4,3,64,64)
# x=torch.rand(4,3,128,128)
# x=torch.rand(4,3,28,28)
out = unet(x)
print(out.shape)
