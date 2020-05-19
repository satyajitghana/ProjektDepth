import torch
import torch.nn as nn

class ResDoubleConv(nn.Module):
    '''Basic DoubleConv of a ResNetV2'''

    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.double_conv(x)

        return out


class ResDownBlock(nn.Module):
    '''Basic DownBlock of a ResNetV2'''

    def __init__(self, in_channels, out_channels):
        super(ResDownBlock, self).__init__()

        self.double_conv = ResDoubleConv(in_channels, out_channels)

        self.proj_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.down_sample = nn.MaxPool2d(2)

    def forward(self, input):
        identity = self.proj_layer(input)
        out = self.double_conv(input)
        out = out + identity

        return self.down_sample(out), out


class ResUpBlock(nn.Module):
    '''Basic UpBlock of a ResNetV2'''

    def __init__(self, in_channels, out_channels, skip_channels, dense_channels=None):
        super(ResUpBlock, self).__init__()

        self.pre_conv = nn.Conv2d(in_channels, in_channels*4, kernel_size=1, bias=False)

        self.skip_conv = nn.Conv2d(skip_channels, in_channels, kernel_size=1, bias=False)

        if dense_channels is not None:
            self.dense_conv = nn.Conv2d(dense_channels, in_channels, kernel_size=1, bias=False)

        self.upsample = nn.PixelShuffle(2)

        self.double_conv = ResDoubleConv(in_channels, out_channels)

        self.proj_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, down_input, skip_input, dense_input=None):

        x = self.pre_conv(down_input)

        x = self.upsample(x) + self.skip_conv(skip_input)

        if dense_input is not None:
            x += self.dense_conv(dense_input)

        identity = self.proj_layer(x)

        out = self.double_conv(x) + identity

        return out


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # Init Conv
        # H ; input = 6, H ; out = 32, H
        self.init_conv = nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2, bias=False)

        # Encoder
        # H / 2   ; in = 32, H      ; out = 64, H/2    ; skip1 = 64, H
        self.res_down1 = ResDownBlock(32, 64)
        # H / 4   ; in = 64, H/2    ; out = 128, H/4   ; skip2 = 128, H/2
        self.res_down2 = ResDownBlock(64, 128)
        # H / 8   ; in = 128, H/4   ; out = 256, H/8   ; skip3 = 256, H/4
        self.res_down3 = ResDownBlock(128, 256)
        # H / 16  ; in = 256, H/8   ; out = 512, H/16  ; skip4 = 512, H/8
        self.res_down4 = ResDownBlock(256, 512)

        # Bridge
        self.bridge = ResDoubleConv(512, 512)

        # Depth Decoder
        # H / 8  ; in = 512, H/8(upscaled)    512, H/8(skip4)   ; out = 256, H/8(dskip4)
        self.d_res_up4 = ResUpBlock(512, 256, 512)
        # H / 4  ; in = 512, H/4(upscaled)    256, H/4(skip3)   ; out = 128, H/4(dskip3)
        self.d_res_up3 = ResUpBlock(256, 128, 256)
        # H / 2  ; in = 256, H/2(upscaled)    128, H/2(skip2)   ; out = 64, H/2(dskip2)
        self.d_res_up2 = ResUpBlock(128, 64, 128)
        # H / 1  ; in = 128, H/1(upscaled)    64, H/1(skip1)    ; out = 16, H/1(dskip1)
        self.d_res_up1 = ResUpBlock(64, 16, 64)

        # Depth Output
        self.depth_output = nn.Conv2d(
            16, 1, kernel_size=1, stride=1, bias=False)  # out = 1, H

        # Segmentation Decoder
        # H / 8  ; in = 512, H/8(upscaled)    512, H/8(skip4)   256, H/8(dkip4)   ; out = 66, H/8
        self.s_res_up4 = ResUpBlock(512, 64, 512, 256)
        # H / 4  ; in = 64, H/4(upscaled)     256, H/4(skip3)   128, H/4(dkip3)   ; out = 64, H/4
        self.s_res_up3 = ResUpBlock(64, 64, 256, 128)
        # H / 2  ; in = 54, H/2(upscaled)     128, H/2(skip2)   64, H/2(dskip2)   ; out = 32, H/2
        self.s_res_up2 = ResUpBlock(64, 32, 128, 64)
        # H / 1  ; in = 32, H/1(upscaled)     64, H/1(skip1)    16, H/1(dskip1)   ; out = 16, H/1
        self.s_res_up1 = ResUpBlock(32, 16, 64, 16)

        # Segmentation Output
        self.segment_output = nn.Conv2d(
            16, 1, kernel_size=1, stride=1, bias=False)  # out = 1, H

    def forward(self, input):

        init = self.init_conv(input)

        # Encoder
        rd1, skip1_out = self.res_down1(init)
        rd2, skip2_out = self.res_down2(rd1)
        rd3, skip3_out = self.res_down3(rd2)
        rd4, skip4_out = self.res_down4(rd3)

        # Bridge
        bridge = self.bridge(rd4)

        # # Depth Decoder
        dru4 = self.d_res_up4(bridge, skip4_out)
        dru3 = self.d_res_up3(dru4, skip3_out)
        dru2 = self.d_res_up2(dru3, skip2_out)
        dru1 = self.d_res_up1(dru2, skip1_out)

        d_out = self.depth_output(dru1)

        # # Segmentation Decoder
        sru4 = self.s_res_up4(bridge, skip4_out, dru4)
        sru3 = self.s_res_up3(sru4, skip3_out, dru3)
        sru2 = self.s_res_up2(sru3, skip2_out, dru2)
        sru1 = self.s_res_up1(sru2, skip1_out, dru1)

        s_out = self.segment_output(sru1)

        return d_out, s_out