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

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.pre_conv = nn.Conv2d(
            in_channels, in_channels*4, kernel_size=1, bias=False)

        self.skip_conv = nn.Conv2d(
            out_channels, in_channels, kernel_size=1, bias=False)

        self.upsample = nn.PixelShuffle(2)

        self.double_conv = ResDoubleConv(in_channels, out_channels)

        self.proj_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, down_input, skip_input):

        x = self.pre_conv(down_input)

        x = self.upsample(x) + self.skip_conv(skip_input)

        identity = self.proj_layer(x)

        out = self.double_conv(x) + identity

        return out


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # Encoder
        # H / 2   ; input = 192x192x6 ; output = 96x96x64   ; skip1 = 192x192x64
        self.res_down1 = ResDownBlock(6, 64)
        # H / 4   ; input = 96x96x64  ; output = 48x48x128  ; skip2 = 96x96x128
        self.res_down2 = ResDownBlock(64, 128)
        # H / 8   ; input = 48x48x128 ; output = 24x24x256  ; skip3 = 48x48x256
        self.res_down3 = ResDownBlock(128, 256)
        # H / 16  ; input = 24x24x256 ; output = 12x12x512  ; skip4 = 24x24x512
        self.res_down4 = ResDownBlock(256, 512)

        # Bridge
        self.bridge = ResDoubleConv(512, 512)

        # Depth Decoder
        # H / 8  ; input = 24x24x512(upscaled)   24x24x512(skip4)  ; output = 24x24x512(dskip4)
        self.d_res_up4 = ResUpBlock(512, 512)
        # H / 4  ; input = 48x48x512(upscaled)   48x48x256(skip3)  ; output = 48x48x256(dskip3)
        self.d_res_up3 = ResUpBlock(512, 256)
        # H / 2  ; input = 96x96x256(upscaled)   96x96x128(skip2)  ; output = 96x96x128(dskip2)
        self.d_res_up2 = ResUpBlock(256, 128)
        # H / 1  ; input = 192x192x128(upscaled) 192x192x64(skip1) ; output = 192x192x64(dskip1)
        self.d_res_up1 = ResUpBlock(128, 64)

        # Depth Output
        self.depth_output = nn.Conv2d(
            64, 1, kernel_size=1, stride=1, bias=False)  # output = 192x192x1

        # Segmentation Decoder
        # H / 8  ; input = 24x24x1024(upscaled)  24x24x512(dskip4)  24x24x512(skip4)  ; output = 24x24x512
        self.s_res_up4 = ResUpBlock(512, 512)
        # H / 4  ; input = 48x48x512(upscaled)   48x48x256(dskip3)  48x48x256(skip3)  ; output = 48x48x256
        self.s_res_up3 = ResUpBlock(512, 256)
        # H / 2  ; input = 96x96x256(upscaled)   96x96x128(dskip2)  96x96x128(skip2)  ; output = 96x96x128
        self.s_res_up2 = ResUpBlock(256, 128)
        # H / 1  ; input = 192x192x128(upscaled) 192x192x64(dskip1) 192x192x64(skip1) ; output = 192x192x64
        self.s_res_up1 = ResUpBlock(128, 64)

        # Segmentation Output
        self.segment_output = nn.Conv2d(
            64, 1, kernel_size=1, stride=1, bias=False)  # output = 192x192x1

    def forward(self, input):

        # Encoder
        rd1, skip1_out = self.res_down1(input)
        rd2, skip2_out = self.res_down2(rd1)
        rd3, skip3_out = self.res_down3(rd2)
        rd4, skip4_out = self.res_down4(rd3)

        # Bridge
        bridge = self.bridge(rd4)

        # Depth Decoder
        dru4 = self.d_res_up4(bridge, skip4_out)
        dru3 = self.d_res_up3(dru4, skip3_out)
        dru2 = self.d_res_up2(dru3, skip2_out)
        dru1 = self.d_res_up1(dru2, skip1_out)

        d_out = self.depth_output(dru1)

        # Segmentation Decoder
        sru4 = self.s_res_up4(bridge, skip4_out + dru4)
        sru3 = self.s_res_up3(sru4, skip3_out + dru3)
        sru2 = self.s_res_up2(sru3, skip2_out + dru2)
        sru1 = self.s_res_up1(sru2, skip1_out + dru1)

        s_out = self.segment_output(sru1)
        return d_out, s_out
