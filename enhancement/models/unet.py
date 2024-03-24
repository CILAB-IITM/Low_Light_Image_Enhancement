from models import register, make
import torch.nn as nn
import torch
from .unet_cbam import SpatialGate, ResBlock
torch.autograd.set_detect_anomaly(True)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        return x


class ConvBlock2(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False, lrelu=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        if lrelu:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU(inplace = True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, batch_norm=batch_norm)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class Encoder2(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, batch_norm=batch_norm)
        self.unshuffle = nn.PixelUnshuffle(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.unshuffle(x)
        return x, p


class Encoder3(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False, act='relu'):
        super().__init__()
        if act == 'relu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=False)
        elif act == 'lrelu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=True)
        self.unshuffle = nn.PixelUnshuffle(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.unshuffle(x)
        return x, p
    
class Encoder3SAM(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False, act='relu'):
        super().__init__()
        self.res = ResBlock(in_c, in_c, bn=batch_norm,weight_normalization=True)
        if act == 'relu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=False)
        elif act == 'lrelu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=True)
        self.sg = SpatialGate()
        self.unshuffle = nn.PixelUnshuffle(2)
    def forward(self, inputs):
        x = self.conv(self.res(inputs))
        p = self.unshuffle(self.sg(x))
        return x, p

    
class Decoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c, batch_norm=batch_norm)
    def forward(self, inputs, skip):
        # print(inputs.shape, 'Before Upsampling from UNet.py line 105')
        x = self.up(inputs)
        # print(x.shape, 'After Upsampling from UNet.py line 107')
        x = torch.cat([x, skip], axis=1)
        # print(x.shape, 'After concatenation from UNet.py line 109')
        x = self.conv(x)
        # print(x.shape, 'After convolution from UNet.py line 111') 
        return x


class Decoder2(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)
        self.conv = ConvBlock(in_c, out_c, batch_norm=batch_norm)
    def forward(self, inputs, skip):
        x = self.shuffle(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class Decoder3(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False, act='relu'):
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)
        if act == 'relu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=False)
        elif act == 'lrelu':
            self.conv = ConvBlock2(in_c, out_c, batch_norm=batch_norm, lrelu=True)
        self.res = ResBlock(in_c, in_c, bn=batch_norm,weight_normalization=True)
    def forward(self, inputs, skip):
        x = self.shuffle(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(self.res(x))
        return x
    

@register('unet-basic')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(512, 1024, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs

@register('unet-basic-PSUpsampling')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(512, 1024, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        """Upsamling"""
        # self.c1 = ConvBlock(64, 3, batch_norm=False)
        
        # self.c2 = ConvBlock(3, 3, batch_norm=False)
        self.c1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.c2 = nn.Conv2d(3,3, kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        o1 = self.c1(d4)
        # o1_ = self.relu(o1)
        # o1 = self.outputs(d4)
        # o2 = self.c1(outputs)
        # print(o2.shape, 'output shape from unet.py from line 216')
        o2 = self.ps(o1)
        o3 = self.c2(o2)
        # o3_ = self.relu(o3)
        # print(outputs.shape, 'outputshape from unet.py line 223')
        torch.autograd.set_detect_anomaly(True)
        

        return o3

@register('unet-basic-PSUpsampling-exp3')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(512, 1024, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        """Upsamling"""
        # self.c1 = ConvBlock(64, 3, batch_norm=False)
        
        # self.c2 = ConvBlock(3, 3, batch_norm=False)
        self.c1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.c2 = nn.Conv2d(3,3, kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        o1 = self.c1(d4)
        # o1_ = self.relu(o1)
        # o1 = self.outputs(d4)
        # o2 = self.c1(outputs)
        # print(o2.shape, 'output shape from unet.py from line 216')
        o2 = self.ps(o1)
        o3 = self.c2(o2)
        # o3_ = self.relu(o3)
        # print(outputs.shape, 'outputshape from unet.py line 223')
        torch.autograd.set_detect_anomaly(True)
        

        return o3


@register('unet-basic-modified')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        # self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(256, 512, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(512, 256, batch_norm=batch_norm)
        self.d2 = Decoder(256, 128, batch_norm=batch_norm)
        self.d3 = Decoder(128, 64, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        # print(s1.shape, p1.shape, 'Skip Connection and Conv output from unet.py line 195')
        s2, p2 = self.e2(p1)
        # print(s2.shape, p2.shape, 'Skip Connection and Conv output from unet.py line 197')
        s3, p3 = self.e3(p2)
        # print(s3.shape, p3.shape, 'Skip Connection and Conv output from unet.py line 199')
        # s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p3)
        # print(b.shape, 'Bottle Neck layer from unet.py line 203')
        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        """ Classifier """
        outputs = self.outputs(d3)
        return outputs








@register('unet-small')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 8, batch_norm=batch_norm)
        self.e2 = Encoder(8, 16, batch_norm=batch_norm)
        self.e3 = Encoder(16, 32, batch_norm=batch_norm)
        self.e4 = Encoder(32, 98, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(98, 256, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(256, 98, batch_norm=batch_norm)
        self.d2 = Decoder(98, 32, batch_norm=batch_norm)
        self.d3 = Decoder(32, 16, batch_norm=batch_norm)
        self.d4 = Decoder(16, 8, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(8, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


@register('unet-small-2')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(in_channels, 32, batch_norm=batch_norm)
        self.e2 = Encoder(32, 128, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(128, 512, batch_norm=batch_norm)
        """ Decoder """
        self.d3 = Decoder(512, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 32, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        #s3, p3 = self.e3(p2)
        #s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p2)
        """ Decoder """
        #d1 = self.d1(b, s4)
        #d2 = self.d2(d1, s3)
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs
    


@register('unet-stereo')
class UNetStereo(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1 = Encoder(int(in_channels/2), 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(1024, 1024, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2(p1l)
        s2r, p2r = self.e2(p1r)
        s3l, p3l = self.e3(p2l)
        s3r, p3r = self.e3(p2r)
        s4l, p4l = self.e4(p3l)
        s4r, p4r = self.e4(p3r)
        """ Bottleneck """
        b = self.b(torch.cat((p4l, p4r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s4l)
        d2 = self.d2(d1, s3l)
        d3 = self.d3(d2, s2l)
        d4 = self.d4(d3, s1l)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


@register('unet-stereo-2')
class UNetStereo2(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1 = Encoder(int(in_channels/2), 64, batch_norm=batch_norm)
        self.e2 = Encoder(64, 128, batch_norm=batch_norm)
        self.e3 = Encoder(128, 256, batch_norm=batch_norm)
        self.e4 = Encoder(256, 512, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(1024, 1024, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder(1024, 512, batch_norm=batch_norm)
        self.d2 = Decoder(512, 256, batch_norm=batch_norm)
        self.d3 = Decoder(256, 128, batch_norm=batch_norm)
        self.d4 = Decoder(128, 64, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2(p1l)
        s2r, p2r = self.e2(p1r)
        s3l, p3l = self.e3(p2l)
        s3r, p3r = self.e3(p2r)
        s4l, p4l = self.e4(p3l)
        s4r, p4r = self.e4(p3r)
        """ Bottleneck """
        b = self.b(torch.cat((p4l, p4r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s4l+s4r)
        d2 = self.d2(d1, s3l+s3r)
        d3 = self.d3(d2, s2l+s2r)
        d4 = self.d4(d3, s1l+s1r)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


@register('unet-stereo-3')
class UNetStereo3(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1 = Encoder2(int(in_channels/2), 8, batch_norm=batch_norm)
        self.e2 = Encoder2(32, 32, batch_norm=batch_norm)
        self.e3 = Encoder2(128, 128, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock(1024, 512, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder2(256, 128, batch_norm=batch_norm)
        self.d2 = Decoder2(64, 32, batch_norm=batch_norm)
        self.d3 = Decoder2(16, 8, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(8, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2(p1l)
        s2r, p2r = self.e2(p1r)
        s3l, p3l = self.e3(p2l)
        s3r, p3r = self.e3(p2r)
        """ Bottleneck """
        b = self.b(torch.cat((p3l, p3r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s3l)
        d2 = self.d2(d1, s2l)
        d3 = self.d3(d2, s1l)
        """ Classifier """
        outputs = self.outputs(d3)
        return outputs


@register('unet-stereo-4')
class UNetStereo4(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, act='relu'):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1 = Encoder3(int(in_channels/2), 8, batch_norm=batch_norm, act=act)
        self.e2 = Encoder3(32, 32, batch_norm=batch_norm, act=act)
        self.e3 = Encoder3(128, 128, batch_norm=batch_norm, act=act)
        """ Bottleneck """
        self.b = ConvBlock2(1024, 512, batch_norm=batch_norm, lrelu=(act=='lrelu'))
        """ Decoder """
        self.d1 = Decoder3(256, 128, batch_norm=batch_norm, act=act)
        self.d2 = Decoder3(64, 32, batch_norm=batch_norm, act=act)
        self.d3 = Decoder3(16, 8, batch_norm=batch_norm, act=act)
        self.outputs = nn.Conv2d(8, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2(p1l)
        s2r, p2r = self.e2(p1r)
        s3l, p3l = self.e3(p2l)
        s3r, p3r = self.e3(p2r)
        """ Bottleneck """
        b = self.b(torch.cat((p3l, p3r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s3l)
        d2 = self.d2(d1, s2l)
        d3 = self.d3(d2, s1l)
        """ Classifier """
        outputs = self.outputs(d3)
        return outputs


@register('unet-stereo-5')
class UNetStereo5(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1l = Encoder3(int(in_channels/2), 6, batch_norm=batch_norm)
        self.e2l = Encoder3(24, 12, batch_norm=batch_norm)
        self.e3l = Encoder3(48, 24, batch_norm=batch_norm)
        self.e4l = Encoder3(96, 64, batch_norm=batch_norm)
        self.e5l = Encoder3(256, 128, batch_norm=batch_norm)
        self.e1r = Encoder3(int(in_channels/2), 6, batch_norm=batch_norm)
        self.e2r = Encoder3(24, 12, batch_norm=batch_norm)
        self.e3r = Encoder3(48, 24, batch_norm=batch_norm)
        self.e4r = Encoder3(96, 64, batch_norm=batch_norm)
        self.e5r = Encoder3(256, 128, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock2(1024, 256, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder3(192, 128, batch_norm=batch_norm)
        self.d2 = Decoder3(96, 96, batch_norm=batch_norm)
        self.d3 = Decoder3(48, 48, batch_norm=batch_norm)
        self.d4 = Decoder3(24, 24, batch_norm=batch_norm)
        self.d5 = Decoder3(12, 6, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(6, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1l(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1r(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2l(p1l)
        s2r, p2r = self.e2r(p1r)
        s3l, p3l = self.e3l(p2l)
        s3r, p3r = self.e3r(p2r)
        s4l, p4l = self.e4l(p3l)
        s4r, p4r = self.e4r(p3r)
        s5l, p5l = self.e5l(p4l)
        s5r, p5r = self.e5r(p4r)
        """ Bottleneck """
        b = self.b(torch.cat((p5l, p5r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s5l)
        d2 = self.d2(d1, s4l)
        d3 = self.d3(d2, s3l)
        d4 = self.d4(d3, s2l)
        d5 = self.d5(d4, s1l)
        """ Classifier """
        outputs = self.outputs(d5)
        return outputs



@register('unet-stereo-5-sam')
class UNetStereo5(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        """ Encoder """
        self.e1l = Encoder3SAM(int(in_channels/2), 6, batch_norm=batch_norm)
        self.e2l = Encoder3SAM(24, 12, batch_norm=batch_norm)
        self.e3l = Encoder3SAM(48, 24, batch_norm=batch_norm)
        self.e4l = Encoder3SAM(96, 64, batch_norm=batch_norm)
        self.e5l = Encoder3SAM(256, 128, batch_norm=batch_norm)
        self.e1r = Encoder3SAM(int(in_channels/2), 6, batch_norm=batch_norm)
        self.e2r = Encoder3SAM(24, 12, batch_norm=batch_norm)
        self.e3r = Encoder3SAM(48, 24, batch_norm=batch_norm)
        self.e4r = Encoder3SAM(96, 64, batch_norm=batch_norm)
        self.e5r = Encoder3SAM(256, 128, batch_norm=batch_norm)
        """ Bottleneck """
        self.b = ConvBlock2(1024, 256, batch_norm=batch_norm)
        """ Decoder """
        self.d1 = Decoder3(192, 128, batch_norm=batch_norm)
        self.d2 = Decoder3(96, 96, batch_norm=batch_norm)
        self.d3 = Decoder3(48, 48, batch_norm=batch_norm)
        self.d4 = Decoder3(24, 24, batch_norm=batch_norm)
        self.d5 = Decoder3(12, 6, batch_norm=batch_norm)
        self.outputs = nn.Conv2d(6, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        s1l, p1l = self.e1l(x[:,:int(self.in_channels/2),:,:])
        s1r, p1r = self.e1r(x[:,int(self.in_channels/2):,:,:])
        s2l, p2l = self.e2l(p1l)
        s2r, p2r = self.e2r(p1r)
        s3l, p3l = self.e3l(p2l)
        s3r, p3r = self.e3r(p2r)
        s4l, p4l = self.e4l(p3l)
        s4r, p4r = self.e4r(p3r)
        s5l, p5l = self.e5l(p4l)
        s5r, p5r = self.e5r(p4r)
        """ Bottleneck """
        b = self.b(torch.cat((p5l, p5r), dim=1))
        """ Decoder """
        d1 = self.d1(b, s5l)
        d2 = self.d2(d1, s4l)
        d3 = self.d3(d2, s3l)
        d4 = self.d4(d3, s2l)
        d5 = self.d5(d4, s1l)
        """ Classifier """
        outputs = self.outputs(d5)
        return outputs