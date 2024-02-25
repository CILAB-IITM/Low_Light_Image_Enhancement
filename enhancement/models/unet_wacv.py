import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from models import register, make

def conv_layer(inc, outc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', relu_before=True,relu_after=False, weight_normalization = True):

    layers = []
    
    if relu_before:
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))

    m = nn.Conv2d(in_channels=inc, out_channels=outc,
    kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=groups, bias=bias, stride=stride)
    init_gain = 0.02
    if init_type == 'normal':
        torch.nn.init.normal_(m.weight, 0.0, init_gain)
    elif init_type == 'xavier':
        torch.nn.init.xavier_normal_(m.weight, gain = init_gain)
    elif init_type == 'kaiming':
        torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
    elif init_type == 'orthogonal':
        torch.nn.init.orthogonal_(m.weight)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)

    if bn:
        m = nn.BatchNorm2d(outc) # check outc
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)
    
    if relu_after:
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            
    return nn.Sequential(*layers)

class ResBlock(nn.Module):    
    def __init__(self,inc,midc,bn, weight_normalization):
        super(ResBlock, self).__init__()
                
        self.conv1 = conv_layer(inc, midc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=0.2, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=True, weight_normalization = weight_normalization)
        
        self.conv2 = conv_layer(midc, inc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=1.0, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=False, weight_normalization = weight_normalization)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


@register('unet-wacv')
class Net(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1 = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96, 96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        #self.dec4 = nn.Sequential(
        #                conv_layer(inc=48, outc=48, relu_before=False,relu_after=True), 
        #                self.pixelup2
        #                )

        #self.dec2 = nn.Sequential(
        #                conv_layer(inc=24, outc=12, relu_before=False,relu_after=True), 
        #                self.nearesrtup2
        #                )

        #self.dec1 = nn.Sequential(
        #                conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0), 
        #                conv_layer(inc=36, outc=3, relu_before=False,relu_after=False, negative_slope=1.0), nn.Sigmoid()
        #                )

        #modification
        self.dec4 = nn.Sequential(
                        conv_layer(inc=72, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=36, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=24, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1(e0l), self.enc1(e0r)
        e2l, e2r = self.enc2(e1l), self.enc2(e1r)
        e4l, e4r = self.enc4(e2l), self.enc4(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8 = self.dec8(torch.cat((e8,d16),dim=1))
        #
        #d4l, d4r = self.dec4(torch.cat((d8,e4l),dim=1)), self.dec4(torch.cat((d8,e4r),dim=1))
        #d2l, d2r = self.dec2(torch.cat((d4l,e2l),dim=1)), self.dec2(torch.cat((d4r,e2r),dim=1))
        #return self.dec1(torch.cat((d2l,e1l),dim=1)), self.dec1(torch.cat((d2r,e1r),dim=1))

        #modification
        d4 = self.dec4(torch.cat((d8, e4l, e4r), dim=1))
        d2 = self.dec2(torch.cat((d4, e2l, e2r), dim=1))

        return self.dec1(torch.cat((d2, e1l, e1r), dim=1))



@register('unet-wacv-2')
class UNet2(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1 = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1(e0l), self.enc1(e0r)
        e2l, e2r = self.enc2(e1l), self.enc2(e1r)
        e4l, e4r = self.enc4(e2l), self.enc4(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8 = self.dec8(torch.cat((e8,d16),dim=1))
        d4l= self.dec4(torch.cat((d8,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))


@register('unet-wacv-scaled-skip')
class UNet2(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1 = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1(e0l), self.enc1(e0r)
        e2l, e2r = self.enc2(e1l), self.enc2(e1r)
        e4l, e4r = self.enc4(e2l), self.enc4(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16), 2*e16),dim=1))
        d8 = self.dec8(torch.cat((d16,2*e8),dim=1))
        d4l= self.dec4(torch.cat((d8,2*e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,2*e2l),dim=1))
        return self.dec1(torch.cat((d2l,2*e1l),dim=1))
    

@register('unet-wacv-double-skip')
class UNet2(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1 = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1(e0l), self.enc1(e0r)
        e2l, e2r = self.enc2(e1l), self.enc2(e1r)
        e4l, e4r = self.enc4(e2l), self.enc4(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d16_ = self.dec16(torch.cat((e16,e16),dim=1))
        d8 = self.dec8(torch.cat((d16+d16_,e8),dim=1))
        d8_ = self.dec8(torch.cat((e8,e8),dim=1))
        d4l= self.dec4(torch.cat((d8+d8_,e4l),dim=1))
        d4l_ = self.dec4(torch.cat((e4r,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l+d4l_,e2l),dim=1))
        d2l_ = self.dec2(torch.cat((e2r,e2l),dim=1))
        return self.dec1(torch.cat((d2l+d2l_,e1l),dim=1))
    

@register('unet-wacv-small-sep')
class UNet2(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc1r = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8 = self.dec8(torch.cat((e8,d16),dim=1))
        d4l= self.dec4(torch.cat((d8,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))


@register('unet-wacv-small-sep-deep')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=512, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96, 96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )
        
        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8l, e8r = self.enc8l(e4l), self.enc8r(e4r)
        #e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(torch.cat((e8l,e8r), dim=1))
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8l = self.dec8(torch.cat((d16,e8l),dim=1))
        d4l= self.dec4(torch.cat((d8l,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))
    

@register('unet-wacv-small-sep-deep-cross-attention')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=512, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        conv_layer(inc=128, outc=128, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=128, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96, 96, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )
        
        self.dec2 = nn.Sequential(
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )

        self.scale1 = nn.Parameter(torch.rand(1))
        self.scale2 = nn.Parameter(torch.rand(1))
        self.scale3 = nn.Parameter(torch.rand(1))
        self.scale4 = nn.Parameter(torch.rand(1))
        self.scale5 = nn.Parameter(torch.rand(1))
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e0l += self.scale1*e0r
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e1l += self.scale2*e1r
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e2l += self.scale3*e2r
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e4l += self.scale4*e4r
        e8l, e8r = self.enc8l(e4l), self.enc8r(e4r)
        e8l += self.scale5*e8r
        #e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(torch.cat((e8l,e8r), dim=1))
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8l = self.dec8(torch.cat((d16,e8l),dim=1))
        d4l= self.dec4(torch.cat((d8l,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))






















































@register('unet-wacv-big')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1 = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=768, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=1024, outc=1024, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc32 = nn.Sequential(
                        conv_layer(inc=1024, outc=512, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=512, outc=1024, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=1280, outc=512, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(384, 384, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=192, outc=192, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        conv_layer(inc=96, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=120, outc=120, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=120, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1(e0l), self.enc1(e0r)
        e2l, e2r = self.enc2(e1l), self.enc2(e1r)
        e4l, e4r = self.enc4(e2l), self.enc4(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8 = self.dec8(torch.cat((e8,d16),dim=1))
        d4l= self.dec4(torch.cat((d8,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))
    

@register('unet-wacv-big-sep')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=768, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=1024, outc=512, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        conv_layer(inc=512, outc=512, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=512, outc=1024, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=768, outc=512, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(384, 384, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=192, outc=192, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )
        
        self.dec2 = nn.Sequential(
                        conv_layer(inc=96, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=120, outc=120, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=120, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(e8)
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8 = self.dec8(torch.cat((e8,d16),dim=1))
        d4l= self.dec4(torch.cat((d8,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))



@register('unet-wacv-big-sep-2')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=384, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=384, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=2048, outc=512, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        conv_layer(inc=512, outc=512, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=512, outc=1024, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=768, outc=512, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(384, 384, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=192, outc=192, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )
        
        self.dec2 = nn.Sequential(
                        conv_layer(inc=96, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=120, outc=120, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=120, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8l, e8r = self.enc8l(e4l), self.enc8r(e4r)
        #e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(torch.cat((e8l,e8r), dim=1))
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8l = self.dec8(torch.cat((d16,e8l),dim=1))
        d4l= self.dec4(torch.cat((d8l,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))


@register('unet-wacv-big-sep-3')
class UNetBig(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=24, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=96, outc=48, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=192, outc=96, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=384, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=384, outc=256, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16 = nn.Sequential(        
                        self.pixeldown2,
                        conv_layer(inc=2048, outc=512, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        conv_layer(inc=512, outc=512, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=512, outc=1024, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        conv_layer(inc=768, outc=512, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(640, 384, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        conv_layer(inc=352, outc=192, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )
        
        self.dec2 = nn.Sequential(
                        conv_layer(inc=144, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(
                        conv_layer(inc=144, outc=120, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=120, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8l, e8r = self.enc8l(e4l), self.enc8r(e4r)
        #e8 = self.enc8(torch.cat((e4l,e4r),dim=1))
        e16 = self.enc16(torch.cat((e8l,e8r), dim=1))
        d16 = self.dec16(torch.cat((self.enc32(e16),e16),dim=1))
        d8l = self.dec8(torch.cat((d16,e8l,e8r),dim=1))
        d4l= self.dec4(torch.cat((d8l,e4l,e4r),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l,e2r),dim=1))
        return self.dec1(torch.cat((d2l,e1l,e1r),dim=1))
    

@register('unet-wacv-sep-resblock')
class UNetRes(nn.Module):
    
    def __init__(self, in_channels=6, out_channels=3, bn=False, weight_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixelup2 = nn.PixelShuffle(2)
        self.pixeldown2 = nn.PixelUnshuffle(2)
        self.nearesrtup2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.enc1l = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(24,24, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(24,24, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )

        self.enc16r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm)
                        )
        
        self.enc32 = nn.Sequential(
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=256, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        ResBlock(192,192, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=96, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm),
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        ResBlock(24,24, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        self.nearesrtup2
                        )

        self.dec1 = nn.Sequential(

                        ResBlock(18,18, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=18, outc=36, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=36, outc=out_channels, relu_before=False,relu_after=False, negative_slope=1.0, bn=bn, weight_normalization=weight_norm)
                        )
    
    def forward(self,x):
        e0l, e0r = x[:,:int(self.in_channels/2),:,:], x[:,int(self.in_channels/2):,:,:]
        e1l, e1r = self.enc1l(e0l), self.enc1r(e0r)
        e2l, e2r = self.enc2l(e1l), self.enc2r(e1r)
        e4l, e4r = self.enc4l(e2l), self.enc4r(e2r)
        e8l, e8r = self.enc8l(e4l), self.enc8r(e4r)
        e16l, e16r = self.enc16l(e8l), self.enc16r(e8r)
        e32 = self.enc32(torch.cat((e16l,e16r),dim=1))

        d16l = self.dec16(torch.cat((e32,e16l),dim=1))
        d8l = self.dec8(torch.cat((d16l,e8l),dim=1))
        d4l= self.dec4(torch.cat((d8l,e4l),dim=1))
        d2l = self.dec2(torch.cat((d4l,e2l),dim=1))
        return self.dec1(torch.cat((d2l,e1l),dim=1))