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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class ResBlock(nn.Module):    
    def __init__(self,inc,midc,bn, weight_normalization):
        super(ResBlock, self).__init__()
                
        self.conv1 = conv_layer(inc, midc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=0.2, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=True, weight_normalization = weight_normalization)
        
        self.conv2 = conv_layer(midc, inc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=1.0, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=False, weight_normalization = weight_normalization)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x
    

class ResBlockSAM(nn.Module):    
    def __init__(self,inc,midc,bn, weight_normalization):
        super(ResBlockSAM, self).__init__()
                
        self.conv1 = conv_layer(inc, midc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=0.2, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=True, weight_normalization = weight_normalization)
        self.sg1 = SpatialGate()
        self.conv2 = conv_layer(midc, inc, kernel_size=3, stride=1, groups=1, bias=True, negative_slope=1.0, bn=bn, init_type='kaiming', fan_type='fan_in', relu_before=False,relu_after=False, weight_normalization = weight_normalization)
        self.sg2 = SpatialGate()

    def forward(self, x):

        return self.sg2(self.conv2(self.sg1(self.conv1(x)))) + x


@register('unet-wacv-sep-resblock-sam')
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
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )

        self.enc4l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )

        self.enc8l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )

        self.enc16l = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )
        
        self.enc1r = conv_layer(inc=int(in_channels/2), outc=6, relu_before=False,relu_after=False, bn=bn, weight_normalization=weight_norm)

        self.enc2r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(24,24, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=24, outc=12, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )

        self.enc4r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=48, outc=24, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )
        
        self.enc8r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=96, outc=64, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )

        self.enc16r = nn.Sequential(        
                        self.pixeldown2,
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=128, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        SpatialGate()
                        )
        
        self.enc32 = nn.Sequential(
                        ResBlock(256,256, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=256, stride=2, relu_before=True,relu_after=False, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=256, outc=256, relu_before=True,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        SpatialGate(),
                        self.pixelup2
                        )

        self.dec16 = nn.Sequential(
                        ResBlock(192,192, bn=bn, weight_normalization=weight_norm),
                        conv_layer(inc=192, outc=128, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        SpatialGate(),
                        self.pixelup2
                        )

        self.dec8 = nn.Sequential(
                        ResBlock(96,96, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=96, outc=96, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm),
                        SpatialGate(),
                        self.pixelup2
                        )

        self.dec4 = nn.Sequential(
                        ResBlock(48,48, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=48, outc=48, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        SpatialGate(),
                        self.pixelup2
                        )

        self.dec2 = nn.Sequential(
                        ResBlock(24,24, bn=bn, weight_normalization=weight_norm), 
                        conv_layer(inc=24, outc=12, relu_before=False,relu_after=True, bn=bn, weight_normalization=weight_norm), 
                        SpatialGate(),
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