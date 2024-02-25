from models import register, make
import torch.nn as nn
import torch
from .unet_cbam import SpatialGate, ResBlock
from patchify import patchify
from PIL import Image
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

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


class Encoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, batch_norm=batch_norm)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, batch_norm=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c, batch_norm=batch_norm)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


@register('unet-small-full-img')
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()

        self.patch_enc = Encoder(in_channels+2, 1)
        """ Encoder """
        self.e1 = Encoder(1, 8, batch_norm=batch_norm)
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

    def forward_func(self, inputs):
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
    
    def patchify_img(self, image, patch_size=256):
        
        size_x = (image.shape[0] // patch_size) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        img_size = list(image.size())
        img_size[1] = 1
        patches =  image.unfold(2,patch_size, patch_size).unfold(3,patch_size, patch_size)
        target_size = list(patches.size())
        target_size[1] = 1
        patches = patches.reshape(image.shape[0], image.shape[1], -1, patch_size, patch_size)
        instances = []

        for i in range(patches.shape[2]):
            instances.append(self.forward_func(patches[:,:,i,:,:]).unsqueeze(2)+patches[:,:,i,:,:].unsqueeze(2))          
        #
        out =  torch.cat(instances, dim=2).reshape(target_size).permute(0,1,2,4,3,5).contiguous().view(img_size)
        #Image.fromarray(out[0,0].cpu().detach().numpy(),'L').save('out.png')
        return out
    
    def forward(self, inputs):
        nx, ny = inputs.shape[-2:]
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        xv, yv = torch.meshgrid(x, y)
        pos = torch.cat([xv.unsqueeze(0), yv.unsqueeze(0)], dim=0).unsqueeze(0).repeat(inputs.shape[0],1,1,1).cuda()
        
        inp, _ = self.patch_enc(torch.cat([inputs, pos], dim=1))
        #print(inp.shape) [B X 4 X 2160 X 4096]

        res =  self.patchify_img(inp, patch_size=512)
        return res

        
