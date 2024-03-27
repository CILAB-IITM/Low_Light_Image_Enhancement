from datasets import register
import random
from torchvision.transforms import Resize
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset
from kornia.enhance import equalize, normalize
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_value)
random.seed(seed_value)


@register("preprocessing")
class Preprocessing(Dataset):
    def __init__(
        self,
        dataset,
        resize=False,
        patch=False,
        random_crop=False,
        amp=None,
        psize=[512, 512],
        augment_rotate=False,
        normalise=False,
        equalize=False,
        monochrome=False,
        stereo=False,
        add_noise=False,
        raw = False, 
        flip = False, 
        rsize = [384,1248], 
        hetero = True
    ):
        self.dataset = dataset
        self.resize = resize
        self.patch = patch
        self.psize = psize
        self.random_crop = random_crop
        self.amp = amp
        self.hetero = hetero
        if self.hetero == False:
            self.pcount = dataset[0][0].shape[0]
        else:
            self.pcount = dataset[0][0][0].shape[0]
        self.augment_rotate = augment_rotate
        self.normalise = normalise
        self.equalize = equalize
        self.monochrome = monochrome
        self.stereo = stereo
        self.add_noise = add_noise
        self.raw = raw
        self.flip = flip
        self.HFlip = RandomHorizontalFlip(p=1)
        self.VFlip = RandomVerticalFlip(p=1)
        self.stereo = stereo
        self.rsize = rsize
        # print(dataset[0][0].shape, 'dataset[0][0].shape from wrappers line 59')
        # print(self.rsize, self.resize, 'To check if resizing is happening from wrappers line 59')
        # print(len(dataset), 'Length of Dataset from Wrapper line 50')

        # print(self.pcount, "probably the shape of the input") #!

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.raw and self.stereo:
            # print(self.dataset, 'dataset')
            # print(len(self.dataset[idx]), 'self.dataset[idx]) from wrappers line 71')
            x = self.dataset[idx // self.pcount][0]
            # print(idx % self.pcount, idx // self.pcount, idx)
            # print(idx % self.pcount)
            # print('Hi')
            # print(self.dataset[idx // self.pcount][1][idx % self.pcount].shape, 'y Shape')
            # print(self.dataset[idx // self.pcount][1][1].shape, self.dataset[idx // self.pcount][1][0].shape,'potenitally the dim of the output')
            # print(self.dataset[idx // self.pcount][1].shape, len(self.dataset), 'outputshape')
            # print(self.dataset[idx // self.pcount][1].shape, len(self.dataset), 'outputshape 2')
            # print('hi')
            # print(len(self.dataset[idx // self.pcount]), 'tuple', self.pcount, 'pcount')
            y = self.dataset[idx // self.pcount][1][0]
            amp = self.dataset[idx // self.pcount][2]
            # print('done')
            # y = self.dataset[idx // self.pcount][1]
            # print('processing of the dataset')
            # x = torch.zeros([2, 2160, 4096])
            # print('bye')
            # y = torch.zeros([3, 2160, 4096])
            # print('bye 2')
            # print(y.shape, 'shape of y')
        else:
            x = self.dataset[idx // self.pcount][0][0]
            y = self.dataset[idx // self.pcount][1][idx % self.pcount]
             #!
            # print(x.shape, y.shape, 'input and output shape from wrapper line 102')
            # print(amp, 'amp from wrapper line 97') 
        if self.amp is not None:
            if self.amp == "cstm":
                # Will not work for Stereo 
                H = x.shape[1]
                W = x.shape[2]
                sumOfPixels = torch.sum(x)
                amp = 0.5 * (3 * H * W) / (sumOfPixels)  #! m = 0.5
                x = (x * amp).clamp_(0, 1)
            elif self.amp:
                if self.amp != True:
                    print('Wrong')
                    x = (x * self.amp).clamp_(0, 1)
                else:
                    amp = self.dataset[idx // self.pcount][0][1]
                    x = (x * amp).clamp_(0, 1)

        if self.patch:
            if self.raw and not self.stereo:
                rnum_h = random.randint(0, x.shape[-2] - self.psize[0])
                rnum_w = random.randint(0, x.shape[-1] - self.psize[1])
                if rnum_h % 2 != 0:
                    rnum_h += 1
                if rnum_w % 2 != 0:
                    rnum_w += 1
                new_x = x[:,
                    rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
                ]
                raw = new_x[0]
                raw_h, raw_w = raw.shape
                r = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g1 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g2 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                b = torch.zeros((raw_h // 2, raw_w // 2, 1))

                #! Here is where prolly we need to do the unpacking. Lets seee
                r = raw[0::2, 0::2]  # r
                g1 = raw[0::2, 1::2]  # gr
                g2 = raw[1::2, 0::2]  # gb
                b = raw[1::2, 1::2]  # b
                new_x = torch.stack((r, g1, g2, b))
                # img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)

                new_y = y[
                    :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
                ]
                
                

            elif self.raw and self.stereo:
                # print(x.shape, 'Input and Output Size from Wrapper line 105')
                rnum_h = random.randint(0, x.shape[1] - self.psize[0])
                rnum_w = random.randint(0, x.shape[-1] - self.psize[1])
                # print(rnum_h, rnum_w, 'The Random Numbers Generated', 'from Wrappers line 130')
                if rnum_h % 2 != 0:
                    rnum_h += 1
                if rnum_w % 2 != 0:
                    rnum_w += 1
                new_x = x[:, 
                    rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
                ]
                # print(new_x.shape, 'Patch Shape from wrapper line 146')
                # Can try refactoring the below piece of code
                raw_left = new_x[0]
                raw_h, raw_w = raw_left.shape
                r = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g1 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g2 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                b = torch.zeros((raw_h // 2, raw_w // 2, 1))

                r = raw_left[0::2, 0::2]  # r
                g1 = raw_left[0::2, 1::2]  # gr
                g2 = raw_left[1::2, 0::2]  # gb
                b = raw_left[1::2, 1::2]  # b
                new_x_left = torch.stack((r, g1, g2, b))
                
                raw_right = new_x[1]
                raw_h, raw_w = raw_right.shape
                r = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g1 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g2 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                b = torch.zeros((raw_h // 2, raw_w // 2, 1))

                r = raw_right[0::2, 0::2]  # r
                g1 = raw_right[0::2, 1::2]  # gr
                g2 = raw_right[1::2, 0::2]  # gb
                b = raw_right[1::2, 1::2]  # b
                new_x_right = torch.stack((r, g1, g2, b))
                new_x = torch.cat((new_x_left, new_x_right), dim=0)
                # img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)

                new_y = y[
                    :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
                ]
                # print(new_y.shape, 'jai baahubali')


            else:
                rnum_h = random.randint(0, x.shape[-2] - self.psize[0])
                rnum_w = random.randint(0, x.shape[-1] - self.psize[1])
                new_x = x[
                    :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
                ]
                new_y = y[:, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]]
                # print(new_x.shape, new_y.shape, 'From Wrapper line 135')
               
        # if self.patch:
        #     rnum_h = random.randint(0, x.shape[-2] - self.psize[0])
        #     rnum_w = random.randint(0, x.shape[-1] - self.psize[1])
        #     if self.raw:
        #         print('Confirm RAW is happening')
        #         new_x = x[
        #             :, rnum_h : rnum_h + self.psize[0]//2, rnum_w : rnum_w + self.psize[1]//2
        #         ]
                
        #         new_y = y[
        #             :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
        #         ]
        #     else:
        #         new_x = x[
        #             :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
        #         ]
        #         new_y = y[
        #             :, rnum_h : rnum_h + self.psize[0], rnum_w : rnum_w + self.psize[1]
        #     ]
        else:
            # print('Patching not done from wrapper line 215')
            if self.raw and self.stereo:
                raw_left = x[0]
                raw_h, raw_w = raw_left.shape
                r = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g1 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g2 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                b = torch.zeros((raw_h // 2, raw_w // 2, 1))

                r = raw_left[0::2, 0::2]  # r
                g1 = raw_left[0::2, 1::2]  # gr
                g2 = raw_left[1::2, 0::2]  # gb
                b = raw_left[1::2, 1::2]  # b
                new_x_left = torch.stack((r, g1, g2, b))
                
                raw_right = x[1]
                raw_h, raw_w = raw_right.shape
                r = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g1 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                g2 = torch.zeros((raw_h // 2, raw_w // 2, 1))
                b = torch.zeros((raw_h // 2, raw_w // 2, 1))

                r = raw_right[0::2, 0::2]  # r
                g1 = raw_right[0::2, 1::2]  # gr
                g2 = raw_right[1::2, 0::2]  # gb
                b = raw_right[1::2, 1::2]  # b
                new_x_right = torch.stack((r, g1, g2, b))
                new_x = torch.cat((new_x_left, new_x_right), dim=0)
                new_y = y
            elif self.raw and not self.stereo:
                pass
            else:
                new_x = x
                new_y = y

        if self.resize:
            # print(x.shape, 'size before resizing')
            # print('resize')

            resizer = Resize(self.rsize)
            new_x = resizer(new_x)
            new_y = new_y
            # print(new_x.shape, 'After Resizing from Wrappers line 221')
            # new_y = resizer(y) # Changed for the new loss function
        else:
            # print('Yes patching is not happening', 'from Wrapper Line 217')
            new_x = new_x
            new_y = new_y

        if self.add_noise:
            new_x += torch.abs(torch.randn(new_x.size()) * 0.005)
            new_x = new_x.clamp_(0, 1)

        if self.augment_rotate and random.random() > 0.5:
            angle = random.randint(-30, 30)
            new_x = rotate(new_x, angle)
            new_y = rotate(new_y, angle)
        
        if self.flip and random.random() > 0.01:
            print('Flipping happening from wrappers line 286')
            if random.random() > 0.5: # Horizontal Flip
            #     temp_x = new_x[:,::-1,:].copy()
            #     temp_y = new_y[:,::-1,:].copy() 
                new_x = self.HFlip(new_x)
                new_y = self.HFlip(new_y)
            #     new_y = temp_y
            else: # vertical flip
                new_x = self.VFlip(new_x)
                new_y = self.VFlip(new_y)
            #     temp_x = new_x[::-1,:,:].copy()
            #     temp_y = new_y[::-1,:,:].copy()
            #     new_x = temp_x
            #     new_y = temp_y
        if self.monochrome and self.stereo:
            new_x = new_x[[0, 3], :, :]
            new_y = new_y[:1, :, :]

        elif self.monochrome:
            new_x = new_x[:1, :, :]
            new_y = new_y[:1, :, :]

        if self.normalise:
            mean = torch.mean(new_x, dim=[1, 2]).unsqueeze(1)
            std = torch.std(new_x, dim=[1, 2]).unsqueeze(1)
            shape = new_x.shape
            new_x = ((new_x.reshape(new_x.shape[0], -1) - mean) / std).view(shape)

        if self.equalize:
            new_x = equalize(new_x)
        # print(new_x.shape, new_y.shape, 'Input and output shape after This is from Wrapper line 297')
        return {"inp": new_x, "out": new_y}
