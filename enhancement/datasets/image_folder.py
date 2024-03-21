import os
import imageio
from PIL import Image
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import utils
from datasets import register
import math
from patchify import patchify
from torchvision.transforms import functional as F
import cv2
from models.warp import Warp_image
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_value)

@register("image-folder")
class ImageFolder(Dataset):
    def __init__(
        self,
        root_path,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
        hetero = True
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size
        self.hetero = True
        #! Check here
        """
        The below piece of code needs to be understood 
        split_file -  #! What is this for? Could be that 
                   -  file path to the splitfile 
        
        
        """
        #
        if split_file is None:
            if isinstance(root_path, list):
                filenames = []
                for path in root_path:
                    filenames.extend(sorted(os.listdir(path)))
            else:
                if self.hetero == True:
                    filenames = sorted(os.listdir(root_path), key=lambda x: int(x.split('.')[0]))
                else:
                    print('This code needs to be completed')
                    pass
                # print(filenames, 'Output List of files from image_folder line 55')
                # print()
                # print(len(filenames), 'Output List of files from image_folder line 56')

            if split_key == "train":
                filenames = filenames[: math.ceil(len(filenames) * split_ratio)]
                # print(len(filenames), 'Length of filenames from image_folder line 58')
            elif split_key == "test":
                filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
        else:
            with open(split_file, "r") as f:
                filenames = json.load(f)[split_key]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            # print(file, 'Output file name from image_folder line 68')

            if cache == "none":
                self.files.append(file)

            elif cache == "memory":
                img_arr = Image.open(file)

                if self.patchify:
                    img_arr = self.patchify_img(img_arr, self.patch_size)
                    img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
                else:
                    img = (
                        torch.from_numpy(np.array(img_arr) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                self.files.append(img)

    def patchify_img(self, image, patch_size=256):
        # print('Out patchify starts', 'from image_folder line 86')
        size_x = (
            image.shape[0] // patch_size
        ) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        instances = []

        # Crop original image to size divisible by patch size from top left corner
        image = image[:size_x, :size_y, :]

        # Extract patches from each image, step=patch_size means no overlap
        patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        # iterate over vertical patch axis
        for j in range(patch_img.shape[0]):
            # iterate over horizontal patch axis
            for k in range(patch_img.shape[1]):
                # patches are located like a grid. use (j, k) indices to extract single patched image
                single_patch_img = patch_img[j, k]

                # Drop extra extra dimension from patchify
                instances.append(np.squeeze(single_patch_img))
        patches = np.vstack([np.expand_dims(x, 0) for x in instances])
        # print(patches.shape, 'Output Patch shape from image_folder line 109')
        return patches

    def crop_center(self, img, cropx, cropy):
        x, y, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[startx : startx + cropx, starty : starty + cropy, :]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "none":
            img_arr = imageio.imread(x)
            if len(img_arr.shape) == 3:
                pass
            elif len(img_arr.shape) == 2:
                # print('Some Big Error' , 'from image_folder line 121')
                img_arr = np.repeat(np.expand_dims(img_arr, axis=2), 3, axis=2)

            # img_arr = cv2.resize(
            #     img_arr, dsize=(4096, 2160), interpolation=cv2.INTER_LINEAR
            # )
            """
            if img_arr.shape[0] < 2160 and img_arr.shape[1] < 4096:
                img_arr = np.lib.pad(img_arr, (((2160-img_arr.shape[0])//2, 2160-img_arr.shape[0]-((2160-img_arr.shape[0])//2)), ((4096-img_arr.shape[1])//2, 4096-img_arr.shape[1]-((4096-img_arr.shape[1])//2)), (0, 0)), 'constant', constant_values=(0))
            
            elif img_arr.shape[0] > 2160 and img_arr.shape[1] < 4096:
                img_arr = np.lib.pad(img_arr, ((0,0), ((4096-img_arr.shape[1])//2, 4096-img_arr.shape[1]-((4096-img_arr.shape[1])//2)), (0, 0)), 'constant', constant_values=(0))
                img_arr = self.crop_center(img_arr, 2160, 4096)

            elif img_arr.shape[0] < 2160 and img_arr.shape[1] > 4096:
                img_arr = np.lib.pad(img_arr, (((2160-img_arr.shape[0])//2, 2160-img_arr.shape[0]-((2160-img_arr.shape[0])//2)), (0,0), (0, 0)), 'constant', constant_values=(0))
                img_arr = self.crop_center(img_arr, 2160, 4096)

            else:
                img_arr = self.crop_center(img_arr, 2160, 4096)
            """
            # img_arr = cv2.resize(img_arr, dsize=(2160, 4096), interpolation=cv2.INTER_CUBIC)

            if self.patchify:
                img_arr = self.patchify_img(img_arr, self.patch_size)
                img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
            else:
                img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)
            # print(img.shape, 'Processed Image from Image Folder line 158')
            return img

        elif self.cache == "memory":
            return x


@register("image-folder-basic-raw")
# The Below is used
class ImageFolderOutRAW(Dataset):
    def __init__(
        self,
        root_path,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
        hetero = True
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size
        self.hetero = hetero
        #! Check here
        """
        The below piece of code needs to be understood 
        split_file -  #! What is this for? Could be that 
                   -  file path to the splitfile 
        
        
        """
        print("ImageFolderOutRAW from image_folder line 195")
        if self.hetero == False:
            if split_file is None:
                if isinstance(root_path, list):
                    filenames = []
                    for path in root_path:
                        filenames.extend(sorted(os.listdir(path)))
                else:
                    filenames = sorted(os.listdir(root_path))

                if split_key == "train":
                    filenames = filenames[: math.ceil(len(filenames) * split_ratio)]
                elif split_key == "test":
                    filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
            else:
                with open(split_file, "r") as f:
                    filenames = json.load(f)[split_key]
        else:
            if split_file is None:
                filenames = sorted(os.listdir(root_path), key=lambda x: int(x.split('_')[0]))
                if split_key == "train":
                    filenames = filenames[:math.ceil(len(filenames) * split_ratio)]
                elif split_key == "test":
                    filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
            else:
                with open(split_file, "r") as f:
                    filenames = json.load(f)[split_key]
        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == "none":
                if self.hetero == True: 
                    img_name = os.path.basename(file)
                    amp = float(img_name.split('_')[-1].split('.')[0])
                    self.files.append([file, amp])
                else:
                    self.files.append(file)

            elif cache == "memory":
                img_arr = Image.open(file)

                if self.patchify:
                    img_arr = self.patchify_img(img_arr, self.patch_size)
                    img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
                else:
                    img = (
                        torch.from_numpy(np.array(img_arr) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                self.files.append(img)

    def patchify_img(self, image, patch_size=256):
        if len(image.shape) == 2:
            size_x = (
                image.shape[0] // patch_size
            ) * patch_size  # get width to nearest size divisible by patch size
            size_y = (image.shape[1] // patch_size) * patch_size
            instances = []

            image = image[:size_x, :size_y]

            """
            Prolly the unpacking needs to be done here?  
            """
            patch_img = patchify(image, (patch_size, patch_size), step=patch_size)


            for j in range(patch_img.shape[0]):
                # iterate over horizontal patch axis
                for k in range(patch_img.shape[1]):
                    # patches are located like a grid. use (j, k) indices to extract single patched image
                    # print('happening')
                    single_patch_img = patch_img[j, k]
                    raw = single_patch_img
                    raw_h, raw_w = raw.shape
                    r = np.zeros((raw_h // 2, raw_w // 2, 1))
                    g1 = np.zeros((raw_h // 2, raw_w // 2, 1))
                    g2 = np.zeros((raw_h // 2, raw_w // 2, 1))
                    b = np.zeros((raw_h // 2, raw_w // 2, 1))

                    #! Here is where prolly we need to do the unpacking. Lets seee
                    r = raw[0::2, 0::2]  # r
                    # print(r.shape, 'r.shape')
                    g1 = raw[0::2, 1::2]  # gr
                    g2 = raw[1::2, 0::2]  # gb
                    b = raw[1::2, 1::2]  # b
                    # print(x)
                    single_patch_img = np.dstack((r, g1, g2, b))
                    # print(single_patch_img.shape)
                    # Drop extra extra dimension from patchify
                    instances.append(np.squeeze(single_patch_img))
            return np.vstack([np.expand_dims(x, 0) for x in instances])



        else:
            size_x = (
                image.shape[0] // patch_size
            ) * patch_size  # get width to nearest size divisible by patch size
            size_y = (image.shape[1] // patch_size) * patch_size
            instances = []

            # Crop original image to size divisible by patch size from top left corner
            image = image[:size_x, :size_y, :]

            # Extract patches from each image, step=patch_size means no overlap
            patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

            # iterate over vertical patch axis
            for j in range(patch_img.shape[0]):
                # iterate over horizontal patch axis
                for k in range(patch_img.shape[1]):
                    # patches are located like a grid. use (j, k) indices to extract single patched image
                    single_patch_img = patch_img[j, k]

                    # Drop extra extra dimension from patchify
                    instances.append(np.squeeze(single_patch_img))
            return np.vstack([np.expand_dims(x, 0) for x in instances])

    def crop_center(self, img, cropx, cropy):
        x, y, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[startx : startx + cropx, starty : starty + cropy, :]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        # print(x, "x = self.files[idx n(self.files)]")
        if self.cache == "none":
            # print('Unpacking Started')
            #! This is where prolly the processing needs to be done
            if self.hetero == True:
                raw = cv2.imread(x[0], -1)
                amp = x[1]                
            else:
                x = cv2.imread(x, -1)
            # print('It comes here')
            img_arr = raw
            if self.patchify:
                img_arr = self.patchify_img(img_arr, self.patch_size)
                img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
            else:
                # raw_h, raw_w = raw.shape
                # r = np.zeros((raw_h // 2, raw_w // 2, 1))
                # g1 = np.zeros((raw_h // 2, raw_w // 2, 1))
                # g2 = np.zeros((raw_h // 2, raw_w // 2, 1))
                # b = np.zeros((raw_h // 2, raw_w // 2, 1))

                # #! Here is where prolly we need to do the unpacking. Lets seee
                # r = raw[0::2, 0::2]  # r
                # # print(r.shape, 'r.shape')
                # g1 = raw[0::2, 1::2]  # gr
                # g2 = raw[1::2, 0::2]  # gb
                # b = raw[1::2, 1::2]  # b
                # # print(x)
                # img_arr = np.dstack((r, g1, g2, b))
                # print(img_arr.shape)
                img = torch.from_numpy(img_arr / 255).unsqueeze(0)
                # img = img_arr
                # print(img_arr.shape)
                # img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)
                # print(img_arr.shape)
                # print(img.shape, 'imgshape')
            # print('Successful')
            # print(raw.shape, 'Oh god plich try to be the same way')
            # raw = raw / 255.0
            # print(raw.shape, "raw_shape", x)
            # print(img.shape)
            # raw_h, raw_w = raw.shape
            # r = np.zeros((raw_h // 2, raw_w // 2, 1))
            # g1 = np.zeros((raw_h // 2, raw_w // 2, 1))
            # g2 = np.zeros((raw_h // 2, raw_w // 2, 1))
            # b = np.zeros((raw_h // 2, raw_w // 2, 1))

            # # r = (255.0 * r).astype(np.uint8)
            # # g1 = (255.0 * g1).astype(np.uint8)
            # # g2 = (255.0 * g2).astype(np.uint8)
            # # b = (255.0 * b).astype(np.uint8)

            # r = raw[0::2, 0::2]  # r
            # # print(r.shape, 'r.shape')
            # g1 = raw[0::2, 1::2]  # gr
            # g2 = raw[1::2, 0::2]  # gb
            # b = raw[1::2, 1::2]  # b
            # # print(x)
            # img_arr = np.dstack((r, g1, g2, b))
            # # cv2.imshow("helo", 10 * r)
            # # print(img_arr.shape, "After splitting into 4 channels")
            # # img_arr = imageio.imread(x)
            # if len(img_arr.shape) == 4:
            #     pass
            # elif len(img_arr.shape) == 2:
            #     img_arr = np.repeat(np.expand_dims(img_arr, axis=2), 3, axis=2)

            # img_arr = cv2.resize(
                # img_arr, dsize=(4096, 2160), interpolation=cv2.INTER_LINEAR
            # )
            """
            if img_arr.shape[0] < 2160 and img_arr.shape[1] < 4096:
                img_arr = np.lib.pad(img_arr, (((2160-img_arr.shape[0])//2, 2160-img_arr.shape[0]-((2160-img_arr.shape[0])//2)), ((4096-img_arr.shape[1])//2, 4096-img_arr.shape[1]-((4096-img_arr.shape[1])//2)), (0, 0)), 'constant', constant_values=(0))
            
            elif img_arr.shape[0] > 2160 and img_arr.shape[1] < 4096:
                img_arr = np.lib.pad(img_arr, ((0,0), ((4096-img_arr.shape[1])//2, 4096-img_arr.shape[1]-((4096-img_arr.shape[1])//2)), (0, 0)), 'constant', constant_values=(0))
                img_arr = self.crop_center(img_arr, 2160, 4096)

            elif img_arr.shape[0] < 2160 and img_arr.shape[1] > 4096:
                img_arr = np.lib.pad(img_arr, (((2160-img_arr.shape[0])//2, 2160-img_arr.shape[0]-((2160-img_arr.shape[0])//2)), (0,0), (0, 0)), 'constant', constant_values=(0))
                img_arr = self.crop_center(img_arr, 2160, 4096)

            else:
                img_arr = self.crop_center(img_arr, 2160, 4096)
            """
            # img_arr = cv2.resize(img_arr, dsize=(2160, 4096), interpolation=cv2.INTER_CUBIC)

            # if self.patchify:
            #     img_arr = self.patchify_img(img_arr, self.patch_size)
            #     img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
            # else:
            #     img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)
            return img, amp

        elif self.cache == "memory":
            return x


@register("image-folder-pairs")
class ImageFolder2(Dataset):
    def __init__(
        self,
        root_path1,
        root_path2,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size

        if split_file is None:
            filenames = sorted(os.listdir(root_path1), key=lambda x: int(x.split('_')[0]))
              # assuming both paths have same image names
            # print(filenames, 'Input List of files from image_folder line 428')
            # print()
            # print(len(filenames), 'Input List of files from image_folder line 429')
            # print()
            if split_key == "train":
                filenames = filenames[:math.ceil(len(filenames) * split_ratio)]
            elif split_key == "test":
                filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
        else:
            with open(split_file, "r") as f:
                filenames = json.load(f)[split_key]

        self.files = []
        for filename in filenames:
            file1 = os.path.join(root_path1, filename)
            file2 = os.path.join(root_path2, filename)

            if cache == "none":
                """
                Start
                
                """
                # print(file1, 'Left Image File name from image_folder line 445')
                img1_name = os.path.basename(file1)
                img2_name = os.path.basename(file2)
                # print(img1_name, 'File Name from image_folder line 447')
                amp = float(img1_name.split('_')[-1].split('.')[0])
                # print(self.amp, 'amp from image_folder line 449')
                # print(amp1-amp2, 'amplitude value from image_folder line 449')



                # print(img1_name, img2_name, 'Hopefully the file names from image_folder line 443')
                img1 = imageio.imread(file1)
                img2 = imageio.imread(file2)
                if self.patchify:
                    img_arr1 = self.patchify_img(img1, patch_size)
                    img_arr2 = self.patchify_img(img2, patch_size)
                    for i in range(len(img_arr1)):
                        self.files.append((img_arr1[i], img_arr2[i]))
                else:
                    # print(img1.shape, 'Left Image Shape from image_folder from line 449')
                    self.files.append((img1, img2,amp))

                # self.files.append([file1, file2]
            elif cache == "memory":
                print('from memory the files are being fetched', 'image_folder line 441')
                img_arr1 = imageio.imread(file1)
                img_arr2 = imageio.imread(file2)
                if len(img_arr1.shape) == 3:
                    pass
                elif len(img_arr1.shape) == 2:
                    img_arr1 = np.repeat(np.expand_dims(img_arr1, axis=2), 3, axis=2)

                if len(img_arr2.shape) == 3:
                    pass
                elif len(img_arr2.shape) == 2:
                    img_arr2 = np.repeat(np.expand_dims(img_arr2, axis=2), 3, axis=2)

                if self.patchify:
                    img_arr1 = self.patchify_img(img_arr1, patch_size)
                    img_arr2 = self.patchify_img(img_arr2, patch_size)
                    for i in range(len(img_arr1)):
                        self.files.append((img_arr1[i], img_arr2[i]))
                    print(len(self.files), 'Length of Files from image_folder line 458')
                    # potentially this needs to be changed for other setup of experiments
                    # img1 = torch.from_numpy(img_arr1 / 255).permute(0, 3, 1, 2)
                    # img2 = torch.from_numpy(img_arr2 / 255).permute(0, 3, 1, 2)
                else:
                    img1 = (
                        torch.from_numpy(np.array(img_arr1) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    img2 = (
                        torch.from_numpy(np.array(img_arr2) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    print(img1.shape, 'Shape of Left Image from image folder line 483')
                    self.files.append(torch.cat((img1, img2), axis=1))
        # print(len(self.files), 'Length of files from ImageFolder2 line 462')

    def patchify_img(self, image, patch_size=256):
        # print('Right Patchify Function is being used', 'From Image_folder line 471')
        size_x = (
            image.shape[0] // patch_size
        ) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        instances = []

        # Crop original image to size divisible by patch size from top left corner
        # image = image[:size_x, :size_y, ]
        image = image[:size_x, :size_y]

        # Extract patches from each image, step=patch_size means no overlap
        # patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patch_img = patchify(image, (patch_size, patch_size), step=patch_size)

        # iterate over vertical patch axis
        for j in range(patch_img.shape[0]):
            # iterate over horizontal patch axis
            for k in range(patch_img.shape[1]):
                # patches are located like a grid. use (j, k) indices to extract single patched image
                single_patch_img = patch_img[j, k]

                # Drop extra extra dimension from patchify
                instances.append(np.squeeze(single_patch_img))
        # patches = np.vstack([np.expand_dims(x, 0) for x in instances]) 
        # print(instances[0].shape, 'Shape of Instances[0] from line 498')
        # print('Patchify completed Successfully', 'From image_folder line 4
        # 94')
        return instances

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        # print(idx, 'idx printed from image_folder line 492')
        x = self.files[idx % len(self.files)]
        if self.cache == "none":
            # img_arr1 = imageio.imread(x[0])  # read the images from files
            # img_arr2 = imageio.imread(x[1])
            img1 = torch.from_numpy(x[0] / 255).unsqueeze(0) # Since it is raw
            img2 = torch.from_numpy(x[1] / 255).unsqueeze(0)
            img = torch.cat((img1, img2), dim=0)
            # print(x[2], 'amplification needed')
            # print('Jeichutom Maaara')

            # print(img_arr1.shape, img_arr2.shape, img_arr1.dtype, img_arr2.dtype, 'hola')
            # if len(img_arr1.shape) == 3:
            #     pass
            # elif len(img_arr1.shape) == 2:
            #     img_arr1 = np.repeat(np.expand_dims(img_arr1, axis=2), 3, axis=2)

            # if len(img_arr2.shape) == 3:
            #     pass
            # elif len(img_arr2.shape) == 2:
            #     img_arr2 = np.repeat(np.expand_dims(img_arr2, axis=2), 3, axis=2)

            # # img_arr1 = np.lib.pad(img_arr1, ((0, 2162-img_arr1.shape[0]), (0, 4186-img_arr1.shape[1]), (0, 0)), 'constant', constant_values=(0))
            # # img_arr2 = np.lib.pad(img_arr2, ((0, 2162-img_arr2.shape[0]), (0, 4186-img_arr2.shape[1]), (0, 0)), 'constant', constant_values=(0))
            # img_arr1 = cv2.resize(
            #     img_arr1, dsize=(1248, 384), interpolation=cv2.INTER_LINEAR
            # )
            # img_arr2 = cv2.resize(
            #     img_arr2, dsize=(1248, 384), interpolation=cv2.INTER_LINEAR
            # )

            # if self.patchify:
            #     img_arr1 = self.patchify_img(img_arr1, self.patch_size)
            #     img1 = torch.from_numpy(img_arr1 / 255).unsqueeze(0)
            #     img_arr2 = self.patchify_img(img_arr2, self.patch_size)
            #     img2 = torch.from_numpy(img_arr2 / 255).unsqueeze(0)
            #     img = torch.cat((img1, img2), dim=0)
            #     print(img.shape, 'Patch and Concatenated Image', 'From iamge_folder line 537')

        # else:
        #     img1 = torch.from_numpy(img_arr1 / 255).unsqueeze(0) # Since it is raw
        #     img2 = torch.from_numpy(img_arr2 / 255).unsqueeze(0)
        #     img = torch.cat((img1, img2), dim=0)
        #         # img1 = torch.from_numpy(img_arr1 / 255).permute(2, 0, 1).unsqueeze(0)
        #         # img2 = torch.from_numpy(img_arr2 / 255).permute(2, 0, 1).unsqueeze(0)
            # print(img.shape, 'input shape from image_folder2 line 528')
            # print(self.amp, 'self.amp from image_folder line 584')
            return (img, x[2])

        elif self.cache == "memory":
            return x


class ImageFolder3(Dataset):
    def __init__(
        self,
        root_path,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size

        if split_file is None:
            if isinstance(root_path, list):
                filenames = []
                for path in root_path:
                    filenames.extend(sorted(os.listdir(path)))
            else:
                filenames = sorted(os.listdir(root_path))

            if split_key == "train":
                filenames = filenames[: math.ceil(len(filenames) * split_ratio)]
            elif split_key == "test":
                filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
        else:
            with open(split_file, "r") as f:
                filenames = json.load(f)[split_key]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == "none":
                self.files.append(file)

            elif cache == "memory":
                img_arr = np.fromfile(open(file, "rb"), dtype=np.uint8).reshape(
                    2160, 4096
                )
                if self.patchify:
                    img_arr = self.patchify_img(img_arr, self.patch_size)
                    img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
                else:
                    img = (
                        torch.from_numpy(np.array(img_arr) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                self.files.append(img)

    def patchify_img(self, image, patch_size=256):
        size_x = (
            image.shape[0] // patch_size
        ) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        instances = []

        # Crop original image to size divisible by patch size from top left corner
        image = image[:size_x, :size_y, :]

        # Extract patches from each image, step=patch_size means no overlap
        patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        # iterate over vertical patch axis
        for j in range(patch_img.shape[0]):
            # iterate over horizontal patch axis
            for k in range(patch_img.shape[1]):
                # patches are located like a grid. use (j, k) indices to extract single patched image
                single_patch_img = patch_img[j, k]

                # Drop extra extra dimension from patchify
                instances.append(np.squeeze(single_patch_img))
        return np.vstack([np.expand_dims(x, 0) for x in instances])

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "none":
            # print(np.fromfile(open(x, "rb"), dtype=np.uint8).shape, "Raw Image Shape")
            img_arr = np.fromfile(open(x, "rb"), dtype=np.uint8).reshape(2160, 4096)

            if len(img_arr.shape) == 3:
                pass
            elif len(img_arr.shape) == 2:
                img_arr = np.repeat(np.expand_dims(img_arr, axis=2), 3, axis=2)

            # img_arr = np.lib.pad(img_arr, ((0, 2162-img_arr.shape[0]), (0, 4186-img_arr.shape[1]), (0, 0)), 'constant', constant_values=(0))
            Image.fromarray(img_arr).save("raw_img.png")
            if self.patchify:
                img_arr = self.patchify_img(img_arr, self.patch_size)
                img = torch.from_numpy(img_arr / 255).permute(0, 3, 1, 2)
            else:
                img = torch.from_numpy(img_arr / 255).permute(2, 0, 1).unsqueeze(0)

            return img

        elif self.cache == "memory":
            return x


class ImageFolder4(Dataset):
    def __init__(
        self,
        root_path1,
        root_path2,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size

        if split_file is None:
            if isinstance(root_path1, list):
                filenames = []
                for path in root_path1:
                    filenames.extend(sorted(os.listdir(path)))
                print(len(filenames))
            else:
                filenames = sorted(os.listdir(root_path1))

            if split_key == "train":
                filenames = filenames[: math.ceil(len(filenames) * split_ratio)]
            elif split_key == "test":
                filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
        else:
            with open(split_file, "r") as f:
                filenames = json.load(f)[split_key]

        self.files = []
        for filename in filenames:
            file1 = os.path.join(root_path1, filename)
            file2 = os.path.join(root_path2, filename)

            if cache == "none":
                self.files.append([file1, file2])

            elif cache == "memory":
                img_arr1 = np.fromfile(open(file1, "rb"), dtype=np.uint8).reshape(
                    2160, 4096
                )
                img_arr2 = np.fromfile(open(file2, "rb"), dtype=np.uint8).reshape(
                    2160, 4096
                )

                if self.patchify:
                    img_arr1 = self.patchify_img(img_arr1, self.patch_size)
                    img_arr2 = self.patchify_img(img_arr2, self.patch_size)
                    img1 = torch.from_numpy(img_arr1 / 255).permute(0, 3, 1, 2)
                    img2 = torch.from_numpy(img_arr2 / 255).permute(0, 3, 1, 2)
                else:
                    img1 = (
                        torch.from_numpy(np.array(img_arr1) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    img2 = (
                        torch.from_numpy(np.array(img_arr2) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )

                self.files.append(torch.cat((img1, img2), axis=1))

    def patchify_img(self, image, patch_size=256):
        size_x = (
            image.shape[0] // patch_size
        ) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        instances = []

        # Crop original image to size divisible by patch size from top left corner
        image = image[:size_x, :size_y, :]

        # Extract patches from each image, step=patch_size means no overlap
        patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        # iterate over vertical patch axis
        for j in range(patch_img.shape[0]):
            # iterate over horizontal patch axis
            for k in range(patch_img.shape[1]):
                # patches are located like a grid. use (j, k) indices to extract single patched image
                single_patch_img = patch_img[j, k]

                # Drop extra extra dimension from patchify
                instances.append(np.squeeze(single_patch_img))
        return np.vstack([np.expand_dims(x, 0) for x in instances])

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "none":
            img_arr1 = np.fromfile(open(x[0], "rb"), dtype=np.uint8).reshape(2160, 4096)
            img_arr2 = np.fromfile(open(x[1], "rb"), dtype=np.uint8).reshape(2160, 4096)

            if len(img_arr1.shape) == 3:
                pass
            elif len(img_arr1.shape) == 2:
                img_arr1 = np.repeat(np.expand_dims(img_arr1, axis=2), 3, axis=2)

            if len(img_arr2.shape) == 3:
                pass
            elif len(img_arr2.shape) == 2:
                img_arr2 = np.repeat(np.expand_dims(img_arr2, axis=2), 3, axis=2)
            # img_arr = np.lib.pad(img_arr, ((0, 2162-img_arr.shape[0]), (0, 4186-img_arr.shape[1]), (0, 0)), 'constant', constant_values=(0))

            if self.patchify:
                img_arr1 = self.patchify_img(img_arr1, self.patch_size)
                img1 = torch.from_numpy(img_arr1 / 255).permute(0, 3, 1, 2)
                img_arr2 = self.patchify_img(img_arr2, self.patch_size)
                img2 = torch.from_numpy(img_arr2 / 255).permute(0, 3, 1, 2)
            else:
                img1 = torch.from_numpy(img_arr1 / 255).permute(2, 0, 1).unsqueeze(0)
                img2 = torch.from_numpy(img_arr2 / 255).permute(2, 0, 1).unsqueeze(0)
            return torch.cat((img1, img2), axis=1)

        elif self.cache == "memory":
            return x


@register("image-folder-basic")
class ImageFolderBasic(Dataset):
    def __init__(self, root_path_inp, root_path_out, **kwargs):
        self.dataset1 = ImageFolder(root_path_inp, **kwargs)
        self.dataset2 = ImageFolder(root_path_out, **kwargs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]


@register("image-folder-pair") # This is where the processing is happening
class ImageFolderBasic(Dataset):
    def __init__(self, root_path_inp1, root_path_inp2, root_path_out, **kwargs):
        self.dataset1 = ImageFolder2(root_path_inp1, root_path_inp2, **kwargs)
        # print(len(self.dataset1), 'Length of Dataset', 'from image_folder line 805')
        self.dataset2 = ImageFolder(root_path_out, **kwargs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        x, amp = self.dataset1[idx]
        y = self.dataset2[idx]
        # print(amp, 'amp from image_folder line 855')
        # print(y.shape, 'output size from image folder')
        return x, y, amp


@register("image-folder-basic-raw")
class ImageFolderBasic(Dataset):
    def __init__(self, root_path_inp, root_path_out, **kwargs):
        # self.dataset1 = ImageFolder3(root_path_inp, **kwargs)
        # print("ImageFolderBasic Class, image_folder.py")  #!
        self.dataset1 = ImageFolderOutRAW(root_path_inp, **kwargs)
        self.dataset2 = ImageFolder(root_path_out, **kwargs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]


@register("image-folder-pair-raw")
class ImageFolderBasic(Dataset):
    def __init__(self, root_path_inp1, root_path_inp2, root_path_out, **kwargs):
        self.dataset1 = ImageFolder4(root_path_inp1, root_path_inp2, **kwargs)
        self.dataset2 = ImageFolder(root_path_out, **kwargs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]


class ImageFolder2Warp(Dataset):
    def __init__(
        self,
        root_path1,
        root_path2,
        disp_path,
        occ_path,
        use_gt=False,
        split_file=None,
        split_key=None,
        split_ratio=0.8,
        repeat=1,
        cache="none",
        patchify=False,
        patch_size=256,
    ):
        self.repeat = repeat
        self.cache = cache
        self.patchify = patchify
        self.patch_size = patch_size
        self.use_gt = use_gt

        if split_file is None:
            filenames = sorted(
                os.listdir(root_path1)
            )  # assuming both paths have same image names

            if split_key == "train":  # changes to be made
                filenames = filenames[: math.ceil(len(filenames) * split_ratio)]
            elif split_key == "test":
                filenames = filenames[math.ceil(len(filenames) * split_ratio) :]
        else:
            with open(split_file, "r") as f:
                filenames = json.load(f)[split_key]

        self.files = []
        for filename in filenames:
            file1 = os.path.join(root_path1, filename)
            file2 = os.path.join(root_path2, filename)
            disp_file = os.path.join(disp_path, filename.split(".")[0] + ".npy")
            occ_file = os.path.join(occ_path, filename.split(".")[0] + "_occ_mask.npy")

            if cache == "none":
                self.files.append([file1, file2, disp_file, occ_file])

            elif cache == "memory":
                img_arr1 = imageio.imread(file1)
                img_arr2 = imageio.imread(file2)
                if len(img_arr1.shape) == 3:
                    pass
                elif len(img_arr1.shape) == 2:
                    img_arr1 = np.repeat(np.expand_dims(img_arr1, axis=2), 3, axis=2)

                if len(img_arr2.shape) == 3:
                    pass
                elif len(img_arr2.shape) == 2:
                    img_arr2 = np.repeat(np.expand_dims(img_arr2, axis=2), 3, axis=2)

                if self.patchify:
                    img_arr1 = self.patchify_img(img_arr1, patch_size)
                    img_arr2 = self.patchify_img(img_arr2, patch_size)
                    img1 = torch.from_numpy(img_arr1 / 255).permute(0, 3, 1, 2)
                    img2 = torch.from_numpy(img_arr2 / 255).permute(0, 3, 1, 2)
                else:
                    img1 = (
                        torch.from_numpy(np.array(img_arr1) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    img2 = (
                        torch.from_numpy(np.array(img_arr2) / 255)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )

                # warping
                disp = np.load(disp_file)
                # print(disp.shape)
                self.files.append(torch.cat((img1, img2), axis=1))

    def patchify_img(self, image, patch_size=256):
        size_x = (
            image.shape[0] // patch_size
        ) * patch_size  # get width to nearest size divisible by patch size
        size_y = (image.shape[1] // patch_size) * patch_size
        instances = []

        # Crop original image to size divisible by patch size from top left corner
        image = image[:size_x, :size_y, :]

        # Extract patches from each image, step=patch_size means no overlap
        patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        # iterate over vertical patch axis
        for j in range(patch_img.shape[0]):
            # iterate over horizontal patch axis
            for k in range(patch_img.shape[1]):
                # patches are located like a grid. use (j, k) indices to extract single patched image
                single_patch_img = patch_img[j, k]

                # Drop extra extra dimension from patchify
                instances.append(np.squeeze(single_patch_img))
        return np.vstack([np.expand_dims(x, 0) for x in instances])

    def image_warp(self, right, disp):
        return Warp_image(right, -1 * disp, wrap_mode="edge")

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "none":
            img_arr1 = imageio.imread(x[0])
            img_arr2 = imageio.imread(x[1])
            disp = np.load(x[2])
            occ = np.load(x[3])

            if len(img_arr1.shape) == 3:
                pass
            elif len(img_arr1.shape) == 2:
                img_arr1 = np.repeat(np.expand_dims(img_arr1, axis=2), 3, axis=2)

            if len(img_arr2.shape) == 3:
                pass
            elif len(img_arr2.shape) == 2:
                img_arr2 = np.repeat(np.expand_dims(img_arr2, axis=2), 3, axis=2)

            if len(disp.shape) == 3:
                disp = disp[0]

            if len(occ.shape) == 3:
                occ = occ[0]

            # resizing
            disp = cv2.resize(
                (disp / 1248) * 4096, dsize=(4096, 2160), interpolation=cv2.INTER_CUBIC
            )
            occ = cv2.resize(occ, dsize=(4096, 2160), interpolation=cv2.INTER_NEAREST)

            # resizing original image to disparity dimensions

            # img_arr1 = np.lib.pad(img_arr1, ((0, 2162-img_arr1.shape[0]), (0, 4186-img_arr1.shape[1]), (0, 0)), 'constant', constant_values=(0))
            # img_arr2 = np.lib.pad(img_arr2, ((0, 2162-img_arr2.shape[0]), (0, 4186-img_arr2.shape[1]), (0, 0)), 'constant', constant_values=(0))

            img_arr1 = cv2.resize(
                img_arr1, dsize=(4096, 2160), interpolation=cv2.INTER_LINEAR
            )
            img_arr2 = cv2.resize(
                img_arr2, dsize=(4096, 2160), interpolation=cv2.INTER_LINEAR
            )

            # warping
            if self.use_gt:
                img_arr2 = (
                    self.image_warp(
                        torch.from_numpy(img_arr2).permute(2, 0, 1).unsqueeze(0),
                        torch.from_numpy(disp * occ),
                    )
                    .squeeze()
                    .permute(1, 2, 0)
                    .numpy()
                ) * np.expand_dims(occ, axis=-1)
                img_arr2 += img_arr2 * np.abs(np.expand_dims(occ, axis=-1) - 1)
            else:
                img_arr2 = (
                    self.image_warp(
                        torch.from_numpy(img_arr2).permute(2, 0, 1).unsqueeze(0),
                        torch.from_numpy(disp),
                    )
                    .squeeze()
                    .permute(1, 2, 0)
                    .numpy()
                )
                img_arr2 = img_arr2 * occ[:, :, None] + img_arr1 * (1 - occ[:, :, None])

            if self.patchify:
                img_arr1 = self.patchify_img(img_arr1, self.patch_size)
                img1 = torch.from_numpy(img_arr1 / 255).permute(0, 3, 1, 2)
                img_arr2 = self.patchify_img(img_arr2, self.patch_size)
                img2 = torch.from_numpy(img_arr2 / 255).permute(0, 3, 1, 2)
            else:
                img1 = torch.from_numpy(img_arr1 / 255).permute(2, 0, 1).unsqueeze(0)
                img2 = torch.from_numpy(img_arr2 / 255).permute(2, 0, 1).unsqueeze(0)

            return torch.cat((img1, img2), axis=1)

        elif self.cache == "memory":
            return x


@register("image-folder-warped-pair")
class ImageFolderBasic(Dataset):
    def __init__(
        self,
        root_path_inp1,
        root_path_inp2,
        root_path_out,
        disp_path,
        occ_path,
        use_gt,
        **kwargs
    ):
        self.dataset1 = ImageFolder2Warp(
            root_path_inp1, root_path_inp2, disp_path, occ_path, use_gt, **kwargs
        )
        self.dataset2 = ImageFolder(root_path_out, **kwargs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]
