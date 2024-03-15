import numpy as np
import cv2

dep = np.load('/home/gpu/girish/dataset/Stereo_Depth_Estimation_Expts-depth_gt_share/ground_best_depths_mode/2.npy')

print(dep.shape)

img = cv2.imread('/home/gpu/girish/dataset/OUTDOOR_RGB/well_lit/png/cam1/hist/1.png')
print(img.shape)