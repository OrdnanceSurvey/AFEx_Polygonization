# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:51:29 2019

@author: Jrainbow
"""

from skimage.io import imread, imsave
import matplotlib.pyplot as plt

mask = imread(r"O:\tiled\GT2031_HACK_RGB400_PADDED\roads_masks\GT2031_normalized_0.tif", as_gray=True)
h, w = mask.shape

cropped = mask[200: -200, 200: -200]

plt.imshow(cropped)

imsave(r"O:\tiled\GT2031_HACK_RGB400_PADDED\roads_masks\GT2031_normalized_0_cropped.tif", cropped)



