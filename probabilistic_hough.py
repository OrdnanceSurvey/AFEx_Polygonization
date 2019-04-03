# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:00:00 2019

@author: Jrainbow
"""

import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
import skimage
from skimage.morphology import binary_dilation

img = skimage.io.imread('images/building_outline_1.png',0)
img = img[:,:,2]
# crop for now
img = img[50:-50,50:-50]
img[img < 100] = 0
img[img >= 100] = 255
dilated = binary_dilation(img)
image = binary_dilation(dilated)
fig, ax = plt.subplots(1,1, figsize=(15,15))
ax.imshow(image)
ax.set_axis_off()


lines = probabilistic_hough_line(image, threshold=10, line_length=5,
                                 line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(25, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(image * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

