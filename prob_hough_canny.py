# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:02:08 2019

@author: Jrainbow
"""

import numpy as np
from numpy import load

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm


# Constructing test image
image = np.zeros((100, 100))
idx = np.arange(25, 75)
image[idx[::-1], idx] = 255
image[idx, idx] = 255

building = load(r'images\building.npy')

# Line finding using the Probabilistic Hough Transform
#image = data.camera()
image = building * 255
edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=5, line_length=5,
                                 line_gap=5)

edges = canny(building, 2, 1, 25)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
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