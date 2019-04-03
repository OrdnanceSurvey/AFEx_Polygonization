# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:44:31 2019

@author: Jrainbow
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import (label, regionprops, find_contours)
#from skimage.measure import regionprops
from skimage.io import imread
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage import feature
from scipy import ndimage
#import skimage

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

segment = '6_8'

img_path = rf"O:\tiled\SJ7394\buildings_masks\buildings_{segment}.tif"
rgb_path = rf"O:\tiled\SJ7394\{segment}.tif"

img = cv2.imread(img_path, 0)
bgr = cv2.imread(rgb_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#padded = np.pad(img, 1, pad_with)

padded = img

#cv2.imshow("original",img)
#cv2.waitKey(0)
labelled_img = label(padded)
#plt.imshow(img)
#plt.imshow(labelled_img)

numBuildings = len(np.unique(labelled_img))
print(f'{numBuildings} buildings found')

blank = np.zeros_like(padded)
output = np.zeros_like(padded)


candidate = 6

props = regionprops(labelled_img)
prop = props[candidate]
region = np.where(labelled_img == candidate)
blank[region] = 1


# draw contours
contours = find_contours(blank, 0.8)

fig, ax = plt.subplots(1,1, figsize=(15,15))
ax.imshow(rgb)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=3)
    ax.set_axis_off()
#fig.savefig('images/building_outline_1.png', bbox_inches='tight', pad_inches=0)    

