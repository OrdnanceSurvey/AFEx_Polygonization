# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:12:24 2019

@author: Jrainbow
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
    
# load contours
    
contours = np.load(r'images/contours1.npy')

hand = contours.astype(np.uint8)[0]

# subdivide polygon using 2nd degree B-Splines
new_hand = hand.copy()
for _ in range(5):
    new_hand = subdivide_polygon(new_hand, degree=7, preserve_ends=True)

# approximate subdivided polygon with Douglas-Peucker algorithm
appr_hand = approximate_polygon(new_hand, tolerance=5)

print("Number of coordinates:", len(hand), len(new_hand), len(appr_hand))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(30, 15))

ax1.plot(hand[:, 1], hand[:, 0])
ax1.plot(new_hand[:, 1], new_hand[:, 0])
ax2.plot(appr_hand[:, 1], appr_hand[:, 0])

appr_building = appr_hand

np.save(r'images/approx_buidling.npy', appr_building)

