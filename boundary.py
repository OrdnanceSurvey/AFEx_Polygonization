# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:12:57 2019

@author: Jrainbow
"""

import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

a = np.zeros((50,50))
a[10:30,10:30] = 1
a[35:45,35:45] = 2

distance = ndimage.distance_transform_edt(a)

distance[distance != 1] = 0
plt.imshow(distance)
plt.show()

np.where(distance == 1)