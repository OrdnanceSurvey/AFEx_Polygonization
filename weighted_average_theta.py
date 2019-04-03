# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:02:48 2019

@author: Jrainbow
"""

import numpy as np
from numpy import load
import matplotlib.pyplot as plt
#import pandas as pd

corners = load(r'images/approx_building.npy')

def calc_thetas(arr):
    # CLEAN THIS UP IN FUTURE - e.g. pandas
    thetas = []
    dists = []
    for i in range(len(arr + 1)):
#        print(i)
        if i < len(arr) - 1:
            print(i)
            dx = arr[i, 1] - arr[i + 1, 1] 
            dy = arr[i, 0] - arr[i + 1, 0]
            theta = (180/np.pi) * np.arctan2(dy,dx)
            D = np.sqrt(dx ** 2 + dy ** 2)
            if theta < 0:
                theta += 180
            thetas.append(theta)
            dists.append(D)
    return thetas, dists

def calc_thetas_2(arr):
    # CLEAN THIS UP IN FUTURE - e.g. pandas
    thetas = []
    dists = []
    for i in range(len(arr + 1)):
#        print(i)
        if i < len(arr) - 1:
            print(i)
            dx = arr[i, 1] - arr[i + 1, 1] 
            dy = arr[i, 0] - arr[i + 1, 0]
            theta = (180/np.pi) * np.arctan2(dy,dx)
            D = np.sqrt(dx ** 2 + dy ** 2)
            if theta < 0:
                theta += 180
            thetas.append(theta)
            dists.append(D)
    return thetas, dists
    
    
def theta_av(thetas, dists):
    wth = 0
    for i in range(len(thetas)):
        wth += (dists[i] * thetas[i])
    result = wth / np.sum(dists)
    return result

def calc_intercept_not_working(m, x_0, x_1, y_0, y_1):
    numerator = (1 / (m + 1)) * ((x_0 + x_1 + m * (y_0 + y_1)) / (m * (m + 1)) - (x_0 + x_1)) - \
                (m / m ** 2) * ((m ** 2 * (y_0 + y_1) + m * (x_0 + x_1)) / (m ** 2 + 1) - (y_0 + y_1))
    denominator = m / (m ** 2 + 1) ** 2 + m / (m * (m + 1) ** 2)
    return numerator / denominator

def calc_intercept(x_0, y_0, x_1, y_1, m):
    c = (1 / 2) * (y_0 + y_1 - m * (x_0 + x_1))
    return c

    
        
        
intcorners = corners.astype(int)
blank=np.zeros((300,300))
#for i in range(len(intcorners)):
for i in range(6):
    print(i)
    blank[intcorners[i,0]][intcorners[i,1]] = 255

crop = blank[20:100,120:240]

fig, ax = plt.subplots(1,1, figsize=(20,20))
ax.imshow(crop)

#i = 4
#dx = intcorners[i + 1, 1] - intcorners[i, 1]
#dy = intcorners[i + 1, 0] - intcorners[i, 0]
#theta = (180/np.pi) * np.arctan2(dy,dx)
#length = np.sqrt(dx ** 2 + dy ** 2)
#print(theta, length)

th, di = calc_thetas(corners)

# NECESSARY TO ADD 90 DEGREES TO THE ONES THAT ARE ROUGHLY ORTHOGONAL. CAN FIND LONGEST LINE AND ADD 90 TO OTHERS UNTIL ROUGHLY ALL MODAL



