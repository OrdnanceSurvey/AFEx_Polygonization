# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:01:19 2019

@author: Jrainbow
"""
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import polygon_perimeter, line
###############################################################################
# HELPER FUNCTIONS
###############################################################################
def find_terminal_vertices(arr):
    start_vertex = []
    end_vertex = []
    for i in range(1, len(arr)):
        end_x, start_x = arr[i, 1], arr[i - 1, 1]
        end_y, start_y = arr[i, 0], arr[i - 1, 0]
        start = (start_x, start_y)
        end = (end_x, end_y)
        start_vertex.append(start)
        end_vertex.append(end)
    return start_vertex, end_vertex

def calc_lengths(arr):
    lengths = []
    for i in range(1, len(arr)):
        dx = arr[i, 1] - arr[i - 1, 1]
        dy = arr[i, 0] - arr[i - 1, 0]
        length = np.sqrt(dx ** 2 + dy ** 2)
        lengths.append(length)
    return lengths

def calc_angles(arr):
    angles = []
    for i in range(1, len(arr)):
        dx = arr[i, 1] - arr[i - 1, 1]
        dy = arr[i, 0] - arr[i - 1, 0]
        angle = np.arctan2(dy, dx)
        angle *= (180 / np.pi)
        angles.append(angle)
    return angles

def calc_grads(arr):
    grads = []
    for i in range(1, len(arr)):
        dx = arr[i, 1] - arr[i - 1, 1]
        dy = arr[i, 0] - arr[i - 1, 0]
        grad = - dy / dx
        grads.append(grad)
    return grads

def optimize_angles(arr):
    angles = calc_angles(arr)
    for i, a in enumerate(angles):
        if a < 0:
            angles[i] = [angles[i] + 180, angles[i] + 360]
        elif a > 180:
            angles[i] = [angles[i], angles[i] - 180]
        else:
            angles[i] = [angles[i], angles[i] + 180]
    return angles