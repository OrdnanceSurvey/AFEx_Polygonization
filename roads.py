# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:09:13 2019

@author: Jrainbow
"""
import numpy as np
import os
from skimage.io import imread
import matplotlib.pyplot as plt

def fit_images(img1, img2):
    """
    Makes images the same size gt = img2, cf = img1
    """
    cf_h, cf_w = img1.shape
    gt_h, gt_w = img2.shape
    min_h, min_w = np.min([cf_h, gt_h]), np.min([cf_w, gt_w])
    smallest_raster = np.argmin([cf_h, gt_h])
    if smallest_raster:
        print("Second argument is smaller")
        img1 = img1[: min_h, : min_w]
    else:
        print("First argument is smaller")
        img2 = img2[: min_h, : min_w]
    
    return img1, img2

class RoadTile:
    
    ROOT_DIR = r"O:\tiled"
    APPENDAGE = 'HACK_RGB400_PADDED'
    EXTRAS = '_epoch_100'
    NORM = '0'
    
    def __init__(self, tile):
        self.tile = tile
        
    def get_mask_path(self):
        mask_path = os.path.join(RoadTile.ROOT_DIR, rf'{self.tile}_{RoadTile.APPENDAGE}',
                     rf'roads_masks{RoadTile.EXTRAS}',
                     rf'{self.tile}_normalized_{RoadTile.NORM}.tif')
        return mask_path
    
    def read_mask(self, margin=0, norm=False):
        mask_path = self.get_mask_path()
        mask = imread(mask_path, as_gray=True)[margin: -margin, margin: -margin]
        if norm == True:
            mask = mask / 255
        return mask, mask.shape
    
    def get_image_path(self):
        BASE_DIR = os.path.dirname(RoadTile.ROOT_DIR)
        image_path = os.path.join(BASE_DIR, 'Guyana',
                                  'imagery',
                                  rf'{self.tile}.tif')
        return image_path
    
    def read_image(self, as_gray=False):
        image_path = self.get_image_path()
        image = imread(image_path, as_gray=as_gray)
        return image, image.shape
    
    
def main():
    road = RoadTile('GT1033')
    mask, mask_shp = road.read_mask(margin=200, norm=True)
    img, img_shp = road.read_image(as_gray=True)
    if mask_shp == img_shp:
        print("Images the same size")
    else:
        print("Fitting images...")
        mask, img = fit_images(mask, img)
        
        
    fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(20,20))
    
    ax1.imshow(img, 'gray', interpolation='none')
    ax1.imshow(mask, 'jet', interpolation='none', alpha=0.5)


main()    
        
        
    