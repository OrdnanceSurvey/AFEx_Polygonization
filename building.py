# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:58:41 2019

@author: Jrainbow


NB: this version of skimage uses "as_grey" argument on imread.  Later versions use "as_gray"!

"""
from skimage.io import imread
import os
from skimage.measure import label
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks)
from matplotlib import pyplot as plt
from matplotlib import cm

#segment = '1_6'
#
#masks_dir = r"O:\tiled\GT1033_HACK_RGB800\buildings_masks_epoch50_09"
#imagery_dir = r"O:\tiled\GT1033_HACK_RGB800"

# Helper Functions
def pad(img, amount):
    if len(img.shape) == 2:
        # GrayScale
        h, w = img.shape
        new_img = np.zeros((h + 2 * amount, w + 2 * amount))
        new_img[amount: -amount, amount: -amount] = img
    elif len(img.shape) == 3:
        # ThreeBand Imagery
        h, w, c = img.shape
        new_img = np.zeros((h + 2 * amount, w + 2 * amount, c))
        new_img[amount: -amount, amount: -amount, :] = img 
    else:
        raise Exception('Imagery neither grayscale nor threeband')
    return new_img

def find_crop_region(region):
    x_coords, y_coords = region[0], region[1]
    
    
    
class BuildingTile:
    
    ROOT_DIR = r"B:\EOResearch\Guyana\HackData\BuildingData"
    TILE = 'GT1033'
    APPENDAGE = 'HACK_RGB800'
    EXTRAS = '_epoch50_09'
    
    def __init__(self, segment):
        self.segment = segment
        self.__tile = BuildingTile.TILE
        self.__appendage = BuildingTile.APPENDAGE
        self.__extras = BuildingTile.EXTRAS
        
    def _get_mask_folder(self):
        mask_folder = os.path.join(BuildingTile.ROOT_DIR, 
                                   rf'{self.__tile}_{self.__appendage}', 
                                   rf'buildings_masks{self.__extras}')
        return mask_folder
    
    def _get_img_path(self):
        img_path = os.path.join(BuildingTile.ROOT_DIR, 
                                   rf'{self.__tile}_{self.__appendage}',
                                   rf'{self.segment}.tif') 
        return img_path                                  
    
    def _get_mask_path(self):
        mask_folder = self._get_mask_folder()
        mask_path = os.path.join(mask_folder, rf'buildings_{self.segment}.tif')
        return mask_path  
    
    def get_mask(self):
        mask_path = self._get_mask_path()
        mask = imread(mask_path, as_grey=True)
        return mask
    
    def get_img(self, as_gray=False):
        img_path = self._get_img_path()
        img = imread(img_path, as_grey=as_gray)
        return img

    def count_building(self):
        from skimage.measure import label
        import numpy as np
        
        mask = self.get_mask()
        labelled = label(mask)
        count = len(np.unique(labelled))
        return count
    
    def select_building(self, building_id):
        numBuildings = self.count_building()
        assert building_id < numBuildings, 'Building does not exist!'
        
        
        
        mask = self.get_mask()
        padded_mask = pad(mask, 20)
        labelled = label(padded_mask)
        singleBuilding = np.zeros_like(labelled)
        region = np.where(labelled == int(building_id))
        singleBuilding[region] = 1
        
        return singleBuilding
    
    def get_one_building(self,building_id):
        buffer = 10
        numBuildings = self.count_building()
        assert building_id < numBuildings, "Building does not exist!"
        
        mask = self.get_mask()
        labelled = label(mask)
        region = np.where(labelled == int(building_id))
        npregion = np.array(region)
        y0 = npregion[1].min() - buffer
        y1 = npregion[1].max() + buffer
        x0 = npregion[0].min() - buffer
        x1 = npregion[0].max() + buffer
        bldg = self.get_img()[x0:x1,y0:y1]
        msk = mask[x0:x1,y0:y1]
        return bldg,msk
        
def houghLines(greyImage):
    
    img = feature.canny(greyImage,sigma=2)
    h,theta,d = hough_line(img)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                         subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    
    ax[2].imshow(greyImage, cmap=cm.gray)
    
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, img.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')
    plt.show()
    
def main():
    import matplotlib.pyplot as plt
    
    segment = '1_7'
    
    myBuilding = BuildingTile(segment)
    mask = myBuilding.get_mask()
    img = myBuilding.get_img()
    count = myBuilding.count_building()
    print("{} buildings detected".format(count))
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(10,10))
    ax1.imshow(mask)
    ax2.imshow(img)
    
main()
    
        
        
        
        
        
        