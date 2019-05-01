# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:58:41 2019

@author: Jrainbow
"""
from skimage.io import imread
import os

segment = '1_6'

masks_dir = r"O:\tiled\GT1033_HACK_RGB800\buildings_masks_epoch50_09"
imagery_dir = r"O:\tiled\GT1033_HACK_RGB800"

# Helper Functions
def pad(img, amount):
    import numpy as np
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
    
    

class Building:
    
    ROOT_DIR = r"O:\tiled"
    SEGMENT = '1_6'
    TILE = 'GT1033'
    APPENDAGE = 'HACK_RGB800'
    EXTRAS = '_epoch50_09'
    
    def __init__(self, building_id):
        self.building_id = building_id
        self.__segment = Building.SEGMENT
        self.__tile = Building.TILE
        self.__appendage = Building.APPENDAGE
        self.__extras = Building.EXTRAS
        
    def _get_mask_folder(self):
        mask_folder = os.path.join(Building.ROOT_DIR, 
                                   rf'{self.__tile}_{self.__appendage}', 
                                   rf'buildings_masks{self.__extras}')
        return mask_folder
    
    def _get_img_path(self):
        img_path = os.path.join(Building.ROOT_DIR, 
                                   rf'{self.__tile}_{self.__appendage}',
                                   rf'{self.__segment}.tif') 
        return img_path                                  
    
    def _get_mask_path(self):
        mask_folder = self._get_mask_folder()
        mask_path = os.path.join(mask_folder, rf'buildings_{self.__segment}.tif')
        return mask_path  
    
    def get_mask(self):
        mask_path = self._get_mask_path()
        mask = imread(mask_path, as_gray=True)
        return mask
    
    def get_img(self, as_gray=False):
        img_path = self._get_img_path()
        img = imread(img_path, as_gray=as_gray)
        return img

    def select_building(self):
        from skimage.measure import label
        import numpy as np
        
        mask = self.get_mask()
        padded_mask = pad(mask, 20)
        labelled = label(padded_mask)
        singleBuilding = np.zeros_like(labelled)
        region = np.where(labelled == int(self.building_id))
        singleBuilding[region] = 1
        
        return singleBuilding
        
        
        
        
        
        