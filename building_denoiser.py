# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
from skimage.io import imread
#from tqdm import tqdm

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

for candidate in range(1, numBuildings - 1):
#candidate = 8
#    print(candidate)
    props = regionprops(labelled_img)
    prop = props[candidate]
    #c = prop.centroid
    #major = prop.major_axis_length
    #minor = prop.minor_axis_length
    #print(f'major: {major} \t minor: {minor}')
    #th = (180/np.pi) * prop.orientation
    region = np.where(labelled_img == candidate)
    blank[region] = 1
    #plt.imshow(blank)
    #print(f'orientation: {th}')
    
    
    cnts = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    
    
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(output, [box], 0, 1, 2)
    blank = np.zeros_like(padded)
    

rgb_norm = rgb / 255
alpha = 0.9
cleaned_overlay = rgb_norm[:,:,2] + output * alpha

dl_overlay = rgb_norm[:,:,2] + (img / 255) * alpha

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
#ax1.imshow(labelled_img)
#ax2.imshow(output)
#ax3.imshow(overlay)
#ax4.imshow(output)


fig, ax = plt.subplots(2,2, figsize=(15,15), sharex=True, sharey=True)
#ax1.imshow(img_crop)
ax[0][0].imshow(rgb)
ax[0][0].set_title('Original Image')
ax[0][1].imshow(img)
ax[0][1].set_title('DL Mask')
ax[1][0].imshow(dl_overlay)
ax[1][0].set_title('DL Overlay')
ax[1][1].imshow(cleaned_overlay)
ax[1][1].set_title('Polygonized')

ax[1, 0].set_title('Segmentation')
for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()




#props = regionprops(segs_sample_labelled)
#blank = np.zeros_like(img_sample)
#for seg in tqdm(range(len(np.unique(segs_sample_labelled)))):
#    region = np.where(segs_sample_labelled == seg)
#    if np.isin(img_sample[region], 1).any():
##        blank[region] = 1
#        overlap = (img_sample[region] == 1).sum()
#        mismatch = (img_sample[region] == 0).sum()
##        if mismatch < 500:
#        if 5 * overlap > mismatch and mismatch < 2000:
#            blank[region] = 1


