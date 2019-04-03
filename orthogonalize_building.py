# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:23:25 2019

@author: Jrainbow
"""

import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import polygon_perimeter, line, polygon
from orthogonal_helpers import find_terminal_vertices, calc_lengths, calc_grads, calc_angles, optimize_angles
from skimage.morphology import binary_dilation, dilation, closing
from skimage.measure import label, regionprops

corners = load(r'images/approx_building.npy')

def filt_lengths(df, threshold=30):
    """
    Filters lines to find rough parallels (within a certain threshold) to the longest line.
    
    Args:
        df: Panda DataFrame with lines and angles.
        threshold: Threshold of angle variance to longest line. Default is 30 degrees.
    
    Returns:
        DataFrame of only rows with lines parallel (withinh certain threshold) of the longest line.
    """
    longest_line = df['Length'].idxmax()
    single_angle = []
    #1. Find angle corresponding to longest line
    crit_row = df.iloc[longest_line]
    crit_angles = np.array(crit_row['Optimized Angles'])
    crit_angle = crit_angles[crit_angles < 180]
    crit_angle = crit_angle[0]
    #2. Filter all lines that have one angle within a threshold of the angle in 1.
    parallels = []
    idxs = [i for i in range(len(df))]
    for index, row in df.iterrows():
        angles = np.array(row['Optimized Angles'])
        for angle in angles:
            if angle > (crit_angle - threshold) and angle < (crit_angle + threshold):
                parallels.append(index)
                single_angle.append(angle)
    perpendiculars = [x for x in idxs if x not in parallels]
    print(perpendiculars)
    filt_para = df.loc[parallels]
    filt_perp = df.loc[perpendiculars]
    filt_para['Single Angle'] = single_angle    
    return filt_para, filt_perp


def weight_avs(df):
    lengths = 0
    compounds = 0
    for index, row in df.iterrows():
        lengths += row['Length']
        compounds += row['Length'] * row['Single Angle']
    theta_av = compounds / lengths
    return theta_av


def av_gradient(angle):
    m = - np.tan(angle * (np.pi / 180))
    return m

def find_orientation(arr):
    intcorners = arr.astype(int)
    start, end = find_terminal_vertices(arr)
    lengths = calc_lengths(intcorners)
    angles = calc_angles(intcorners)
    gradients = calc_grads(intcorners)
    opt_angles = optimize_angles(intcorners)
    df = pd.DataFrame({"Length": lengths,
                       "Angle": angles,
                       "Gradient": gradients,
                       "Optimized Angles": opt_angles,
                       "Start Vertex": start,
                       "End Vertex": end})
    filt_para, filt_perp = filt_lengths(df)
    orientation = weight_avs(filt_para)
    grad = av_gradient(orientation)
    print(f"Orientation is: \t {orientation}" )
    print(f"Gradient is: \t \t {grad} \n")
    print(filt_para)
    return filt_para, filt_perp, grad

def draw_polygon(arr, crop=True):
    buffer = 25 
    intcorners = arr.astype(int)
    img = np.zeros((800,800))
    r = intcorners[:, 0]
    c = intcorners[:, 1]
    rr, cc = polygon_perimeter(r, c)
    img[rr, cc] = 127
    if crop == True:
        minx, miny = intcorners[:, 1].min(), intcorners[:, 0].min()
        maxx, maxy = intcorners[:, 1].max(), intcorners[:, 0].max()
        img = img[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(img)
    return img

def fill_polygon(arr, crop=True):
    buffer = 25 
    intcorners = arr.astype(int)
    img = np.zeros((800,800))
    r = intcorners[:, 0]
    c = intcorners[:, 1]
    rr, cc = polygon(r, c)
    img[rr, cc] = 127
    if crop == True:
        minx, miny = intcorners[:, 1].min(), intcorners[:, 0].min()
        maxx, maxy = intcorners[:, 1].max(), intcorners[:, 0].max()
        img = img[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(img)
    return img   
    
def ortho_line(filt_para, m):
    """
    Input is a FILTERED dataframe.
    
    Args:
        filt: A filtered dataFrame.
        m: Gradient......
    
    Returns:
        A filtered dataFrame # TODO - of what???
    # TODO - Make lines appropriate length for building.
    # TODO - Make lines high-resolution (i.e. no int coercion)
    """
    starts = []
    ends = []
    intercepts = []
    for index, row in filt_para.iterrows():
        orig_start, orig_end = row['Start Vertex'], row['End Vertex']
        intercept =  (((- 1 / 2) * m) * (orig_start[0] + orig_end[0])) + ((1/2) * (orig_start[1] + (orig_end[1])))
        
        new_start_x =  0
        new_start_y = (m * new_start_x) - intercept
        new_end_x = 799
        new_end_y = (m * new_end_x) - intercept
        
        new_start_x, new_start_y, new_end_x, new_end_y = int(new_start_x), int(new_start_y), int(new_end_x), int(new_end_y)
        
        start = (new_start_x, - new_start_y)
        end = (new_end_x, - new_end_y)
        starts.append(start)
        ends.append(end)
        intercepts.append(intercept)
        
    filt_para['Intercept'] = intercepts
    filt_para['m'] = [m] * len(intercepts)
    filt_para['New Start Vertex'] = starts
    filt_para['New End Vertex'] = ends
    return filt_para


def perp_line(filt_perp, m):
    """
    Input is a FILTERED dataframe.
    
    Args:
        filt: A filtered dataFrame.
        m: Gradient......
    
    Returns:
        A filtered dataFrame # TODO - of what???
    # TODO - Make lines appropriate length for building.
    # TODO - Make lines high-resolution (i.e. no int coercion)
    """
    m = (-1) / m
    starts = []
    ends = []
    intercepts = []
    for index, row in filt_perp.iterrows():
        orig_start, orig_end = row['Start Vertex'], row['End Vertex']
        intercept =  (((- 1 / 2) * m) * (orig_start[0] + orig_end[0])) + ((1/2) * (orig_start[1] + (orig_end[1])))
        
        new_start_y = 0
        new_start_x =  (new_start_y + intercept) / m
        new_end_y = 799
        new_end_x = (new_end_y + intercept) / m
        
        new_start_x, new_start_y, new_end_x, new_end_y = int(new_start_x), int(new_start_y), int(new_end_x), int(new_end_y)
        
        start = (- new_start_x, new_start_y)
        end = (- new_end_x, new_end_y)
        starts.append(start)
        ends.append(end)
        intercepts.append(intercept)
      
    filt_perp['Intercept'] = intercepts
    print(len(intercepts))
    filt_perp['m'] = [m] * len(intercepts)
    filt_perp['New Start Vertex'] = starts
    filt_perp['New End Vertex'] = ends
    return filt_perp


def plot_parallels(ortho, dim, crop=True):
    background = np.zeros((800,800))
    for index, row in ortho.iterrows():
        start_tuple, end_tuple = ortho['New Start Vertex'][index], ortho['New End Vertex'][index]
        rr, cc = line(start_tuple[1], start_tuple[0], end_tuple[1], end_tuple[0])
        background[rr, cc] = 127
    if crop == True:
        buffer = 25
        minx, miny, maxx, maxy = dim
        background = background[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]       
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(background)
    return background    


def plot_perpendiculars(perp, dim, crop=True):
    background = np.zeros((800,800))
    for index, row in perp.iterrows():
        start_tuple, end_tuple = perp['New Start Vertex'][index], perp['New End Vertex'][index]
        rr, cc = line(start_tuple[1], start_tuple[0], end_tuple[1], end_tuple[0])
        background[rr, cc] = 127
    if crop == True:
        buffer = 25
        minx, miny, maxx, maxy = dim
        background = background[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]       
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(background)
    return background  

def concat_df(ortho, perp):
    result = pd.concat([ortho, perp], sort=True)
    return result

def buffer_filled_poly(img, n_iter):
    if n_iter == 0:
        return img
    else:
        buffered = dilation(img)
        return buffer_filled_poly(buffered, n_iter - 1)

def plot_grid(ortho, perp, dim, crop=True):
    background = np.zeros((800,800))
    lattice = pd.concat([ortho, perp])
    for index, row in lattice.iterrows():
        start_tuple, end_tuple = lattice['New Start Vertex'][index], lattice['New End Vertex'][index]
        rr, cc = line(start_tuple[1], start_tuple[0], end_tuple[1], end_tuple[0])
        background[rr, cc] = 127
    if crop == True:
        buffer = 25
        minx, miny, maxx, maxy = dim
        background = background[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]       
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(background)
    return background

def plot_original_corners(arr, crop=True):
    buffer = 25
    intcorners = arr.astype(int)
    blank = np.zeros((800,800))
    minx, miny = intcorners[:, 1].min(), intcorners[:, 0].min()
    maxx, maxy = intcorners[:, 1].max(), intcorners[:, 0].max()
    for i in range(len(arr)):
        blank[intcorners[i, 0]][intcorners[i, 1]] = 255
    if crop == True:
        blank = blank[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(blank)
    dim = [minx, miny, maxx, maxy]
    return blank, dim

def plot_new_corners(df, dim, crop=True):
    blank = np.zeros((800,800))
    xs = np.array(df['x_coord']).astype(int)
    ys = np.array(df['y_coord']).astype(int)
    for i in range(len(xs)):
        blank[xs[i]][ys[i]] = 255
    if crop == True:
        buffer = 25
        minx, miny, maxx, maxy = dim
        blank = blank[miny - buffer: maxy + buffer, minx - buffer: maxx + buffer]       
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.imshow(blank)
    return blank

def label_regions(lattice):
    assert len(np.unique(lattice)) == 2, \
        "Lattice has more than two values"
    binary = lattice / lattice.max()
    # Dilate
    dilated = binary_dilation(binary)
    # Invert
    inv = 1 - dilated
    # Label
    labelled = label(inv)
    # Dilate twice
    d1 = dilation(labelled)
    d2 = dilation(d1)
    
    return d2          

def find_lattice_vertices(ortho, perp):
    xcoords = []
    ycoords = []
    m_o = np.array(ortho['m'])[0]
    m_p = np.array(perp['m'])[0]
    # Calculate intersection points from manually inverting matrix 
    for i, r_o in ortho.iterrows():
        c_o = r_o['Intercept']
        for j, r_p in perp.iterrows():
            c_p = r_p['Intercept']
            x = (1 / (m_o - m_p)) * (m_o * c_p - m_p * c_o)
            y = (1 / (m_o - m_p)) * (c_p - c_o)
            xcoords.append(x)
            ycoords.append(y)
    vertices = pd.DataFrame({"x_coord": xcoords,
                             "y_coord": ycoords})    
    
    return vertices

def filter_corners(filled_poly, vertices, buffer_dist=2):
    """
    TODO: make sure filled_poly is NOT cropped at this stage
    """
    
    # buffer filled_poly
    buffered = buffer_filled_poly(filled_poly, buffer_dist)
    for i, row in vertices.iterrows():
        x = row['x_coord'].astype(int)
        y = row['y_coord'].astype(int)
        if buffered[x][y] == 0:
            vertices = vertices.drop(i, axis=0)
    
    return vertices
            

if __name__ == "__main__":
    filt_para, filt_perp, m = find_orientation(corners)
    ortho = ortho_line(filt_para, m)
    perp = perp_line(filt_perp, m)
    corner_pts, dim = plot_original_corners(corners, crop=True)
    poly = draw_polygon(corners, crop=True)
    fill = fill_polygon(corners, crop=False)
    para = plot_parallels(ortho, dim, crop=True)
    perp_grid = plot_perpendiculars(perp, dim, crop=True)
    lattice = plot_grid(ortho, perp, dim, crop=True)
    regions = label_regions(lattice)
    vertices = find_lattice_vertices(ortho, perp)
    new_corners = plot_new_corners(vertices, dim, crop=True)
    filt_vertices = filter_corners(fill, vertices, buffer_dist=5)
    filt_corners = plot_new_corners(filt_vertices, dim, crop=True)

    fig, ax = plt.subplots(2,2,figsize=(20,20))
    ax[0][0].imshow(poly)
    ax[0][1].imshow(fill)
    ax[1][0].imshow(new_corners)
    ax[1][1].imshow(filt_corners)
    
    concat = concat_df(ortho, perp)


buffer = 25
props = regionprops(regions)
output = np.zeros((dim[3] - dim[1] + 2 * buffer, dim[2] - dim[0] + 2 * buffer))
for seg in range(len(np.unique(regions))):
    region = np.where(regions == seg)
    if np.isin(fill[region], 127).any():
        overlap = (fill[region] == 127).sum()
        mismatch = (fill[region] == 0).sum()
        if overlap > 2 * mismatch:
            output[region] = 127
            closed = closing(output)
            plt.imshow(closed)
            
buffered = buffer_filled_poly(fill, 5)

#for seg in tqdm(range(len(np.unique(segs_sample_labelled)))):
#    region = np.where(segs_sample_labelled == seg)
#    if np.isin(img_sample[region], 1).any():
##        blank[region] = 1
#        overlap = (img_sample[region] == 1).sum()
#        mismatch = (img_sample[region] == 0).sum()
##        if mismatch < 500:
#        if 5 * overlap > mismatch and mismatch < 2000:
#            blank[region] = 1
    
