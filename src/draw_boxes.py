# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:57:33 2021

@author: tsfeith
"""
import cv2
import numpy as np
import scipy.ndimage as ndi
import glob
from box_process import box_process, txt_generator
import shutil

def cutout_filter(img, boxes):
    """
    This function performs random cutouts on the image, inside the locations
    of the bounding boxes. To avoid removing too much information, it is not
    allowed to cutout in areas where boxes overlap.

    Parameters
    ----------
    img : np.array
        Original image to convert.
    boxes : pd.DataFrame
        DataFrame containing the location and dimensions of the bounding boxes.

    Returns
    -------
    img_trans : np.array
        Image with cutouts.

    """

    img_trans = img


    # iterate through all the boxes
    for box in boxes.reset_index(drop=True).index:
        # start by finding the position and size of the box
        x = int(boxes.iloc[box]['x'].item())
        y = int(boxes.iloc[box]['y'].item())
        w = int(boxes.iloc[box]['width'].item())
        h = int(boxes.iloc[box]['height'].item())
        # and create a mask, 1 where the box is, 0 everywhere else
        this_box = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        this_box[y:y+h, x:x+w, :] = 1
        # now we update the forbidden regions based on this
        # if forbidden is already 1, that means some box already was found
        # there, so we update it to 2 (overlapping region)
        img_trans = np.where(this_box == 1, [0,0,255], img_trans)


    return img_trans

def transform_img(img,
                  base_name = '',
                  file_type = '.png',
                  original = True,
                  grayscale = True,
                  gauss_blur = True,
                  gauss_noise = True,
                  color_jitter = True,
                  sobel = True,
                  laplace = True,
                  cutout = 0,
                  params = {},
                  box_info = None):
    """
    This function generates transformed versions of an image, and saves them
    for later usage.
    The filter applied are
        - Grayscale
        - Gaussian Blur
        - Gaussian Noise
        - Color Distortion (Jitter)
        - Sobel Filter
        - Laplacian Filter
        - Random Cutouts

    Parameters
    ----------
    img : np.array
        Image on which to perform the operations.
    base_name : str
        Prefix to apply to the name of all images. The default is ''.
    file_type : str
        Format to apply to the saved images. The default is '.png'.
    original : bool, optional
        Whether to also save the unaltered image on the final folder.
        The default is True.
    grayscale : bool, optional
        Whether or not to convert to grayscale. The default is True.
    gauss_blur : bool, optional
        Whether or not to apply gaussian blur. The default is True.
    gauss_noise : bool, optional
        Whether or not to apply gaussian noise. The default is True.
    color_jitter : bool, optional
        Whether or not to apply color jitter. The default is True.
    sobel_filter : bool, optional
        Whether or not to apply sobel filter. The default is True.
    laplace : bool, optional
        Whether or not to apply the laplacian filter. The default is True.
    cutouts : int, optional
        How many cutouts variations to perform per image. All values <=0 will
        be treated as 0. For any value >0 the parameter box_info also must be
        provided. The default value is 0.
    params : dict, optional
        Dict to write extra parameters for the transformations if necessary.
        The default is None, and does nothing.
        Possible parameters: 'sigma_blur', 'sigma_noise', 'jitter_range'
    box_info : pd.DataFrame, optional
        Information about the bounding boxes in each image. If cutouts <=0,
        this does nothing. Otherwise it is used to determine where to draw
        the cutouts. DataFrame must contain columns ['x','y','width','height']
        corresponding to the dimensions of the boxes.
    Returns
    -------
    None.

    """

    # if the user wants cutouts, then we also generate those
    if cutout > 0:
        for _ in range(cutout):
            trans_img = cutout_filter(img, box_info)
            cv2.imwrite(base_name + file_type, trans_img)

if __name__ == '__main__':

    # get all the information from the bounding boxes
    prefix = '../data/Bounding_Boxes/Original'
    boxes_synth = glob.glob(prefix + '/Floor/*.json') + \
                  glob.glob(prefix + '/Random/*.json')

    boxes = box_process(boxes_synth, bulk = True)
    # define helper variables to keep track of progress
    files = [glob.glob('../data/Model_Images/Random/*')[0]]
    n_files = len(files)
    i = 0
    # and now we apply the transformations for each model image
    print(boxes)
    for file in files:
        if 'Floor' in file:
            name = file[-13:-4]
        if 'Random' in file:
            name = file[-14:-4]
        output = './test'
        img = cv2.imread(file)
        cv2.imwrite('original.png', img)
        # apply all the transformations at once to that image
        transform_img(img = img,
                      base_name = output,
                      file_type = '.png',
                      sobel=False,
                      cutout = 1,
                      box_info = boxes[boxes['img'] == name + '.png'],
                      )
        i += 1
        print(f'SYNTHETIC: {i / n_files * 100} % done!')
    print('Synthetic Images complete!')
