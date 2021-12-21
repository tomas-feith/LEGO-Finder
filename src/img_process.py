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
import time
import shutil

def grayscale_filter(img):
    """
    This function turns an image into its grayscale version.

    Parameters
    ----------
    img : np.array
        Original image to convert.

    Returns
    -------
    img_trans : np.array
        Image converted to grayscale.
    """
    img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_trans

def gauss_blur_filter(img, sigma):
    """
    This function applies a gaussian blur to an image.

    Parameters
    ----------
    img : np.array
        Original image to convert.

    Returns
    -------
    img_trans : np.array
        Image after gaussian blur
    sigma : float
        Parameter to specify the std (and thus the range) of the gaussian blur
        applied.
    """
    img_trans = ndi.gaussian_filter(img, sigma)
    return img_trans

def gauss_noise_filter(img, sigma):
    """
    This function applies gaussian noise to an image.

    Parameters
    ----------
    img : np.array
        Original image to convert.
    sigma : float
        Parameter to specify the std (and thus the range) of the gaussian
        noise applied

    Returns
    -------
    img_trans : np.array
        Image with noise.
    """
    row,col,ch= img.shape
    gauss = np.random.normal(0,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    img_trans = img + gauss
    return img_trans

def color_jitter_filter(img, power):
    """
    This function performs a small color distortion to an image.

    Parameters
    ----------
    img : np.array
        Original image to convert.
    power : np.int
        Distortive power of the function (i.e. maximum shift of pixel value).
        Should be a positive number.

    Returns
    -------
    img_trans : np.array
        Image after distortion.
    """
    h,w,c = img.shape

    noise = np.random.randint(0,power+1,(h, w)) # design jitter/noise here
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise

    img_trans = cv2.add(img, zitter)
    return img_trans

def sobel_filter(img):
    """
    This function applies a sobel filter to an image.

    Parameters
    ----------
    img : np.array
        Original image to convert.

    Returns
    -------
    img_trans : np.array
        Image resulting from sobel filter.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_trans = ndi.sobel(img_gray)
    return img_trans

def laplace_filter(img):
    """
    This function applies a laplacian filter to an image.

    Parameters
    ----------
    img : np.array
        Original image to convert.

    Returns
    -------
    img_trans : np.array
        Image with noise.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_trans = ndi.laplace(img_gray)
    return img_trans

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

    # we don't want to make cutout on regions where boxes overlap,
    # so first we will define the forbidden regions
    forbidden = np.zeros((img.shape[0], img.shape[1]))

    # iterate through all the boxes
    for box in boxes.reset_index(drop=True).index:
        # start by finding the position and size of the box
        x = int(boxes.iloc[box]['x'].item())
        y = int(boxes.iloc[box]['y'].item())
        w = int(boxes.iloc[box]['width'].item())
        h = int(boxes.iloc[box]['height'].item())
        # and create a mask, 1 where the box is, 0 everywhere else
        this_box = np.zeros((img.shape[0], img.shape[1]))
        this_box[y:y+h, x:x+w] = 1
        # now we update the forbidden regions based on this
        # if forbidden is already 1, that means some box already was found
        # there, so we update it to 2 (overlapping region)
        forbidden = np.where(forbidden == 1, this_box + 1, forbidden)
        # if it is still 0 then it's a new area and we update it one
        forbidden = np.where(forbidden == 0, this_box, forbidden)

    # after this, the regions where forbidden is 2 are overlap of boxes,
    # where it's 1 or 0 it's fair game
    forbidden = np.where(forbidden == 1, 0, forbidden)
    forbidden = np.where(forbidden == 2, 1, forbidden)
    # final output is a mask which is 1 on the forbidden areas and
    # 0 everywhere else

    # now that we already know the forbidden areas we can get started
    for box in boxes.reset_index(drop=True).index:
        # once again, get the dimensions of each box
        box_x = int(boxes.iloc[box]['x'].item())
        box_y = int(boxes.iloc[box]['y'].item())
        box_w = int(boxes.iloc[box]['width'].item())
        box_h = int(boxes.iloc[box]['height'].item())

        #generate a random rectange inside the box
        x1 = np.random.randint(box_x, box_x + box_w + 1)
        x2 = np.random.randint(box_x, box_x + box_w + 1)
        x1,x2 = min(x1,x2), max(x1,x2)

        y1 = np.random.randint(box_y, box_y + box_h + 1)
        y2 = np.random.randint(box_y, box_y + box_h + 1)
        y1,y2 = min(y1,y2), max(y1,y2)

        # cutout = 0 on the rectangle, 1 everywhere else
        cutout = np.ones((img.shape[0], img.shape[1]))
        cutout[y1:y2,x1:x2] = 0

        # if there is any pixel in the rectangle contained in the forbidden
        # areas, we change it to 1 so it remains unchanged
        cutout = np.where(np.logical_and(cutout == 0, forbidden == 1), 1, cutout)

        # create the 4 channels for the cutout and asign a random value to
        # the image on these locations
        cutout = np.dstack((cutout, cutout, cutout))
        img_trans = img_trans*cutout+np.random.randint(0,256)*(1-cutout)

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

    if not isinstance(params, dict):
        print('The parameters need to be provided in a dictionary! Correct the input and try again.')
        return

    # if the user forgot the dot we add it
    # example 'jpg' -> '.jpg'
    if file_type[0] != '.':
        file_type = '.' + file_type

    # save the original image
    if original:
        cv2.imwrite(base_name + file_type, img)

    # save the grayscaled image
    if grayscale:
        trans_img = grayscale_filter(img)
        cv2.imwrite(base_name + '_grayscale' + file_type, trans_img)

    # save the gaussian blur'd image
    if gauss_blur:
        # if the user provided a sigma we use it
        if 'sigma_blur' in params:
            trans_img = gauss_blur_filter(img, params['sigma_blur'])
        # otherwise the default is 1
        else:
            trans_img = gauss_blur_filter(img, 1)
        cv2.imwrite(base_name + '_gauss_blur' + file_type, trans_img)

    # save the image with gaussian noise
    if gauss_noise:
        # if the user provided a sigma we use it
        if 'sigma_noise' in params:
            trans_img = gauss_noise_filter(img, params['sigma_noise'])
        # otherwise the default is 1
        else:
            trans_img = gauss_noise_filter(img, 1)
        cv2.imwrite(base_name + '_gauss_noise' + file_type, trans_img)

    # save the image with color distortion
    if color_jitter:
        if 'jitter_range' in params:
            trans_img = color_jitter_filter(img, params['jitter_range'])
        else:
            trans_img = color_jitter_filter(img, 30)
        cv2.imwrite(base_name + '_color_jitter' + file_type, trans_img)

    # save the image with sobel filter
    if sobel:
        trans_img = sobel_filter(img)
        cv2.imwrite(base_name + '_sobel_filter' + file_type, trans_img)

    # save the image with laplacian filter
    if laplace:
        trans_img = laplace_filter(img)
        cv2.imwrite(base_name + '_laplace' + file_type, trans_img)

    # if the user wants cutouts, then we also generate those
    if cutout > 0:
        for _ in range(cutout):
            trans_img = cutout_filter(img, box_info)
            cv2.imwrite(base_name + f'_cutout{_}' + file_type, trans_img)

if __name__ == '__main__':

    # we start with the real images

    # define the relevant parameters for the transformations
    # params = {'sigma_blur' : 2, 'sigma_noise' : 10, 'jitter_range' : 50}
    # n_files = len(glob.glob('../data/Real_Images/*'))
    # i = 0
    # for file in glob.glob('../data/Real_Images/*'):
    #     output = '../data/Training_Images/Real/' + file.split('/')[-1][:-4]
    #     img = cv2.imread(file)
    #     transform_img(img = img,
    #                   base_name = output,
    #                   file_type = '.png',
    #                   sobel=False,
    #                   laplace=False,
    #                   params = params,
    #                   cutout = 0,
    #                   box_info = None,
    #                   )
    #     i += 1
    #     print(f'REAL: {i / n_files * 100} % done!')

    # suffixes = ['color_jitter', 'gauss_blur', 'gauss_noise', 'grayscale']
    # files = glob.glob('../data/Bounding_Boxes/Real/*.txt')
    # for file in files:
    #     if all(suffix not in file for suffix in suffixes):
    #         with open(file, 'r') as f:
    #             boxes = f.read()
    #         for suffix in suffixes:
    #             with open(file[:-4] + f'_{suffix}.txt', 'w') as f:
    #                 f.write(boxes)

    # print('Real Images completed!')


    # and now on to the synthetic images

    # params = {'sigma_blur' : 2, 'sigma_noise' : 20, 'jitter_range' : 40}
    # # get all the information from the bounding boxes
    # prefix = '../data/Bounding_Boxes/Original'
    # boxes_synth = glob.glob(prefix + '/Floor/*.json') + \
    #               glob.glob(prefix + '/Random/*.json')

    # boxes = box_process(boxes_synth, bulk = True)
    # # define helper variables to keep track of progress
    # files = glob.glob('../data/Model_Images/Random/*')
    # n_files = len(files)
    # i = 0
    # # and now we apply the transformations for each model image
    # for file in files:
    #     if 'Floor' in file:
    #         name = file[-13:-4]
    #         output = '../data/Training_Images/Floor/' + name
    #     if 'Random' in file:
    #         name = file[-14:-4]
    #         output = '../data/Training_Images/Random/' + name
    #     img = cv2.imread(file)
    #     # apply all the transformations at once to that image
    #     transform_img(img = img,
    #                   base_name = output,
    #                   file_type = '.png',
    #                   sobel=False,
    #                   params = params,
    #                   cutout = 3,
    #                   box_info = boxes[boxes['img'] == name + '.png'],
    #                   )
    #     i += 1
    #     print(f'SYNTHETIC: {i / n_files * 100} % done!')
    # # and generate the txts with the bounding boxes for the new images
    txt_generator(glob.glob('../data/Bounding_Boxes/Original/*/*.json'), bulk = True)
    print('Synthetic Images complete!')

    # # copy the negative samples into the train data and generate the empty txt
    # i = 0
    # n_files = len(glob.glob('../data/Negative_Samples/*'))
    # for index, file in enumerate(glob.glob('../data/Negative_Samples/*')):
    #     shutil.copy(file, f'../data/Training_Images/Negative/NegSample_{index:03}.jpg')
    #     with open(f'../data/Bounding_Boxes/Negative/NegSample_{index:03}.txt', 'w') as f:
    #         # don't write anything to the file, there are no bounding boxes
    #         f.write('')
    #     i += 1
    #     print(f'NEGATIVE: {i/n_files * 100} % done!')
    # print('Negative Samples complete!')
