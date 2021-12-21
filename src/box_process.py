
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:46:40 2021

@author: tsfeith
"""
import glob
import pandas as pd
import cv2

def txt_generator(file_name, bulk = False):
    """
    This function will generate txt of the original json files with the info
    about the bounding boxes for all the new images generated.

    Parameters
    ----------
    file_name : str OR iterable
        File(s) to clone.
    bulk : bool, optional
        If True, file_name should be an iterable of str, containing all the
        files to use. If False, then file_name should be one single file.
        The default is False.

    Returns
    -------
    None.

    """
    if bulk:
        temp_dfs = []
        # for each json file, save it to a DataFrame
        for file in file_name:
            temp_dfs.append(pd.read_json(file))
        # and then join all the DataFrames
        df = pd.concat(temp_dfs)
        df.reset_index(inplace = True, drop = True)
    else:
        # if it's a single file just read it
        df = pd.read_json(file_name)

    # now we save the info of the bounding boxes of all images
    # each image gets a txt file
    # first we get the names of the new files
    new_files = []
    prefix = '../data/Training_Images'
    files =  glob.glob(prefix + '/Floor/*') + \
                  glob.glob(prefix + '/Random/*')

    for file in files:
        new_files.append(file)

    img_x, img_y, _ = cv2.imread(new_files[0]).shape

    # for all files, we get their info from the DataFrame and save it to a txt
    for file in new_files:
        # prepare to stor all the information
        if 'random' in file:
            add = 1000
        if 'floor' in file:
            add = 0

        img = df.iloc[int(file.split('_')[2].split('.')[0]) + add].copy().rename(None)
        ids = []
        xs = []
        ys = []
        widths = []
        heights = []
        # iterate through each box and get the needed info
        for box in img['captures']['annotations'][0]['values']:
            id_ = box['label_id']

            # we need to be careful because the extracted positions are for
            # the upper left corner, and we want the center position for YOLO
            x_center = box['x'] + box['width'] / 2
            y_center = box['y'] + box['height'] / 2

            # also, these positions need to be in units relative to the
            # image size
            x_center /= img_x
            y_center /= img_y

            # the width and height should also be in relative units
            width = box['width'] / img_x
            height = box['height'] / img_y

            # convert id from 1-index to 0-index
            ids.append(id_ - 1)
            xs.append(x_center)
            ys.append(y_center)
            widths.append(width)
            heights.append(height)

        # and write everything to the txt file
        if 'random' in file:
            filename = file[31:-4]
            output = f'../data/Bounding_Boxes/Random/{filename}.txt'
        if 'floor' in file:
            filename = file[30:-4]
            output = f'../data/Bounding_Boxes/Floor/{filename}.txt'
        with open(output, 'w') as f:
            for i in range(len(ids)):
                f.write(f'{ids[i]} {xs[i]} {ys[i]} {widths[i]} {heights[i]}\n')

def box_process(file_name, bulk = False):
    """
    Function to save all the relevant information from the json file(s) into a
    pandas DataFrame, to make it easier to read.

    Parameters
    ----------
    file_name : str OR iterable
        File(s) from which to extract the info about the boxes
    bulk : bool, optional
        If True, file_name should be an iterable of str, containing all the
        files to read. If False, then file_name should be one single file. The
        default is False.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame containing all the relevant info for the boxes.

    """

    if bulk:
        dfs = []
        # read all the files
        for file in file_name:
            # convert each file to a DataFrame
            df = pd.read_json(file)

            # create a dict that will contain only the relevant info
            data = {'img':[],
                    'x':[],
                    'y':[],
                    'width':[],
                    'height':[]
                    }

            # and now extract all the relevant information from the DataFrame
            for row in df.index:
                vals = df.iloc[row]['captures']['annotations'][0]['values']
                if 'random' in file:
                    pre_img = 'random_'
                if 'floor' in file:
                    pre_img = 'floor_'
                post_img = df.iloc[row]['captures']['filename'].split('_')[-1]

                img = pre_img + f'{int(post_img[:-4]) - 2:03}' + post_img[-4:]
                for box in vals:
                    data['img'].append(img)
                    data['x'].append(box['x'])
                    data['y'].append(box['y'])
                    data['width'].append(box['width'])
                    data['height'].append(box['height'])

            # save a DataFrame created from the dict
            dfs.append(pd.DataFrame.from_dict(data))

        # and join all the dataframes
        df_clean = pd.concat(dfs)
        df_clean.reset_index(inplace = True, drop = True)

    # if bulk == False, the only difference is that there is only one file,
    # so we don't join dataframes at the end. Everything else is the same.
    else:
        df = pd.read_json(file_name)

        data = {'img':[],
                'x':[],
                'y':[],
                'width':[],
                'height':[]
                }

        for row in df.index:
            vals = df.iloc[row]['captures']['annotations'][0]['values']
            img = df.iloc[row]['captures']['filename'].split('/')[-1]
            for box in vals:
                data['img'].append(img)
                data['x'].append(box['x'])
                data['y'].append(box['y'])
                data['width'].append(box['width'])
                data['height'].append(box['height'])

        df_clean = pd.DataFrame.from_dict(data)

    return df_clean

## MAIN ##
if __name__=='__main__':
    ## Script to convert txt from 23 to 10 pieces
    ## We take the already made txt's and delete pieces we don't want first (by their label_id),
    ## and then rename their label_id to a new one, basically resetting the index

    # Pieces we'll use label_name: label_id
    pieces_dict = {'2540': 1, # Use this for the synthetic data
    '3001': 2,
    '3003': 7,
    '3004': 14,
    '3020': 12,
    '3021': 5,
    '3022': 8,
    '3023': 6,
    '3039': 11,
    '3660': 13}

    pieces_dict_real = {'2540': 11, # Use this for the real data
    '3001': 1,
    '3003': 0,
    '3004': 18,
    '3020': 21,
    '3021': 20,
    '3022': 13,
    '3023': 17,
    '3039': 22,
    '3660': 8}

    new_index = {} # New label_id for pieces
    for i in range(10):
        # Uncomment following line for synthetic data
        new_index[list(pieces_dict.values())[i]] = i

        # Uncomment following line for real data
        # new_index[list(pieces_dict_real.values())[i]] = i


    # Select all files in folder
    for file in glob.glob('../data/Bounding_Boxes/Random/*.txt'):

        # Read txt lines
        with open(file) as f:
            lines = f.readlines()

        # Filter only boxes we want
        bboxes = []
        for line in lines:
            bbox = []
            line_split = line.split()

            # Uncomment following line for synthetic data
            if int(line_split[0]) in pieces_dict.values():
            # if int(line_split[0]) in pieces_dict_real.values():
                old_id = int(line_split[0])
                new_id = new_index[old_id]

                # Create correct bound box
                new_line_split=[new_id] + line_split[1:]
                bboxes.append(new_line_split)

        # Write to file
        with open(file, 'w') as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
    print('FINISHED')
