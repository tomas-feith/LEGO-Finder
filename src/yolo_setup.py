# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:40:21 2021

@author: tsfei

THIS SCRIPT ONLY WORKS IF YOU'RE AT THE SRC OF **OUR** GITHUB REPO
It is meant to be used a script to setup yolo on a local machine

For an explanation as to what we're doing here see
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

GITHUB REPO VERSION USED : 7fe4fb7c457c789fe2c00195b6d8f851083b1514

"""
import glob
import cv2
import os
import numpy as np

if __name__ == '__main__':

    # edit here all the relevant parameters
    class_names = ['3003',
                   '3001',
                   '30136',
                   '99780',
                   '32000',
                   '32028',
                   '30526',
                   '44567',
                   '3660',
                   '3676',
                   '99021',
                   '2540',
                   '57909',
                   '3022',
                   '6249',
                   '57908',
                   '41677',
                   '3023',
                   '3004',
                   '6632',
                   '3021',
                   '3020',
                   '3039'] # we're using the LEGO pieces IDs
    n_classes = len(class_names)

    # define how many images to use
    n_imgs = 50

    # define the various distribution of images

    # dists_full : distribution between the various image categories
    # dists_synth : for the synthetic, distribution between the backgrounds
    # augment_ratio : on how many images to apply augmentations

    # CAREFUL! Applying augmentation leads to a 9-fold increase of the number
    # of images. So if n_imgs = 1000 and aug_ratio = 0.5, then we'll actually
    # use 500*1 + 500*9 = 5000 images for trainig.

    dists_full = {'real': 0.1, 'synth': 0.8, 'neg': 0.1}
    dists_synth = {'blank': 1/3, 'floor': 1/3, 'random': 1/3}
    aug_ratio = 0.5

    # CAVEATS!!!!
        # If there are not enough real images to satisfy what was request,
        # synthetic images will be used in replacement

        # No augmentations will be performed on negative samples

    # if it was requested more real images than possible, we let the user know
    if dists_full['real'] * n_imgs > len(glob.glob('../data/Real_Images/*')):
        requested = dists_full['real'] * n_imgs
        possible = len(glob.glob('../data/Real_Images/*'))
        print(f'WARNING! Not enough real images ({requested} > {possible}). The remaining ones will be replaced by synthetic images.')
        dists_full['real'] = len(glob.glob('../data/Real_Images/*')) / n_imgs
        dists_full['synth'] = 1 - dists_full['real'] - dists_full['neg']

    # if more negative were requested than possible then it's the same thing
    if dists_full['neg'] * n_imgs > len(glob.glob('../data/Negative_Samples/*')):
        requested = dists_full['real'] * n_imgs
        possible = len(glob.glob('../data/Negative_Samples/*'))
        print(f'WARNING! Not enough negative samples ({requested} > {possible}). The remaining ones will be replaced by synthetic images.')
        dists_full['neg'] = len(glob.glob('../data/Negative_Samples/*')) / n_imgs
        dists_full['synth'] = 1 - dists_full['real'] - dists_full['neg']

    # define the suffixes for the image augmentation
    suffixes = ['color_jitter',
                'cutout',
                'gauss_blur',
                'gauss_noise',
                'grayscale',
                'laplace']

    # No more definitions! Now just let the code run by itself :)

    blank_max = len(glob.glob('../data/Model_Images/Blank/*'))
    rand_max  = len(glob.glob('../data/Model_Images/Random/*'))
    floor_max = len(glob.glob('../data/Model_Images/Floor/*'))

    # find out which 'blank', 'random' and 'floor' synthetic images to use
    blanks = np.random.choice(range(0, blank_max),
                              size = round(dists_full['synth'] * dists_synth['blank'] * n_imgs),
                              replace = False)
    blanks_aug = np.random.choice(blanks,
                                  size = round(aug_ratio * len(blanks)),
                                  replace = False)
    randoms = np.random.choice(range(0, rand_max),
                               size = round(dists_full['synth'] * dists_synth['random'] * n_imgs),
                               replace = False)
    randoms_aug = np.random.choice(randoms,
                                   size = round(aug_ratio * len(randoms)),
                                   replace = False)
    floors = np.random.choice(range(0, floor_max),
                              size = round(dists_full['synth'] * dists_synth['floor'] * n_imgs),
                              replace = False)
    floors_aug = np.random.choice(floors,
                                  size = round(aug_ratio * len(floors)),
                                  replace = False)

    # find out which real images to use
    reals_max = len(glob.glob('../data/Real_Images/*'))
    reals = np.random.choice(class_names,
                             size = round(dists_full['real'] * n_imgs),
                             replace = False
                             )
    reals_aug = np.random.choice(reals,
                                 size = round(aug_ratio * len(reals)),
                                 replace = False)

    # find out which negative samples to use
    negs_max = len(glob.glob('../data/Negative_Samples/*'))
    negs = np.random.choice(range(0, negs_max),
                            size = round(dists_full['neg'] * n_imgs),
                            replace = False)

    print('BLANKS', blanks)
    print('RANDOMS', randoms)
    print('FLOORS', floors)
    print('REALS', reals)
    print('NEGATIVES', negs)

    # get all the Training Images
    img_blank = glob.glob('../data/Training_Images/Blank/*')
    img_random = glob.glob('../data/Training_Images/Random/*')
    img_floor = glob.glob('../data/Training_Images/Floor/*')

    img_real = glob.glob('../data/Training_Images/Real/*')

    img_neg = glob.glob('../data/Training_Images/Negative/*')

    img_training = []

    for file in img_blank:
        num = '_'.join(file.split('.'))
        num = [int(val) for val in num.split('_') if val.isdigit()][0]

        if num in blanks:
            if num in blanks_aug:
                img_training.append(file)
            else:
                if all(suffix not in file for suffix in suffixes):
                    img_training.append(file)

    for file in img_random:
        num = '_'.join(file.split('.'))
        num = [int(val) for val in num.split('_') if val.isdigit()][0]

        if num in randoms:
            if num in randoms_aug:
                img_training.append(file)
            else:
                if all(suffix not in file for suffix in suffixes):
                    img_training.append(file)

    for file in img_floor:
        num = '_'.join(file.split('.'))
        num = [int(val) for val in num.split('_') if val.isdigit()][0]

        if num in floors:
            if num in floors_aug:
                img_training.append(file)
            else:
                if all(suffix not in file for suffix in suffixes):
                    img_training.append(file)

    for file in img_real:
        num = '_'.join(file.split('/'))
        num = [str(val) for val in num.split('_') if val.isdigit()][0]

        if num in reals:
            if num in reals_aug:
                img_training.append(file)
            else:
                if all(suffix not in file for suffix in suffixes):
                    img_training.append(file)

    for file in img_neg:
        num = '_'.join(file.split('.'))
        num = [int(val) for val in num.split('_') if val.isdigit()][0]

        if num in negs:
            img_training.append(file)

    # find the number of batches to use
    max_batches = max([len(img_training),
                       6000,
                       n_classes * 2000])

    # create file yolo-obj
    file = '../yolo_v4/cfg/yolov4-custom.cfg'
    # now we create a copy of this file and we go to edit some of its lines
    lines = []
    # extract all the lines
    with open(file, 'r') as f_in:
        for line in f_in:
            lines.append(line)

    with open('../yolo_v4/yolo-obj.cfg', 'w') as f_out:
        for i, line in enumerate(lines):
            # if it's the batch line, we set it to 64
            if line[:5] == 'batch' and 'normalize' not in line:
                f_out.write('batch=64\n')
            # the subdivisions should be 16 (or potentially 32 or 64 if this gives memory error when training)
            elif line[:12] == 'subdivisions':
                f_out.write('subdivisions=16\n')
            # the width and the height need to be a multiple of 32
            elif 'width' in line:
                f_out.write('width=608\n')
            elif 'height' in line:
                f_out.write('height=608\n')
            # max batches should be the parameters calculated previously
            elif 'max_batches' in line:
                f_out.write(f'max_batches={max_batches}\n')
            # from the batches we also calculate the steps
            elif line[:5] == 'steps':
                f_out.write(f'steps={int(max_batches*0.8)},{int(max_batches*0.9)}\n')
            # define our classes too
            elif 'classes' in line:
                f_out.write(f'classes={n_classes}\n')
            # and the filters are defined in function of the classes
            elif 'filters' in line and '[yolo]' in lines[i + 4]:
                f_out.write(f'filters={(n_classes + 5)*3}\n')
            # finally add the 'max' parameter to the last [yolo] layer
            elif '[yolo]' in lines[i - 1] and len(lines) - i < 20:
                f_out.write('max=100\n')
                f_out.write(line)
            # if the line is none of the above, then just write it
            else:
                f_out.write(line)

    # create file obj.names
    # this file should contain the name of all the classes
    with open('../yolo_v4/data/obj.names', 'w') as f_out:
        for name in class_names:
            f_out.write(name + '\n')

    # create file obj.data
    # this file contains some relevant information
    with open('../yolo_v4/data/obj.data', 'w') as f_out:
        f_out.write(f'classes={n_classes}\n')
        f_out.write('train = data/train.txt\n')
        f_out.write('valid = data/train.txt\n')
        f_out.write('names = data/obj.names\n')
        f_out.write('backup = backup/')

    # and now we need to save each of the training images we want to a
    # specific folder for yolo to work
    i = 0
    max_size = len(img_training)
    # if the needed folder doesn't exist yet, we create it
    if not os.path.exists('../yolo_v4/data/obj/'):
        os.makedirs('../yolo_v4/data/obj/')
    # delete all images and txt already in the folder
    old_files = glob.glob('../yolo_v4/data/obj/*')
    for f in old_files:
        os.remove(f)
    # copy all images and txt into yolo_v4/data/obj/
    for img in img_training:
        image = cv2.imread(img)
        # Save .jpg image
        out_name = img.split('/')[-1].split('.')[0] + '.jpg'
        cv2.imwrite(f'../yolo_v4/data/obj/{out_name}',
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # txt should have the same name as the images (.txt instead of .jpg)
        filename = out_name[:-4] + '.txt'
        if 'blank' in img:
            txt_file = f'../data/Bounding_Boxes/Blank/{filename}'
        if 'random' in img:
            txt_file = f'../data/Bounding_Boxes/Random/{filename}'
        if 'floor' in img:
            txt_file = f'../data/Bounding_Boxes/Floor/{filename}'
        if 'Real' in img:
            txt_file = f'../data/Bounding_Boxes/Real/{filename}'
        if 'Negative' in img:
            txt_file = f'../data/Bounding_Boxes/Negative/{filename}'
        with open(txt_file, 'r') as f_in:
            txt_data = f_in.read()
        with open(f'../yolo_v4/data/obj/{filename}', 'w') as f_out:
            f_out.write(txt_data)

        i += 1
        if i%100 == 0:
            print(f'{i}/{max_size} images copied.')

    # create file train.txt
    # this file should contain the name of all the images to use
    write_file = []
    for file in glob.glob('../yolo_v4/data/obj/*.jpg'):
        write_file.append('/'.join(file.split('/')[-3:]))
    with open('../yolo_v4/data/train.txt', 'w') as f_out:
        for file in write_file:
            f_out.write(file + '\n')
