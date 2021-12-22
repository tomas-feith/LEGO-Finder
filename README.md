# README

## Abstract

AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

## Detailed Usage Explanation

### Constructing The Training Dataset

#### Synthetic Images

The dataset is made of synthetic models generated using [Unity Perception](https://arxiv.org/abs/2107.04259). This is a package for the graphics rendering software [Unity](https://unity.com/). It allows for image generation with several random parameters, such as object position, rotation, background and lighting. Along with these images, the bounding boxes (the box that indicates the position of each object in the image) are also created automatically.

This is the basis for the full training dataset but then we expand on it. So, for starters, it is necessary to place all these synthetic images (ours can be found [here](https://drive.google.com/drive/folders/1heB_pehuNhrJ7PqsK-NEslMq448RHbW6?usp=sharing) but you may also generate them from scratch) inside the `./data/Model_Images` folder. The images should be divided by categories, either 'floor' or 'random' background, and they should be in different folders inside `./data/Model_Images/<background_type>`. Unity also generates a set of files (which by default are called <capture_xxx>) which contain the information of all the bounding boxes (the ones for our data can be found [here](https://drive.google.com/drive/folders/1TrtjqkKUG58p9dD14L1_8ZPpSWtkdsIO?usp=sharing)). These files, and **only** these files should be placed inside the `./data/Bounding_Boxes/original` folder, again divided by background type. These images should then be named `floor_xxx` and `random_xxx`, like 'floor_000', 'floor_001', 'floor_002',...

For a more visual representation of the internal structure of the repo, see the 'Repo Architecture' section at the end of the README.

#### Real Images

Real images force the model to not only fit to synthetic data but also to real pieces, which fills in the gap between synthetic data and real data. These images were extracted from the web, and the ones we used can be found [here](https://drive.google.com/drive/folders/1uIx1OR_u_MD30901NbTG2PKRRTMHMpC9?usp=sharing). They should be placed inside `./data/Real_Images`. Their bounding boxes should be in `./data/Bounding_Boxes/Real`.

#### Negative Samples

To improve accuracy it is possible to use negative samples, i.e. images that do not contain any of the objects we wish to identify, and as such contain no bounding boxes. These images help to train the model on what **not** to identify. The ones we used can be found [here](https://drive.google.com/drive/folders/1Nszj9aVZWFuUoJblLeR92h3wo8OpQEE4?usp=sharing) but they can also be extracted from google iamges automatically via the script `./src/get_neg_imgs.py`. **WARNING**: These images have been extracted from Google as is, and as so it is highly recommended to do some manual selection and/or editing (like cropping) from all the images extracted, keeping only an adequate dataset. What we did was remove all the boundaries of the images that contained irrelevant information (like logos and markings to the website), while also removing all the images that had too strong watermarks.

#### Putting It All Together

Once all the needed files have been placed in their respective folders, the full training dataset can be generated. This is composed of two parts, the actual images and the *.txt* files containing the information about the bounding boxes. To generate both of them all that is needed is to run `python src/img_process.py`. **WARNING**: This script takes a while to run, as it is performing data augmentation on 2,000 images.

##### Bounding Boxes

The info about the bounding boxes is stored in a set of *.txt* files in the `./data/Bounding_Boxes` folder. We have one file per image and it is **imperative** that they have exactly the same name as their counterpart image. In each of these files, for each bounding boxes, it is specified the class of the object, and the position and dimensions of the box. These latter are given in units relative to the dimensions of the box, and the position given is the one of the center of the box.

##### Training Images

The training images consist of all the synthetic images, the negative samples and real iamges, as well as a set of augmentated images. These augmentations were applied to each of the synthetic images and real images, leading to an increase of x8 synthetic images and x5 real images (no cutouts were applied on the reals). The transformations applied were:
 - Color Jitter: Add random noise to each pixel, under a uniform distribution;
 - Cutout: Remove random portions of each bounding box (to prevent overcutting information, we never removed pixels from areas where bounding boxes overlapped);
 - Gaussian Blur: Apply a gaussian blur to the image;
 - Gaussian Noise: Apply noise to the image, following a normal distribution;
 - Grayscale: Grayscale the image;
 - Laplacian Filter: Apply a Laplacian Filter aka edge-detector.

Each of these was applied to every image, and the cutout was applied three times per image, due to the randomness of the cuts.


### How To Setup YOLO_v4

To setup YOLO_v4 it is necessary to first get the repository from [here](https://github.com/AlexeyAB/darknet). The contents of this repository should then be placed inside the `./yolo_v4` folder. **Warning**: the contents should be placed inside this folder, not the repo itself. That means that it should look like `./yolo_v4/[all_content]` and not `./yolo_v4/darknet/[all_content]`. If this is not obeyed then the setup will fail.

YOLO_v4 needs to be compiled. For that [these](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg) instructions should be used for Windows, and [these](https://github.com/AlexeyAB/darknet#how-to-compile-on-linuxmacos-using-cmake) for MacOS/Linux.

After that, and **after** having placed all the training images inside`./data/Training_Images`, to setup YOLO it is only necessary to run the `python src/yolo_setup.py`, and it should be ready to be trained. Certain parameters of the setup, however, can be adjusted from what we have defined. For more information about that see [this](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects). Keep in mind that our `yolo_setup.py` script also has some possibilities for customization in the beggining of the code, which can be edited to fit your needs. Namely, it is possible to only use a subset of the training images, to use or not the data augmentation, to use or not the negative samples.

### Usage of the trained model
A Google Drive repository with all the necessary files for the usage of the trained model has been created here: [Google Drive repo](https://drive.google.com/drive/folders/1ztf3WHBJsJkSXs-_6hj5kpcijvhyjzkt?usp=sharing). To start using it, just create a copy of the repository to your own Google Drive, and run `yolov4_test.ipynb`. There, it's possible to perform inference with the model on pictures and videos.

### Technical Details

FILL THIS IN AT THE END!!
### Repo Architecture

<pre>  
├─── data : folder containing all the necessary data
    ├─── Bounding_Boxes: folder containing all the information for the bounding boxes. gitignore'd in the repo, needs to be filled.
        ├─── Floor: folder containing the .txt with the info about the bounding boxes for the `floor_xxx` images. gitignore'd in the repo, is filled by running `img_process.py`.
        ├─── Negative: folder containing the .txt with the info about the bounding boxes for the `NegSample_xxx` images. gitignore'd in the repo, is filled by running `img_process.py`.
        ├─── Original: folder containing the .json extracted from Unity with the info for the bounding boxes.
            ├─── Floor: folder for the .json of images with a floor background. gitignore'd in the repo, needs to be filled.
            └─── Random:  folder for the .json of images with a random background. gitignore'd in the repo, needs to be filled.
        ├─── Random: folder containing the .txt with the info about the bounding boxes for the `random_xxx` images. gitignore'd in the repo, is filled by running `img_process.py`.
        └─── Real: folder containing the .txt with the info about the bounding boxes for the real images. gitignore'd in the repo, is filled by running `img_process.py`.
    ├─── Model_Images: folder containing all the images generated by Unity.
        ├─── Floor: Folder containing all the synthetic images with a floor background. Should be named `floor_xxx`. gitignore'd in the repo, needs to be filled.
        └─── Random: Folder containing all the synthetic images with a random background. Should be named `random_xxx`. gitignore'd in the repo, needs to be filled.
    ├─── Negative_Samples: folder containing all the negative samples to use for training. gitignore'd in the repo, needs to be filled.
    ├─── Real_Images: folder containing all the real life images of LEGOs. Should be named `<PIECE_ID>_x` gitignore'd in the repo, needs to be filled.
    └─── Training_Images: folder containing the full dataset for training.
        ├─── Floor: Training images with a floor background. Should be named `floor_xxx_<augmentation>`. gitignore'd in the repo, is filled by running `img_process.py`.
        ├─── Negative: Negative Samples. gitignore'd in the repo, is filled by running `img_process.py`.
        ├─── Random: Training images with a random background. Should be named `random_xxx_<augmentation>`. gitignore'd in the repo, is filled by running `img_process.py`.
        └─── Real: Real-world images of LEGO pieces. Should be named `<PIECE_ID>_x_<augmentation>` gitignore'd in the repo, is filled by running `img_process.py`.
├─── src: folder containing all the src files needed
    ├─── box_process.py: script used to extract and use all the info about the bounding boxes
    ├─── get_neg_imgs.py: script use to extract the negative samples
    ├─── img_process.py: script used to generate the full training dataset from all the other scattered pieces
    └─── yolo_setup.py: script used to generate the necessary config files for yolo to work. It is very customizable.
├─── train_files_examples: folder containing some examples of the different files needed for the training. This is the only data file that is not gitignored.
├─── yolo_v4: folder where the yolo_v4 should be compiled. gitignore'd in the repo, needs to be filled.
└─── README.md : [ERROR: Infinite loop] :)
</pre>
