import multiSero.array_analyzer.extract.image_parser as image_parser
import multiSero.array_analyzer.extract.img_processing as processing
import multiSero.array_analyzer.extract.constants as constants
from multiSero.array_analyzer.extract.metadata import MetaData
import multiSero.array_analyzer.utils.io_utils as io_utils

import time
import skimage.io as io
import pandas as pd
import string
import os
import numpy as np


def well_analysis(input_dir, output_dir, method='segmentation'):
    """
    Workflow that pulls all images scanned on a multi-well plate in a standard ELISA format (one antigen per well)
    It loops over the images in the input_folder (for images acquired using Micro-Manager ONLY).
        Extracts the center of the well, calculates the median intensity of that spot and background, then computes OD.
        Finally, it writes a summary report (.xlsx) containing plate info and the computed values.

    :param input_dir: str path to experiment directory
    :param output_dir: str output path to write report and diagnostic images
    :param method: str 'segmentation' or 'crop'.  Methods to estimate the boundaries of the well
    :return:
    """
    start = time.time()

    # metadata isn't used for the well format
    MetaData(input_dir, output_dir)

    # Read plate info
    plate_info = pd.read_excel(
        os.path.join(input_dir, 'Plate_Info.xlsx'),
        usecols='A:M',
        sheet_name=None,
        index_col=0,
    )
    # Write an excel file that can be read into jupyter notebook with minimal parsing.
    xlwriter_int = pd.ExcelWriter(os.path.join(constants.RUN_PATH, 'intensities.xlsx'))
    # get well directories
    well_images = io_utils.get_image_paths(input_dir)

    int_well = []
    for well_name, im_path in well_images.items():
        # read image
        image = io_utils.read_gray_im(im_path)
        print(well_name)

        # measure intensity
        if method == 'segmentation':
            # segment well using otsu thresholding
            well_mask = image_parser.get_well_mask(image, segmethod='otsu')
            int_well_ = image_parser.get_well_intensity(image, well_mask)

        elif method == 'crop':
            # get intensity at square crop in the middle of the image
            img_size = image.shape
            radius = np.floor(0.1 * np.min(img_size)).astype('int')
            cx = np.floor(img_size[1]/2).astype('int')
            cy = np.floor(img_size[0]/2).astype('int')
            im_crop = processing.crop_image(image, cx, cy, radius, border_=0)
            well_mask = np.ones_like(im_crop, dtype='bool')
            int_well_ = image_parser.get_well_intensity(im_crop, well_mask)

        int_well.append(int_well_)

        # SAVE FOR DEBUGGING
        if constants.DEBUG:
            output_name = os.path.join(constants.RUN_PATH, well_name)

            # Save mask of the well, cropped grayscale image, cropped spot segmentation.
            io.imsave(output_name + "_well_mask.png",
                      (255 * well_mask).astype('uint8'))

            # Save masked image
            if method == 'segmentation':
                img_ = image.copy()
                img_[~well_mask] = 0
            elif method == 'crop':
                img_ = im_crop.copy()
                img_[~well_mask] = 0
            else:
                raise NotImplementedError(f'method of type {method} not supported')
            io.imsave(output_name + "_masked_image.png",
                      (img_/256).astype('uint8'))

    df_int = pd.DataFrame(
        np.reshape(int_well, (8, 12)),
        index=list(string.ascii_uppercase[:8]),
        columns=range(1, 13),
    )
    plate_info.update({'intensity': df_int})

    # compute optical density
    sample_info = plate_info['sample']
    blanks = np.any(np.dstack((
        sample_info == 'Blank',
        sample_info == 'blank',
        sample_info == 'BLANK')), axis=2)
    if blanks.any():
        # blank intensity is averaged over all blank wells
        int_blank = np.mean(df_int.to_numpy()[blanks])
        df_od = np.log10(int_blank / df_int)
        plate_info.update({'od': df_od})

    # save analysis results
    for k, v in plate_info.items():
        v.to_excel(xlwriter_int, sheet_name=k)
    xlwriter_int.close()

    stop = time.time()
    print(f"\ttime to process={stop - start}")
