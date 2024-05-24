import cv2 as cv
from datetime import datetime
import glob
import logging
import natsort
import numpy as np
import os
import skimage.io as io
from skimage.color import rgb2grey
import re


def read_to_grey(path_, wellimage_):
    """
    a generator that receives file path and returns the next rgb image as greyscale and its name

    :param str path_: path to folder with all images
    :param str wellimage_: name of the file with image of the well.
    :return: next image as greyscale np.ndarray
    """
    image_path = os.path.join(path_, wellimage_)
    im = io.imread(image_path)
    im = rgb2grey(im)
    return im, os.path.basename(image_path)


def read_gray_im(im_path):
    """
    Read image from full path to file location.
    :param str im_path: Path to image
    :return np.array im: Grayscale image
    """
    try:
        im = cv.imread(im_path, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
    except IOError as e:
        raise("Can't read image", e)
    if not isinstance(im, np.ndarray):
        raise IOError('Invalid image path')
    return im


def get_max_intensity(im):
    """
    Gets image max intensity, assuming image dtype is an 8 or 16 bit
    unsigned integer. If image is 16 bit, check max intensity to see if a 12 bit
    image using intensity max.
    Note: For low contrast 16 bit images (max intensity < 2^12), this function
    will assume the image is 12 bit.

    :param np.array im: Unsigned integer image
    :return int max_intensity: Max intensity of image (2^ 8, 12, or 16)
    """
    assert np.issubdtype(im.dtype, np.integer),\
        "Must use integer images, not {}".format(im.dtype)
    max_intensity = np.iinfo(im.dtype).max
    # Check if a 16 bit image is actually 12 bit
    if max_intensity == 65535 and im.max() < 4096:
        max_intensity = 4095
    assert max_intensity in {255, 4095, 65535}, \
        "Image must be uint 8, 12, or 16, not have max {}".format(max_intensity)
    return max_intensity


def get_image_paths(input_dir):
    """
    Searches input directory and its subdirectory for all images
    and returns sorted list of paths.

    :param str input_dir: Input directory, may contain images or subdirectories
        with one image each
    :return dict well_images: Well name key, path to found image value
    """
    extensions = ('.png', '.tif')

    image_names = []
    for ext in extensions:
        search_str = os.path.join(input_dir,'[A-P][0-9]*'+ext)
        image_names.extend(glob.glob(search_str, recursive=False))

    # Sort images
    image_names = natsort.natsorted(image_names)
    # Find well names from image paths
    well_images = {}
    if len(image_names) > 0:
        # Assume images are named e.g. A0.png
        for im_name in image_names:
            well_name = os.path.basename(im_name)[:-4]
            # double-check that the file represents a well
            if re.match(r'[A-P][0-9]{1,2}',well_name):
                well_images[well_name] = im_name
    else:
        # Micromanager naming convention, find well from subdir name
        image_names = []
        for ext in extensions:
            search_str = os.path.join(input_dir,'[A-P][0-9]*','*'+ext)
            image_names.extend(glob.glob(search_str, recursive=False))

        # Sort images
        image_names = natsort.natsorted(image_names)
        for im_name in image_names:
            well_name = im_name.split('/')[-2]
            # split again for well name, assume - separation
            well_name = well_name.split('-')[0]
            #  double-check that the file represents a well
            if re.match(r'[A-P][0-9]{1,2}',well_name):
                well_images[well_name] = im_name

    # Check that wells are found, refer to docs if not
    assert len(well_images) > 0,\
        "No wells found, check documentation for naming conventions"\
        "And conversion scripts 12to16bit.py and rename_only.py"

    return well_images


def make_run_dir(input_dir, output_dir, rerun=False):
    """
    For a specific processing run, create a subdirectory in the output directory
    which specifies which input directory was used and when the processing took
    place.

    :param str input_dir: Path to input directory, to be processed
    :param str output_dir: Path to main output directory
    :param bool rerun: Determine if this is a rerun
    :return str run_dir: Path to directory where processed data is stored
    """
    if rerun:
        return output_dir
    run_dir = os.path.join(
        output_dir,
        '_'.join(['multisero',
                  os.path.basename(os.path.normpath(input_dir)),
                  f"{datetime.now().year:04d}" +
                  f"{datetime.now().month:02d}" +
                  f"{datetime.now().day:02d}",
                  f"{datetime.now().hour:02d}" +
                  f"{datetime.now().minute:02d}"])
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_logger(log_dir, logger_name='multisero.log', log_level=20):
    """
    Creates a logger which writes to a file, not to console.

    :param str log_dir: Path to directory where log file will be written
    :param str logger_name: name of the logger instance
    :param int log_level: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    :return logging instance logger
    """
    log_path = os.path.join(log_dir, logger_name)
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    return logger

