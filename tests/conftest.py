import cv2 as cv
import numpy as np
import os
import pytest


@pytest.fixture(scope="session")
def image_dir(tmpdir_factory):
    """
    Creates one directory with a number of well images named A1.png, ...

    :param tmpdir_factory: PyTest factory for temporary directories
    :return input_dir: Temporary directory with well images
    """
    im = np.zeros((5, 10), dtype=np.uint8)
    im[2:3, 3:8] = 128
    input_dir = tmpdir_factory.mktemp("input_dir")
    # Save a couple of images
    cv.imwrite(os.path.join(input_dir, 'A1.png'), im)
    cv.imwrite(os.path.join(input_dir, 'A2.png'), im)
    cv.imwrite(os.path.join(input_dir, 'B11.png'), im + 50)
    cv.imwrite(os.path.join(input_dir, 'B12.png'), im + 100)
    return input_dir


@pytest.fixture(scope="session")
def micromanager_dir(tmpdir_factory):
    """
    Creates a directory where well images are in one subdirectory each.
    Well names are encoded in the subdirectory name.

    :param tmpdir_factory: PyTest factory for temporary directories
    :return input_dir: Directory with well images in subdirectories
    """
    im = np.zeros((5, 10), dtype=np.uint8)
    im[2:3, 3:8] = 128
    input_dir = tmpdir_factory.mktemp("input_dir")
    well_names = ['A1', 'A2', 'B11', 'B12']
    # Save a couple of images
    for well in well_names:
        sub_dir = input_dir / well + "-what-ever"
        sub_dir.mkdir()
        cv.imwrite(os.path.join(sub_dir, 'micromanager_name.tif'), im)
    return input_dir
