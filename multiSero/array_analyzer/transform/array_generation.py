import itertools
import numpy as np
import pandas as pd

import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.extract.img_processing as img_processing
import multiSero.array_analyzer.extract.txt_parser as txt_parser
import multiSero.array_analyzer.utils.spot_regionprop as regionprop


def build_centroid_binary_blocks(cent_list, image_, params_, return_type='region'):
    """
    places blocks at the center of each centroid supplied by "cent_list".  Returns this mask

    :param cent_list: list
        of centroids (x, y)
    :param image_: ndarray
        underlying image whose shape is output
    :param params_: dict
        parameters parsed from .txt metadata
    :param return_type: str
        to return the region mask or the product of mask*image
    :return: ndarray
    """

    # build_block_array uses values in the params_ to motivate array dimensions, spacings

    # values in mm
    v_pitch = params_['v_pitch']
    h_pitch = params_['h_pitch']
    spot_width = params_['spot_width']
    PIX_SIZE = params_['pixel_size_scienion']

    # values in pixels
    v_pix = v_pitch/PIX_SIZE
    h_pix = h_pitch/PIX_SIZE
    spot_pix = spot_width/PIX_SIZE

    # make the box 1.3x the size of the spot, unless it will cause overlap
    side = int(1.3*spot_pix if 1.3*spot_pix < v_pix-1 and 1.3*spot_pix < h_pix-1 else spot_pix)

    target = np.zeros(image_.shape)
    for centroid in cent_list:
        (center_y, center_x) = centroid

        # check that the blank fits within the bounds of the target array
        x_min = int(center_x - (side/2)) if int(center_x - (side/2)) > 0 else 0
        x_max = int(center_x + (side/2)) if int(center_x + (side/2)) < target.shape[0] else target.shape[0]
        y_min = int(center_y - (side/2)) if int(center_y - (side/2)) > 0 else 0
        y_max = int(center_y + (side/2)) if int(center_y + (side/2)) < target.shape[1] else target.shape[1]

        blank = np.ones((x_max - x_min, y_max - y_min))

        target[x_min:x_max, y_min:y_max] = blank

    if return_type == 'product':
        return target*image_
    elif return_type == 'region':
        return target


def get_spot_intensity(coords, im, background, search_range=2):
    """
    Extract signal and background intensity at each spot given the spot coordinate
    with the following steps:
    1. crop image around each grid point
    2. Segment 1 single spot from each image
    3. Get median intensity, background and OD within the spot mask
    4. If segmentation in 2. returns no mask, use a circular mask with average spot size as the spot mask and do 3.

    :param coords: list or tuple
        [row, col] coordinates of spots
    :param im: ndarray
        intensity image of the spots (signals)
    :param background: ndarray
        background image without spots
    :param float search_range: Factor of bounding box size in which to search for
        spots. E.g. 2 searches 2 * 2 * bbox width * bbox height
    :return pd.DataFrame spots_df: Dataframe containing metrics for
        all spots in the grid
    :return np.array spot_props: A SpotRegionprop object with ROIs for
        each spot in the grid
    """
    # values in mm
    spot_width = constants.params['spot_width']
    pix_size = constants.params['pixel_size']
    n_rows = constants.params['rows']
    n_cols = constants.params['columns']
    # make spot size always odd
    spot_size = 2 * int(0.3 * spot_width / pix_size) + 1
    bbox_width = bbox_height = spot_size
    # Strel disk size for spot segmentation
    disk_size = int(np.rint(spot_size / 2.5))

    # Array of SpotRegionprop objects to hold ROIs
    spot_props = txt_parser.create_array(n_rows, n_cols, dtype=object)
    # Dataframe to hold spot metrics for the well
    spots_df = pd.DataFrame(columns=constants.SPOT_DF_COLS)
    row_col_iter = itertools.product(np.arange(n_rows), np.arange(n_cols))
    for count, (row_idx, col_idx) in enumerate(row_col_iter):
        coord = coords[count, :]
        # make bounding boxes larger to account for interpolation errors
        spot_height = int(np.round(search_range * bbox_height))
        spot_width = int(np.round(search_range * bbox_width))
        # Create large bounding box around spot and segment
        im_spot_lg, bbox_lg = img_processing.crop_image_at_center(
            im=im,
            center=coord,
            height=spot_height,
            width=spot_width,
        )
        mask_spot = img_processing.thresh_and_binarize(
            image=im_spot_lg,
            method='bright_spots',
            disk_size=disk_size,
            thr_percent=75,
            get_lcc=True,
        )
        # Create spot and background instance
        spot_prop = regionprop.SpotRegionprop(
            row_idx=row_idx,
            col_idx=col_idx,
            label=count,
        )

        # Mask spot should cover a certain percentage of ROI
        if np.mean(mask_spot) > constants.SPOT_MIN_PERCENT_AREA:
            # Mask detected
            bg_spot_lg, _ = img_processing.crop_image_at_center(
                im=background,
                center=coord,
                height=spot_height,
                width=spot_width,
            )
            spot_prop.generate_props_from_mask(
                image=im_spot_lg,
                background=bg_spot_lg,
                mask=mask_spot,
                bbox=bbox_lg,
            )
        else:
            # Crop around assumed spot size
            im_spot, bbox = img_processing.crop_image_at_center(
                im,
                coord,
                bbox_height,
                bbox_width,
            )
            bg_spot, _ = img_processing.crop_image_at_center(
                background,
                coord,
                bbox_height,
                bbox_width,
            )
            spot_prop.generate_props_from_disk(
                image=im_spot,
                background=bg_spot,
                bbox=bbox,
                centroid=coord,
            )
        spots_df = spots_df.append(spot_prop.spot_dict, ignore_index=True)
        spot_props[row_idx, col_idx] = spot_prop

    return spots_df, spot_props
