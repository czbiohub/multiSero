import numpy as np
import pandas as pd
from skimage import measure
from skimage.morphology import disk

import multiSero.array_analyzer.extract.constants as constants


class SpotRegionprop:

    def __init__(self, row_idx, col_idx, label=None):
        """
        Object holding spot images, masks, and their properties:
        centroid, bounding box, mean and median intensity.

        :param int label: Spot label
        """
        self.df_cols = constants.SPOT_DF_COLS
        self.image = None
        self.background = None
        self.label = label
        self.mask = None
        self.masked_image = None
        self.spot_dict = dict.fromkeys(self.df_cols)
        self.spot_dict['grid_row'] = row_idx
        self.spot_dict['grid_col'] = col_idx

    @staticmethod
    def make_mask(im_size):
        """
        Creates disk shaped mask the size of image.

        :param int im_size: Image size (assume square shape)
        :return np.array mask: Binary disk shaped mask
        """
        mask = disk(int(im_size / 2), dtype=np.uint8)
        return mask

    def compute_stats(self):
        """
        Compute mean, median and OD values for images and backgrounds.
        Optical density is affected by Beer-Lambert law
        i.e. I = I0*e^-{c*thickness). I0/I = e^{c*thickness).
        """
        intensity_vals = self.image[self.mask > 0]
        self.spot_dict['intensity_mean'] = np.mean(intensity_vals)
        self.spot_dict['intensity_median'] = np.median(intensity_vals)
        bg_vals = self.background[self.mask > 0]
        self.spot_dict['bg_mean'] = np.mean(bg_vals)
        self.spot_dict['bg_median'] = np.median(bg_vals)
        with np.errstate(divide='ignore'):
            self.spot_dict['od_norm'] = np.log10(
                self.spot_dict['bg_median'] / self.spot_dict['intensity_median'],
            )
        if np.isinf(self.spot_dict['od_norm']):
            # TODO: What should be the default when intensity is zero?
            self.spot_dict['od_norm'] = 2.

    def generate_props_from_disk(self, image, background, bbox, centroid):
        """
        Assign properties from disk shaped mask.

        :param np.ndarray image: Large single spot ROI intensity image
        param np.ndarray background: Background corresponding to image
        :param list bbox: Bounding box of single spot image
        :param list/tuple centroid: Center coordinate
        """
        # Create disk mask
        self.mask = self.make_mask(image.shape[0])
        self.image = image
        self.background = background
        self.masked_image = self.image * self.mask

        self.spot_dict['centroid_row'] = centroid[0]
        self.spot_dict['centroid_col'] = centroid[1]
        self.spot_dict['bbox_row_min'] = bbox[0]
        self.spot_dict['bbox_col_min'] = bbox[1]
        self.spot_dict['bbox_row_max'] = bbox[2]
        self.spot_dict['bbox_col_max'] = bbox[3]

        self.compute_stats()

    def generate_props_from_mask(self, image, background, mask, bbox):
        """
        converts binarized image into region-properties using
        scikit-image.

        :param np.ndarray image: Large single spot ROI intensity image
        :param np.ndarray background: Background corresponding to image
        :param np.ndarray mask: Binary mask corresponding to image
        :param list bbox: Bounding box of single spot image
        """
        properties = ('label', 'centroid', 'mean_intensity',
                      'intensity_image', 'image', 'area', 'bbox')
        skimage_props = measure.regionprops_table(
            mask,
            intensity_image=image,
            properties=properties,
        )
        skimage_props = pd.DataFrame(skimage_props)

        self.spot_dict['centroid_row'] = bbox[0] + skimage_props.at[0, 'centroid-0']
        self.spot_dict['centroid_col'] = bbox[1] + skimage_props.at[0, 'centroid-1']

        min_row = skimage_props.at[0, 'bbox-0']
        min_col = skimage_props.at[0, 'bbox-1']
        max_row = skimage_props.at[0, 'bbox-2']
        max_col = skimage_props.at[0, 'bbox-3']

        self.spot_dict['bbox_row_min'] = bbox[0] + min_row
        self.spot_dict['bbox_col_min'] = bbox[1] + min_col
        self.spot_dict['bbox_row_max'] = bbox[0] + max_row
        self.spot_dict['bbox_col_max'] = bbox[1] + max_col

        self.image = image[min_row:max_row, min_col:max_col]
        self.background = background[min_row:max_row, min_col:max_col]
        self.mask = mask[min_row:max_row, min_col:max_col]
        self.masked_image = self.image * self.mask

        self.compute_stats()
