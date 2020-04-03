# bchhun, {2020-03-22}

import os
from copy import copy
import numpy as np
import re
import itertools
import math
import cv2 as cv
from types import SimpleNamespace
from scipy import spatial, stats

from scipy.signal import find_peaks
import skimage.io as io
import skimage.util as u

from skimage.color import rgb2grey
from skimage.filters import threshold_minimum, median, gaussian, threshold_local
from skimage.filters import threshold_minimum, threshold_otsu
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, selem, disk, binary_opening
from skimage.morphology import binary_closing, binary_dilation, selem, disk, binary_opening
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter1d
from skimage import measure

from .img_processing import create_unimodal_mask, create_otsu_mask, create_multiotsu_mask
from ..utils.mock_regionprop import MockRegionprop
# from .img_processing import create_unimodal_mask, create_otsu_mask
# from .img_processing import  create_unimodal_mask
# from ..utils.mock_regionprop import MockRegionprop
"""
method is
1) read_to_grey(supplied images)
# find center of well
2) thresh and binarize from 1
3) find well border from 2
4) crop image from 3

# find center of spots from crop
5) thresh and binarize from 4
6) clean spot binary from 5
7) generate props from 6
8) generate props dict from 7
9) assign props dict to array from 8
"""


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
    return im


def thresh_and_binarize(image_, method='rosin', invert=True):
    """
    receives greyscale np.ndarray image
        inverts the intensities
        thresholds on the minimum peak
        converts the image into binary about that threshold

    :param image_: np.ndarray
    :param method: str
        'bimodal' or 'unimodal'
    :return: spots threshold_min on this image
    """

    if invert:
        image_ = u.invert(image_)

    if method == 'bimodal':
        thresh = threshold_minimum(image_, nbins=512)

        spots = copy(image_)
        spots[image_ < thresh] = 0
        spots[image_ >= thresh] = 1

    elif method == 'otsu':
        spots = create_otsu_mask(image_, scale=1)

    elif method == 'multi_otsu':
        n_class = 3
        spots = create_multiotsu_mask(image_, n_class=n_class, fg_class=n_class - 1)

    elif method == 'rosin':
        spots = create_unimodal_mask(image_, str_elem_size=3)

    elif method == 'bright_spots':
        spots = image_ > np.percentile(image_, 95)
        str_elem = disk(10)
        # spots = binary_closing(spots, str_elem)
        spots = binary_opening(spots, str_elem)
        spots = clear_border(spots)

    else:
        raise ModuleNotFoundError("not a supported method for thresh_and_binarize")

    return spots


def find_well_border(image, segmethod='bimodal', detmethod='region'):
    """
    finds the border of the well to motivate future cropping around spots
        hough_radii are potential radii of the well in pixels
            this should be motivated based on magnification
        edge filter
        fit hough circle
        find the peak of the SINGULAR hough circle

    :param image: np.ndarray
        raw image, not inverted
    :param segmethod: str
        'otsu' or 'hough'
    :return: center x, center y, radius of the one hough circle
    """
    well_mask = thresh_and_binarize(image, method=segmethod, invert=False)
    # Now remove small objects.
    str_elem_size = 10
    str_elem = disk(str_elem_size)
    well_mask = binary_opening(well_mask, str_elem)
    # well_mask = binary_fill_holes(well_mask)

    if detmethod == 'region':
        labels = measure.label(well_mask)
        props = measure.regionprops(labels)

        # let's assume ONE circle for now (take only props[0])
        props = select_props(props, attribute="area", condition="greater_than", condition_value=10**5)
        props = select_props(props, attribute="eccentricity", condition="less_than", condition_value=0.6)
        well_mask[labels != props[0].label] = 0
        cy, cx = props[0].centroid # notice that the coordinate order is different from hough.
        radii = int((props[0].minor_axis_length + props[0].major_axis_length)/ 4 / np.sqrt(2))
        # Otsu threshold fails occasionally and leads to asymmetric region. Averaging both axes makes the segmentation robust.
        # If above files, try bounding box.

    elif detmethod == 'hough':
        hough_radii = [300, 400, 500, 600]

        well_mask = thresh_and_binarize(image, method='bimodal')

        edges = canny(well_mask, sigma=3)
        hough_res = hough_circle(edges, hough_radii)
        aaccums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        cx, cy = cx[0], cy[0]
        radii = radii[0]
    else:
        cx, cy, radii = None, None, None

    return cx, cy, radii, well_mask


def crop_image(arr, cx_, cy_, radius_, border_=200):
    """
    crop the supplied image to include only the well and its spots

    :param arr: image
    :param cx_: float
    :param cy_: float
    :param radius_:
    :param border_:
    :return:
    """
    cx_ = int(np.rint(cx_))
    cy_ = int(np.rint(cy_))
    crop = arr[
           cy_ - (radius_ - border_): cy_ + (radius_ - border_),
           cx_ - (radius_ - border_): cx_ + (radius_ - border_)
           ]

    return crop


def clean_spot_binary(arr, kx=10, ky=10):
    return binary_closing(arr, selem=np.ones((kx, ky)))


def generate_spot_background(spotmask, distance=3, annulus=5):
    """
    
    compute an annulus around each spot to estimate background.
    
    Parameters
    ----------
    spotmask : binary mask of spots
    distance : distance from the edge of segmented spot.
    annulus : width of the annulus

    Returns
    -------
    spotbackgroundmask: binary mask of annuli around spots.
    
    TODO: 'comets' should be ignored, and this approach may not be robust to it.
    """
    se_inner = selem.disk(distance, dtype=bool)
    se_outer = selem.disk(distance+annulus, dtype=bool)
    inner = binary_dilation(spotmask, se_inner)
    outer = binary_dilation(spotmask, se_outer)
    spot_background = np.bitwise_xor(inner, outer)

    return spot_background


def generate_props(mask, intensity_image_=None):
    """
    converts binarized image into a list of region-properties using scikit-image
        first generates labels for the cleaned (binary_closing) binary image
        then generates regionprops on the remaining

    :param mask: np.ndarray
        binary version of cropped image
    :param intensity_image_: np.ndarray
        intensity image corresponding to this binary
    :return: list
        of skimage region-props object
    """
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=intensity_image_)
    return props


def select_props(props_, attribute, condition, condition_value):
    """

    :param props_: RegionProps
    :param attribute: str
        a regionprop attribute
        https://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
    :param condition: str
        one of "greater_than", "equals", "less_than"
    :param condition_value: int, float
        the value to evaluate around
    :return:
    """

    if condition == 'greater_than':
        props = [p for p in props_ if getattr(p, attribute) > condition_value]
    elif condition == 'equals':
        props = [p for p in props_ if getattr(p, attribute) == condition_value]
    elif condition == 'less_than':
        props = [p for p in props_ if getattr(p, attribute) < condition_value]
    elif condition == 'is_in':
        props = [p for p in props_ if getattr(p, attribute) in condition_value]
    else:
        props = props_

    return props


def generate_props_dict(props_, n_rows, n_cols, min_area=100, img_x_max=2048, img_y_max=2048, flag_duplicates=True):
    """
    based on the region props, creates a dictionary of format:
        key = (centroid_x, centroid_y)
        value = region_prop object

    :param props_: list of region props
        approximately 36-48 of these, depending on quality of the image
    :param n_rows: int
    :param n_cols: int
    :param min_area: int
    :param img_x_max: int
    :param img_y_max: int
    :return: dict
        of format (cent_x, cent_y): prop
    """

    # find minx, miny to "zero center" the array
    minx = img_x_max
    miny = img_y_max
    # find maxx, maxy to scale to array index values
    maxx = 0
    maxy = 0
    for prop in props_:
        if prop.area > min_area:
            if prop.centroid[0] < minx:
                minx = prop.centroid[0]
            if prop.centroid[1] < miny:
                miny = prop.centroid[1]
            if prop.centroid[0] > maxx:
                maxx = prop.centroid[0]
            if prop.centroid[1] > maxy:
                maxy = prop.centroid[1]

    # scaled max-x, max-y
    smaxx = maxx - minx
    smaxy = maxy - miny

    chk_list = []
    cent_map = {}
    for prop in props_:
        if prop.area > min_area:
            cx, cy = prop.centroid
            csx = cx - minx
            csy = cy - miny

            # convert the centroid position to an integer that maps to array indices
            norm_cent_x = int(round((n_rows - 1) * (csx / smaxx)))
            norm_cent_y = int(round((n_cols - 1) * (csy / smaxy)))

            # print(f"\ncentroid = {prop.centroid}\n\tnorm_cent = {norm_cent_x, norm_cent_y}")

            chk_list.append((norm_cent_x, norm_cent_y))
            cent_map[(norm_cent_x, norm_cent_y)] = prop

    if flag_duplicates:
        if len(chk_list) != len(set(chk_list)):
            print("ERROR, DUPLICATE ENTRIES")
            raise AttributeError("generate props array failed\n"
                                 "duplicate spots found in one position\n")

    return cent_map


def find_fiducials_markers(props_,
                           fiducial_locations,
                           n_rows,
                           n_cols,
                           v_pitch,
                           h_pitch,
                           img_size,
                           pix_size):
    """
    based on the region props, creates a dictionary of format:
        key = (centroid_x, centroid_y)
        value = region_prop object

    :param props_: list of region props
        approximately 36-48 of these, depending on quality of the image
    :param fiducial_locations: list specifying location of fiducial markers, e.g.
        [(0,0), (0,5), (5,0)] for markers at 3 corners of 6x6 array
    :param n_rows: int
    :param n_cols: int
    :param v_pitch: float
        vertical spot center distance in mm
    :param h_pitch: float
        horizontal spot center distance in mm
    :param img_size tuple
        image size in pixels
    :param pix_size: float
        size of pix in mm
    :return: dict
        of format (cent_x, cent_y): prop for fiducials only
    NOTE (Jenny): Why input pixel size here? All you do is multiply both
    sets of coordinates with it, then divide by it.
    """

    centroids_in_mm = np.array([p.centroid for p in props_]) * pix_size
    spots_x, spots_y = (centroids_in_mm[:,1], centroids_in_mm[:,0])

    cent_y, cent_x = np.array(img_size) / 2 * pix_size
    start_x = cent_x - h_pitch * (n_cols - 1) / 2
    start_y = cent_y - v_pitch * (n_rows - 1) / 2

    x_vals = np.array([f[0] for f in fiducial_locations]) * h_pitch + start_x
    y_vals = np.array([f[1] for f in fiducial_locations]) * v_pitch + start_y
    grid_x = x_vals.flatten()
    grid_y = y_vals.flatten()

    source = np.array([grid_x, grid_y]).T
    target = np.array([spots_x, spots_y]).T
    t_matrix = icp(source, target)

    grid_estimate = cv.transform(np.expand_dims(source, 0), t_matrix[:2])

    reg_x = grid_estimate[0, :, 0]
    reg_y = grid_estimate[0, :, 1]

    cent_map = {}
    for i, f in enumerate(fiducial_locations):
        cent_map[f[::-1]] = SimpleNamespace(centroid=(reg_y[i]/pix_size, reg_x[i]/pix_size))

    return cent_map


def get_spot_coords(im,
                    min_thresh=0,
                    max_thresh=255,
                    min_area=50,
                    max_area=10000,
                    min_circularity=0.1,
                    min_convexity=0.5):
    """
    Use OpenCVs simple blob detector (thresholdings and grouping by properties)
    to detect all dark spots in the image

    :param np.array im: uint8 mage containing spots
    :param int min_thresh: Minimum threshold
    :param int max_thresh: Maximum threshold
    :param int min_area: Minimum spot area in pixels
    :param int max_area: Maximum spot area in pixels
    :param float min_circularity: Minimum circularity of spots
    :param float min_convexity: Minimum convexity of spots
    :return np.array spot_coords: x, y coordinates of spot centroids (nbr spots x 2)
    """
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh
    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = min_convexity

    detector = cv.SimpleBlobDetector_create(params)

    # Normalize image
    im_norm = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
    # Detect blobs
    keypoints = detector.detect(im_norm)

    spot_coords = np.zeros((len(keypoints), 2))
    # Convert to np.arrays
    for c in range(len(keypoints)):
        pt = keypoints[c].pt
        spot_coords[c, 0] = pt[0]
        spot_coords[c, 1] = pt[1]

    return spot_coords


def find_profile_peaks(profile, margin, prominence):
    # invert because black spots
    profile = profile.max() - profile
    max_pos = int(np.mean(np.where(profile == profile.max())[0]))
    # Make sure max is not due to leaving the center
    add_margin = 0
    half_margin = int(margin / 2)
    if max_pos > len(profile) - half_margin:
        profile = profile[:-half_margin]
    elif max_pos < half_margin:
        profile = profile[half_margin:]
        add_margin = half_margin
    profile = gaussian_filter1d(profile, 3)
    min_prom = profile.max() * prominence
    peaks, _ = find_peaks(profile, prominence=min_prom, distance=50)
    if len(peaks) >= 4:
        spot_dists = peaks[1:] - peaks[:-1]
    else:
        spot_dists = None
    mean_pos = peaks[0] + (peaks[-1] - peaks[0]) / 2 + add_margin
    return mean_pos, spot_dists


def grid_estimation(im,
                    spot_coords,
                    margin=50,
                    prominence=.15):
    """
    Based on images intensities and detected spots, make an estimation
    of grid location so that ICP algorithm is initialized close enough for convergence.
    TODO: This assumes that you always detect the first peaks
    this may be unstable so think of other ways to initialize...

    :param np.array im: Grayscale image
    :param np.array spot_coords: Spot x,y coordinates (nbr spots x 2)
    :param int margin: Margin for cropping outside all detected spots
    :param float prominence: Fraction of max intensity to filter out insignificant peaks
    :return tuple start_point: Min x, y coordinates for initial grid estimate
    :return float spot_dist: Estimated distance between spots
    """
    im_shape = im.shape
    x_min = int(max(margin, np.min(spot_coords[:, 0]) - margin))
    x_max = int(min(im_shape[1] - margin, np.max(spot_coords[:, 0]) + margin))
    y_min = int(max(margin, np.min(spot_coords[:, 1]) - margin))
    y_max = int(min(im_shape[0] - margin, np.max(spot_coords[:, 1]) + margin))
    im_roi = im[y_min:y_max, x_min:x_max]
    # Create intensity profiles along x and y and find peaks
    profile_x = np.mean(im_roi, axis=0)
    mean_x, dists_x = find_profile_peaks(profile_x, margin, prominence)
    profile_y = np.mean(im_roi, axis=1)
    mean_y, dists_y = find_profile_peaks(profile_y, margin, prominence)

    mean_point = (x_min + mean_x, y_min + mean_y)
    spot_dist = np.hstack([dists_x, dists_y])
    # Remove invalid distances
    spot_dist = spot_dist[np.where(spot_dist != None)]
    if spot_dist.size == 0:
        # Failed at estimating spot dist. Return default or error out?
        spot_dist = 80
    else:
        spot_dist = np.median(spot_dist)

    return mean_point, spot_dist


# def grid_from_centroids(props_, im, n_rows, n_cols, min_area=100, im_height=2048, im_width=2048):
def grid_from_centroids(props_, im, n_rows, n_cols, dist_flr=True):
    """
    based on the region props, creates a dictionary of format:
        key = (centroid_x, centroid_y)
        value = region_prop object

    :param props_: list of region props
        approximately 36-48 of these, depending on quality of the image
    :param im: array of the intensity image
    :param n_rows: int
    :param n_cols: int
    :param min_area: int
    :param im_height: int
    :param im_width: int
    :return: dict
        of format (cent_x, cent_y): prop
    """


    centroids = np.array([prop.weighted_centroid for prop in props_])
    bbox_area = np.array([prop.bbox_area for prop in props_])
    # calculate mean bbox width for cropping undetected spots
    bbox_area_mean = np.mean(bbox_area)
    bbox_width = bbox_height = np.sqrt(bbox_area_mean)


    y_min_idx = np.argmin(centroids[:, 0])
    y_min = centroids[y_min_idx, 0]
    y_max_idx = np.argmax(centroids[:, 0])
    y_max = centroids[y_max_idx, 0]
    x_min_idx = np.argmin(centroids[:, 1])
    x_min = centroids[x_min_idx, 1]
    x_max_idx = np.argmax(centroids[:, 1])
    x_max = centroids[x_max_idx, 1]
    # apply nearest neighbor distance filter to remove false points if >= 10 points are detected
    if dist_flr and centroids.shape[0] >= 10:
        y_sort_ids = np.argsort(centroids[:, 0])
        x_sort_ids = np.argsort(centroids[:, 1])
        dist_tree = spatial.cKDTree(centroids)
        dist, ids = dist_tree.query(centroids, k=2)
        dist = dist[:, 1]
        dist_median = np.median(dist)
        dist_std = 0.8 * dist.std()
        if dist_std > 5:
            y_min_idx = 0
            while dist[y_sort_ids[y_min_idx]] > dist_median + dist_std or \
                    dist[y_sort_ids[y_min_idx]] < dist_median - dist_std:
                y_min_idx += 1
            y_min = centroids[y_sort_ids[y_min_idx], 0]

            y_max_idx = len(ids) - 1
            while dist[y_sort_ids[y_max_idx]] > dist_median + dist_std or \
                    dist[y_sort_ids[y_max_idx]] < dist_median - dist_std:
                y_max_idx -= 1
            y_max = centroids[y_sort_ids[y_max_idx], 0]

            x_min_idx = 0
            while dist[x_sort_ids[x_min_idx]] > dist_median + dist_std or \
                    dist[x_sort_ids[x_min_idx]] < dist_median - dist_std:
                x_min_idx += 1
            x_min = centroids[x_sort_ids[x_min_idx], 1]

            x_max_idx = len(ids) - 1
            while dist[x_sort_ids[x_max_idx]] > dist_median + dist_std or \
                    dist[x_sort_ids[x_max_idx]] < dist_median - dist_std:
                x_max_idx -= 1
            x_max = centroids[x_sort_ids[x_max_idx], 1]

    # scaled max-x, max-y
    y_range = y_max - y_min
    x_range = x_max - x_min
    grid_ids = list(itertools.product(range(n_rows), range(n_cols)))
    grid_ids_detected = []
    cent_map = {}
    # for prop in props_:
    #         cen_y, cen_x = prop.weighted_centroid
    #         # convert the centroid position to an integer that maps to array indices
    #         grid_y_idx = int(round((n_rows - 1) * ((cen_y - y_min) / y_range)))
    #         grid_x_idx = int(round((n_cols - 1) * ((cen_x - x_min) / x_range)))
    #         grid_id = (grid_y_idx, grid_x_idx)
    #         if grid_id in grid_ids:
    #             grid_ids_detected.append(grid_id)
    #             cent_map[grid_id] = prop

    # if len(grid_ids_detected) != len(set(grid_ids_detected)):
    #     print("ERROR, DUPLICATE ENTRIES")
    #     raise AttributeError("generate props array failed\n"
    #                          "duplicate spots found in one position\n")
    # Add missing spots
    for grid_id in grid_ids:
        if grid_id not in grid_ids_detected:
            # make mock regionprop objects to hold the properties
            prop = MockRegionprop(label=props_[-1].label)
            prop.centroid = (grid_id[0]/(n_rows - 1) * y_range + y_min,
                             grid_id[1]/(n_cols - 1) * x_range + x_min)
            prop.label += 1
            prop.mean_intensity = 1
            prop.intensity_image = crop_image(im,
                                              prop.centroid[1],
                                              prop.centroid[0],
                                              int(bbox_width / 2),
                                              border_=0)
            prop.mean_intensity = np.mean(prop.intensity_image)

            # hardcode the bbox to be box of side = 40 around centroid
            int_shape = prop.intensity_image.shape
            prop.bbox = (int(round(prop.centroid[0]-(int_shape[0]/2))),
                         int(round(prop.centroid[1]-(int_shape[1]/2))),
                         int(round(prop.centroid[0]+(int_shape[0]/2))),
                         int(round(prop.centroid[1]+(int_shape[1]/2))))

            cent_map[grid_id] = prop

    return cent_map


def assign_props_to_array(arr, cent_map_):
    """
    takes an empty array and assigns region_property objects to each position, based on print array position

    :param arr: np.ndarray
        of shape = print array shape
    :param cent_map_: dict
        generated by "generate_props_array"
    :return:
    """

    for key, value in cent_map_.items():
        arr[key[0], key[1]] = value

    return arr


def assign_props_to_array_2(arr, cent_map_):
    """
    takes an empty array and assigns region_property objects to each position, based on print array position
        deals with multiple props assigned to a position and takes only more intense images

    :param arr: np.ndarray
        of shape = print array shape
    :param cent_map_: dict
        generated by "generate_props_array"
    :return:
    """

    for key, value in cent_map_.items():
        if arr[key[0], key[1]] and np.mean(value.intensity_image) > np.mean(arr[key[0], key[1]].intensity_image):
            arr[key[0], key[1]] = value
        elif not arr[key[0], key[1]]:
            arr[key[0], key[1]] = value
        else:
            pass

    return arr


def build_block_array(params_, pix_size=0.0049):
    """
    builds a binary array of squares centered on the expected spot position
    The array dimensions are based on parsed .xml values from the print run

    Daheng camera: IMX226, 12 MP (4000 x 3000), 1.85 um pixel size
    SciReader camera: Camera sensor / mag => 4.9 um/pixel, 2592 x 1944 pixels = 12.701 mm x 9.525 mm.

    :param params_: dict
        param dictionary from "create_xml_dict"
    :param pix_size: float
        size of pix in mm
    :return: np.ndarray, origin

    """

    # fix the pixel size, for now, in mm
    PIX_SIZE = pix_size
    PIX_SIZE = 0.00185

    n_rows = params_['rows']
    n_cols = params_['columns']

    # values in mm
    v_pitch = params_['v_pitch']
    h_pitch = params_['h_pitch']
    spot_width = params_['spot_width']

    # values in pixels
    v_pix = v_pitch/PIX_SIZE
    h_pix = h_pitch/PIX_SIZE
    spot_pix = spot_width/PIX_SIZE

    # make the box 1.3x the size of the spot, unless it will cause overlap
    side = int(1.3*spot_pix if 1.3*spot_pix < v_pix-1 and 1.3*spot_pix < h_pix-1 else spot_pix)

    # create templates
    x_range = int(v_pix*(n_rows-1)) + side
    y_range = int(h_pix*(n_cols-1)) + side
    target = np.zeros((x_range, y_range))

    # center position of the origin
    origin = (side/2, side/2)
    print(origin)
    for row in range(n_rows):
        for col in range(n_cols):
            center_x = origin[0] + row * v_pix
            center_y = origin[1] + col * h_pix

            # check that the blank fits within the bounds of the target array
            x_min = int(center_x - side / 2) if int(center_x - side / 2) > 0 else 0
            x_max = int(center_x + side / 2) if int(center_x + side / 2) < target.shape[0] else target.shape[0]
            y_min = int(center_y - side / 2) if int(center_y - side / 2) > 0 else 0
            y_max = int(center_y + side / 2) if int(center_y + side / 2) < target.shape[1] else target.shape[1]

            blank = np.ones((x_max-x_min, y_max-y_min))

            target[x_min:x_max, y_min:y_max] = blank

    return target, origin


def build_and_place_block_array(props_array_, spot_mask_, params_, return_type='region'):
    """
    Uses the fiducial centroid positions to build a "block array":
        "block array" is composed of (side, side) regions centered on each expected well position
        There are (rows, cols) of
        np.array of shape = (rows, cols)
        whose elements are np.ones(shape=(side, side))

    :param props_array_:
    :param spot_mask_:
    :param params_:
    :param return_type:
    :return:
    """

    rows = params_['rows']
    cols = params_['columns']

    # fiducials are averaged to find x-y bounds.
    #   one or both can be None, if one is None, this is handled
    #   if two are None, we have to try something else
    fiduc_1 = props_array_[0][0].centroid if props_array_[0][0] else (0, 0)
    fiduc_2 = props_array_[0][cols-1].centroid if props_array_[0][cols-1] else (0, 0)
    fiduc_3 = props_array_[rows-1][0].centroid if props_array_[rows-1][0] else (0, 0)
    fiduc_4 = props_array_[rows-1][cols-1].centroid if props_array_[rows-1][cols-1] else (0, 0)

    # average if two else use one
    x_list_min = [fiduc_1[0], fiduc_2[0]]
    x_min = np.sum(x_list_min) / np.sum(len([v for v in x_list_min if v != 0]))

    y_list_min = [fiduc_1[1], fiduc_3[1]]
    y_min = np.sum(y_list_min) / np.sum(len([v for v in y_list_min if v != 0]))

    x_list_max = [fiduc_3[0], fiduc_4[0]]
    x_max = np.sum(x_list_max) / np.sum(len([v for v in x_list_max if v != 0]))

    y_list_max = [fiduc_2[1], fiduc_4[1]]
    y_max = np.sum(y_list_max) / np.sum(len([v for v in y_list_max if v != 0]))

    # check for NaNs - no fiducial was found
    #   instead, we will use ANY spot at the boundaries to motivate the positioning
    if math.isnan(x_min):
        x_mins = [p.centroid[0] for p in props_array_[0, :] if p]
        x_min = np.average(x_mins)
    if math.isnan(y_min):
        y_mins = [p.centroid[1] for p in props_array_[:, 0] if p]
        y_min = np.average(y_mins)
    if math.isnan(x_max):
        x_maxs = [p.centroid[0] for p in props_array_[rows-1, :] if p]
        x_max = np.average(x_maxs)
    if math.isnan(y_max):
        y_maxs = [p.centroid[1] for p in props_array_[:, cols-1] if p]
        y_max = np.average(y_maxs)

    # build_block_array uses values in the params_ to motivate array dimensions, spacings
    # build block array
    template, temp_origin = build_block_array(params_)

    # center the template origin on the expected fiducial 1
    print(x_min, y_min)
    target = np.zeros(spot_mask_.shape)
    target[int(x_min-temp_origin[0]):int(x_min+template.shape[0]-temp_origin[0]),
           int(y_min-temp_origin[1]):int(y_min+template.shape[1]-temp_origin[1])] = template

    if return_type == 'product':
        return target*spot_mask_
    elif return_type == 'region':
        return target


def compute_od(props_array,bgprops_array):
    """

    Parameters
    ----------
    props_array: object:
     2D array of regionprops objects at the spots over data.
    bgprops_array: object:
     2D array of regionprops objects at the spots over background.

    Returns
    -------
    od_norm
    i_spot
    i_bg
    """
    assert props_array.shape == bgprops_array.shape, 'regionprops arrays representing sample and background are not the same.'
    n_rows=props_array.shape[0]
    n_cols=props_array.shape[1]
    i_spot=np.empty((n_rows,n_cols))
    i_bg=np.empty((n_rows,n_cols))
    od_norm=np.empty((n_rows,n_cols))

    i_spot[:]=np.NaN
    i_bg[:]=np.NaN
    od_norm[:]=np.NaN

    for r in np.arange(n_rows):
        for c in np.arange(n_cols):
            if props_array[r,c] is not None:
                i_spot[r,c]=props_array[r,c].mean_intensity
                i_bg[r,c]=bgprops_array[r,c].mean_intensity
    od_norm=i_bg/i_spot
    # Optical density is affected by Beer-Lambert law, i.e. I = I0*e^-{c*thickness). I0/I = e^{c*thickness).

    return od_norm, i_spot, i_bg


def create_reference_grid(mean_point,
                          nbr_grid_rows=6,
                          nbr_grid_cols=6,
                          spot_dist=83):
    """
    Generate initial spot grid based on image scale and number of spots.
    :param tuple start_point: (x,y) coordinates of center of grid
    :param int nbr_grid_rows: Number of spot rows
    :param int nbr_grid_cols: Number of spot columns
    :param int spot_dist: Distance between spots
    :return np.array grid_coords: (x, y) coordinates for reference spots (nbr x 2)
    """
    start_x = mean_point[0] - spot_dist * (nbr_grid_cols - 1) / 2
    start_y = mean_point[1] - spot_dist * (nbr_grid_rows - 1) / 2
    x_vals = np.linspace(start_x, start_x + (nbr_grid_cols - 1) * spot_dist, nbr_grid_cols)
    y_vals = np.linspace(start_y, start_y + (nbr_grid_rows - 1) * spot_dist, nbr_grid_rows)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_coords = np.vstack([grid_x.T, grid_y.T]).T

    return grid_coords


def icp(source, target, max_iterate=50, matrix_diff=1.):
    """
    Iterative closest point. Expects x, y coordinates of source and target in
    an array with shape: nbr of points x 2

    :param np.array source: Source spot coordinates
    :param np.array target: Target spot coordinates
    :param int max_iterate: Maximum number of registration iterations
    :param float matrix_diff: Sum of absolute differences between transformation
        matrices after one iteration
    :return np.array t_matrix: 2D transformation matrix
    """
    src = source.copy().astype(np.float32)
    dst = target.copy().astype(np.float32)

    src = np.expand_dims(src, 0)
    dst = np.expand_dims(dst, 0)

    # Initialize kNN module
    knn = cv.ml.KNearest_create()
    labels = np.array(range(dst.shape[1])).astype(np.float32)
    knn.train(dst[0], cv.ml.ROW_SAMPLE, labels)
    # Initialize transformation matrix
    t_matrix = np.eye(3)
    t_temp = np.eye(3)
    t_old = t_matrix

    # Iterate while matrix difference > threshold
    for i in range(max_iterate):

        # Find closest points
        ret, results, neighbors, dist = knn.findNearest(src[0], 1)
        # Outlier removal
        idxs = np.squeeze(neighbors.astype(np.uint8))
        dist_max = 2 * np.median(dist)
        normal_idxs = np.where(dist < dist_max)[0]
        idxs = idxs[normal_idxs]
        # Find rigid transform
        t_iter = cv.estimateRigidTransform(
            src[0, normal_idxs, :],
            dst[0, idxs, :],
            fullAffine=False,
        )
        if t_iter is None:
            print("Optimization failed. Using initial estimate")
            return np.eye(3)[:2]
        t_temp[:2] = t_iter
        src = cv.transform(src, t_iter)
        t_matrix = np.dot(t_temp, t_matrix)
        # Estimate diff
        t_diff = sum(sum(abs(t_matrix[:2] - t_old[:2])))
        t_old = t_matrix
        if t_diff < matrix_diff:
            break

    return t_matrix[:2]


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(x_vars,
                              y_vars,
                              scale_vars=(1, .1),
                              angle_vars=(0, .5),
                              nbr_particles=100):
    """
    Create particles from parameters x, y, scale and angle given mean and std.
    :param x_vars:
    :param y_vars:
    :param scale_vars:
    :param angle_vars:
    :param nbr_particles:
    :return:
    """
    # x, y, scale, angle
    particles = np.empty((nbr_particles, 4))
    particles[:, 0] = x_vars[0] + (np.random.randn(nbr_particles) * y_vars[1])
    particles[:, 1] = y_vars[0] + (np.random.randn(nbr_particles) * y_vars[1])
    particles[:, 2] = angle_vars[0] + (np.random.randn(nbr_particles) * angle_vars[1])
    particles[:, 3] = scale_vars[0] + (np.random.randn(nbr_particles) * scale_vars[1])
    return particles


def particle_filter(fiducial_coords,
                    spot_coords,
                    particles,
                    stds,
                    max_iter=50,
                    stop_criteria=.1):
    """
    Particle filtering to determine best grid location
    :param fiducial_coords:
    :param spot_coords:
    :param particles:
    :param stds:
    :param max_iter:
    :param stop_criteria:
    :return:
    """

    # Pretrain spot coords
    dst = spot_coords.copy().astype(np.float32)
    knn = cv.ml.KNearest_create()
    labels = np.array(range(dst.shape[0])).astype(np.float32)
    knn.train(dst, cv.ml.ROW_SAMPLE, labels)

    nbr_particles = particles.shape[0]
    weights = np.zeros(nbr_particles)

    # Iterate until min dist doesn't change
    min_dist_old = 10 ** 6
    for i in range(max_iter):
        # Reduce standard deviations a little every iteration
        stds = stds * 0.95 ** i

        # im_roi = image.copy()
        # im_roi = cv.cvtColor(im_roi, cv.COLOR_GRAY2RGB)

        for p in range(nbr_particles):
            particle = particles[p]
            # Generate transformation matrix
            t_matrix = cv.getRotationMatrix2D(
                (particle[0], particle[1]),
                particle[2],
                particle[3],
            )
            trans_coords = cv.transform(np.array([fiducial_coords]), t_matrix)
            trans_coords = trans_coords[0].astype(np.float32)

            # for c in range(trans_coords.shape[0]):
            #     coord = tuple(trans_coords[c, :].astype(np.int))
            #     cv.circle(im_roi, coord, 2, (0, 0, 255), 10)

            # Find nearest spots
            ret, results, neighbors, dist = knn.findNearest(trans_coords, 1)
            weights[p] = np.linalg.norm(dist)

        # plt.imshow(im_roi)
        # plt.axis('off')
        # plt.show()

        min_dist = np.min(weights)
        print(min_dist)
        # Low distance should correspond to high probability
        weights = 1 / weights
        # Make weights sum to 1
        weights = weights / sum(weights)

        # Importance sampling
        idxs = np.random.choice(nbr_particles, nbr_particles, p=weights)
        particles = particles[idxs, :]

        # Distort particles
        for c in range(4):
            distort = np.random.randn(nbr_particles)
            particles[:, c] = particles[:, c] + distort * stds[c]

        # See if min dist is not decreasing anymore
        if abs(min_dist_old - min_dist) < stop_criteria:
            break
        min_dist_old = min_dist

    # Return best particle
    particle = particles[weights == weights.max(), :][0]
    # Generate transformation matrix
    t_matrix = cv.getRotationMatrix2D(
        (particle[0], particle[1]),
        particle[2],
        particle[3],
    )
    return t_matrix