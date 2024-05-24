import cv2 as cv
import numpy as np

from skimage.measure import label
from skimage import util as u
from skimage.morphology import disk, ball, binary_opening, binary_erosion
from skimage.filters import threshold_otsu, threshold_minimum
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border


def get_unimodal_threshold(input_image):
    """Determines optimal unimodal threshold

    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding

    :param np.array input_image: generate mask for this image
    :return float best_threshold: optimal lower threshold for the foreground
     hist
    """

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, 'Error in unimodal thresholding'
    return best_threshold


def create_unimodal_mask(input_image, str_elem_size=3):
    """Create a mask with unimodal thresholding and morphological operations

    unimodal thresholding seems to oversegment, erode it by a fraction

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    thr_image = binary_opening(input_image > thr, str_elem)
    mask = binary_erosion(thr_image, str_elem)
    return mask


def create_otsu_mask(input_image, scale=1):
    """Create a binary mask using morphological operations

    Opening removes small objects in the foreground.

    :param np.array input_image: generate masks from this image
    :param float scale: Scale the threshold
    :return: mask of input_image, np.array
    """
    if np.min(input_image) == np.max(input_image):
        return np.ones(input_image.shape)
    else:
        thr = threshold_otsu(input_image, nbins=512)
    return input_image > (scale * thr)

def crop_image_from_coords(im, coords, margin=500):
    """
    Given image coordinates, crop image around them with a margin.

    :param np.array im: 2D image
    :param np.array coords: Grid coordinates which should be contained
        in the cropped image (nbr points x 2)
    :param int margin: How much margin around the coordinates
    :return np.array im_roi: Cropped image
    :return np.array crop_coords: Grid coordinates with new origin (rows, cols)
    """
    im_shape = im.shape
    row_min = int(max(0, np.min(coords[:, 0]) - margin))
    row_max = int(min(im_shape[0], np.max(coords[:, 0]) + margin))
    col_min = int(max(0, np.min(coords[:, 1]) - margin))
    col_max = int(min(im_shape[1], np.max(coords[:, 1]) + margin))
    im_crop = im[row_min:row_max, col_min:col_max]

    # Update coordinates with new origin
    crop_coords = coords.copy()
    crop_coords[:, 0] = crop_coords[:, 0] - row_min + 1
    crop_coords[:, 1] = crop_coords[:, 1] - col_min + 1
    return im_crop, crop_coords


def crop_image_at_center(im, center, height, width):
    """
    Crop the supplied image to include only the well and its spots

    :param im: image
    :param float center: Center (row, col) of the crop box
    :param float height: height of the crop box
    :param float width: width of the crop box
    :return np.array crop: Cropped image
    :return list bbox: Bounding box coordinates [row min, col min, row max, col max]
    """
    c_row, c_col = center
    im_h, im_w = im.shape
    # truncate the bounding box when it exceeds image size
    bbox = np.rint([max(c_row - height / 2, 0),
                   max(c_col - width / 2, 0),
                   min(c_row + height / 2, im_h),
                   min(c_col + width / 2, im_w)]).astype(np.int32)
    crop = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return crop, bbox


def crop_image(arr, cx_, cy_, radius_, border_=500):
    """
    crop the supplied image to include only the well and its spots

    :param arr: image
    :param float cx_: Center x coordinate
    :param float cy_: Center y coordinate
    :param float radius_: Crop radius
    :param int border_: Margin on each side in pixels
    :return np.array crop: Cropped image
    """
    cx_ = int(np.rint(cx_))
    cy_ = int(np.rint(cy_))
    crop = arr[
           cy_ - (radius_ - border_): cy_ + (radius_ - border_),
           cx_ - (radius_ - border_): cx_ + (radius_ - border_)
           ]

    return crop


def get_largest_component(spot_segm):
    """
    Remove everything but the largest connected component from segmented image.

    :param np.array spot_segm: Binary segmented 2D image
    :return np.array largest_component: Largest connected component in image
    """
    labels = label(spot_segm)
    largest_component = labels.copy()
    if labels.max() > 0:
        largest_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_component.astype(labels.dtype)


def thresh_and_binarize(image,
                        method='rosin',
                        invert=True,
                        disk_size=10,
                        thr_percent=95,
                        get_lcc=False):
    """
    receives greyscale np.ndarray image
        inverts the intensities
        thresholds on the minimum peak
        converts the image into binary about that threshold

    :param np.ndarray image: 2D grayscale image
    :param str method: Trheshold type: 'bimodal', 'otsu', 'rosin' or 'bright_spots'
    :param bool invert: Invert image if spots are dark
    :param int disk_size: Structuring element disk size
    :param int thr_percent: Thresholding percentile
    :param bool get_lcc: Returns only the largest connected component
    :return: spots threshold_min on this image
    """
    image_ = image.copy()
    if invert:
        image_ = u.invert(image_)

    if method == 'bimodal':
        thresh = threshold_minimum(image_, nbins=512)
        spots = (image_ > thresh).astype(np.uint8)

    elif method == 'otsu':
        spots = create_otsu_mask(image_, scale=1)

    elif method == 'rosin':
        spots = create_unimodal_mask(image_, str_elem_size=3)

    elif method == 'bright_spots':
        spots = image_ > np.percentile(image_, thr_percent)
        str_elem = disk(disk_size)
        spots = binary_opening(spots, str_elem)
        spots = binary_fill_holes(spots, str_elem)
        spots = clear_border(spots)

    else:
        raise ModuleNotFoundError("not a supported method for thresh_and_binarize")

    if get_lcc:
        spots = get_largest_component(spots)

    return spots


class SpotDetector:
    """
    Detects spots in well image using a Laplacian of Gaussian filter
    followed by blob detection
    """

    def __init__(self,
                 imaging_params,
                 min_thresh=100,
                 max_thresh=255,
                 min_circularity=.1,
                 min_convexity=.5,
                 min_dist_between_blobs=10,
                 min_repeatability=2):
        """
        :param int min_thresh: Minimum threshold
        :param int max_thresh: Maximum threshold
        :param float min_circularity: Minimum circularity of spots
        :param float min_convexity: Minimum convexity of spots
        :param float min_dist_between_blobs: minimal distance in pixels between two
            spots for them to be called as different spots
        :param int min_repeatability: minimal number of times the same spot has to be
            detected at different thresholds
        """

        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.min_dist_between_blobs = min_dist_between_blobs
        self.min_repeatability = min_repeatability
        self.min_circularity = min_circularity
        self.min_convexity = min_convexity
        self.sigma_gauss = int(np.round(imaging_params['spot_width'] /
                                imaging_params['pixel_size'] / 4))
        self.min_area = 4 * self.sigma_gauss ** 2
        self.max_area = 50 * self.min_area
        self.nbr_expected_spots = imaging_params['rows'] * imaging_params['columns']

        self.blob_detector = self._make_blob_detector()
        self.log_filter = self._make_log_filter()

    def _make_blob_detector(self):
        # Set spot detection parameters
        blob_params = cv.SimpleBlobDetector_Params()
        # Change thresholds
        blob_params.minThreshold = self.min_thresh
        blob_params.maxThreshold = self.max_thresh
        # Filter by Area
        blob_params.filterByArea = True
        blob_params.minArea = self.min_area
        blob_params.maxArea = self.max_area
        # Filter by Circularity
        blob_params.filterByCircularity = True
        blob_params.minCircularity = self.min_circularity
        # Filter by Convexity
        blob_params.filterByConvexity = True
        blob_params.minConvexity = self.min_convexity
        blob_params.minDistBetweenBlobs = self.min_dist_between_blobs
        blob_params.minRepeatability = self.min_repeatability
        # This detects bright spots, which they are after top hat
        blob_params.blobColor = 255
        detector = cv.SimpleBlobDetector_create(blob_params)
        return detector

    def _make_log_filter(self):
        """
        Creates a uniform 2D Laplacian of Gaussian filter with given sigma.

        :param int sigma: Standard deviation of Gaussian
        :return np.array log_filter: 2D LoG filter
        """
        n = np.ceil(self.sigma_gauss * 6)
        rows, cols = np.ogrid[-n // 2:n // 2 + 1, -n // 2:n // 2 + 1]
        sigma_sq = 2 * self.sigma_gauss ** 2
        row_filter = np.exp(-(rows ** 2 / sigma_sq))
        col_filter = np.exp(-(cols ** 2 / sigma_sq))
        log_filter = (-sigma_sq + cols ** 2 + rows ** 2) * \
                     (col_filter * row_filter) * \
                     (1 / (np.pi * sigma_sq * self.sigma_gauss ** 2))
        # Total filter should sum to 1 to not alter mean intensity
        log_filter = log_filter / sum(sum(log_filter))
        return log_filter

    def get_spot_coords(self,
                        im,
                        margin=0,
                        im_mean=100,
                        im_std=25,
                        max_intensity=255,
                        ):
        """
        Use OpenCVs simple blob detector (thresholdings and grouping by properties)
        to detect all dark spots in the image. First filter with a Laplacian of
        Gaussian with sigma matching spots to enhance spots in image.

        :param np.array im: uint8 mage containing spots
        :param int margin: Pixel margin around image edged where spots should be
            ignored (to ignore boundary effects)
        :param float im_mean: Set normalized image to fixed mean
        :param float im_std: Set normalized image to fixed std
        :param int max_intensity: Maximum image intensity (default uint8)
        :return np.array spot_coords: row, col coordinates of spot centroids
            (nbr spots x 2)
        """
        # First invert image to detect peaks
        im_norm = (max_intensity - im) / max_intensity
        # Filter with Laplacian of Gaussian
        im_norm = cv.filter2D(im_norm, -1, self.log_filter)
        # Normalize
        im_norm = im_norm / im_norm.std() * im_std
        im_norm = im_norm - im_norm.mean() + im_mean
        im_norm[im_norm < 0] = 0
        im_norm[im_norm > 255] = 255
        im_norm = im_norm.astype(np.uint8)

        # Detect peaks in filtered image
        keypoints = self.blob_detector.detect(im_norm)

        spot_coords = np.zeros((len(keypoints), 2))
        # Remove outliers and convert to np.array
        row_max, col_max = im.shape
        idx = 0
        for keypoint in range(len(keypoints)):
            pt = keypoints[keypoint].pt
            if margin < pt[0] < row_max - margin and margin < pt[1] < col_max - margin:
                spot_coords[idx, 0] = pt[1]
                spot_coords[idx, 1] = pt[0]
                idx += 1
        spot_coords = spot_coords[:idx, :]
        return spot_coords
