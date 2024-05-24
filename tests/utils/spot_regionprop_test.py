import numpy as np
import pytest

import multiSero.array_analyzer.utils.spot_regionprop as regionprop


@pytest.fixture
def spot_and_mask():
    im_size = 51
    sigma = 10
    mu = 25
    row, col = np.meshgrid(
        np.linspace(0, im_size, im_size),
        np.linspace(0, im_size, im_size),
    )
    im_spot = np.exp(-((row - mu) / sigma) ** 2 / 2 - ((col - mu) / sigma) ** 2 / 2)
    # Invert for dark spot
    im_spot = 1. - im_spot
    mask_spot = (im_spot < .5).astype(np.uint8)
    return im_spot, mask_spot


def test_regionprop_init():
    prop = regionprop.SpotRegionprop(row_idx=2, col_idx=3, label=5)
    assert prop.label == 5
    assert prop.spot_dict['grid_row'] == 2
    assert prop.spot_dict['grid_col'] == 3
    assert prop.image is None
    assert prop.background is None
    assert prop.mask is None
    assert prop.masked_image is None


def test_make_mask():
    prop = regionprop.SpotRegionprop(row_idx=1, col_idx=2, label=0)
    mask = prop.make_mask(51)
    assert mask.shape == (51, 51)
    assert mask[0, 0] == 0
    assert mask[25, 25] == 1


def test_compute_stats(spot_and_mask):
    im_spot, mask_spot = spot_and_mask
    prop = regionprop.SpotRegionprop(row_idx=4, col_idx=5, label=1)
    prop.image = im_spot
    prop.mask = mask_spot
    prop.background = np.zeros_like(im_spot) + 0.5
    prop.compute_stats()
    assert prop.label == 1
    assert prop.spot_dict['intensity_median'] < 0.3
    assert prop.spot_dict['bg_median'] == 0.5
    assert prop.spot_dict['od_norm'] == np.log10(
            0.5 / prop.spot_dict['intensity_median'],
        )


def test_generate_props_from_disk(spot_and_mask):
    im_spot, _ = spot_and_mask
    bg_spot = np.zeros_like(im_spot) + 0.5
    prop = regionprop.SpotRegionprop(row_idx=0, col_idx=2, label=3)
    bbox = [5, 10, 35, 45]
    centroid = [25, 26]
    prop.generate_props_from_disk(im_spot, bg_spot, bbox, centroid)
    assert prop.label == 3
    # Stats for dataframe
    assert prop.spot_dict['centroid_row'] == centroid[0]
    assert prop.spot_dict['centroid_col'] == centroid[1]
    assert prop.spot_dict['bbox_row_min'] == bbox[0]
    assert prop.spot_dict['bbox_col_min'] == bbox[1]
    assert prop.spot_dict['bbox_row_max'] == bbox[2]
    assert prop.spot_dict['bbox_col_max'] == bbox[3]
    assert prop.spot_dict['intensity_mean'] > 0
    assert prop.spot_dict['intensity_median'] > 0.8
    assert prop.spot_dict['intensity_median'] < 0.81
    assert prop.spot_dict['bg_mean'] == 0.5
    assert prop.spot_dict['bg_median'] == 0.5
    assert prop.spot_dict['od_norm'] == np.log10(
            0.5 / prop.spot_dict['intensity_median'],
        )
    # ROIs
    assert prop.image.all() == im_spot.all()
    assert prop.background.all() == bg_spot.all()
    mask = prop.make_mask(im_spot.shape[0])
    assert prop.mask.all() == mask.all()
    assert prop.masked_image.all() == (im_spot * mask).all()


def test_generate_props_from_mask(spot_and_mask):
    im_spot, mask_spot = spot_and_mask
    prop = regionprop.SpotRegionprop(row_idx=2, col_idx=1, label=4)
    bbox = [5, 10, 40, 45]
    bg_spot = np.zeros_like(im_spot) + 1.
    prop.generate_props_from_mask(im_spot, bg_spot, mask_spot, bbox)
    assert prop.label == 4
    # Stats for dataframe
    assert 29 <= prop.spot_dict['centroid_row'] <= 30
    assert 34 <= prop.spot_dict['centroid_col'] <= 35
    assert prop.spot_dict['bbox_row_min'] == 18
    assert prop.spot_dict['bbox_col_min'] == 23
    assert prop.spot_dict['bbox_row_max'] == 42
    assert prop.spot_dict['bbox_col_max'] == 47
    assert prop.spot_dict['intensity_mean'] == np.mean(im_spot[mask_spot > 0])
    assert prop.spot_dict['intensity_median'] == np.median(im_spot[mask_spot > 0])
    assert prop.spot_dict['bg_mean'] == 1.
    assert prop.spot_dict['bg_median'] == 1.
    assert prop.spot_dict['od_norm'] == np.log10(
            1. / prop.spot_dict['intensity_median'],
        )
    # ROIs
    assert prop.image.all() == im_spot.all()
    assert prop.mask.all() == mask_spot.all()
    assert prop.masked_image.all() == (im_spot * mask_spot).all()
