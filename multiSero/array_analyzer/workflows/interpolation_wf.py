import time
import os
import numpy as np
import pandas as pd
import skimage.io as io

import multiSero.array_analyzer.extract.image_parser as image_parser
import multiSero.array_analyzer.extract.txt_parser as txt_parser
import multiSero.array_analyzer.extract.img_processing as img_processing
import multiSero.array_analyzer.load.debug_plots as debug_plots
import multiSero.array_analyzer.load.report as report
import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.transform.array_generation as array_gen
import multiSero.array_analyzer.extract.background_estimator as background_estimator
import multiSero.array_analyzer.utils.io_utils as io_utils
from multiSero.array_analyzer.extract.metadata import MetaData


def interp(input_dir, output_dir):

    MetaData(input_dir, output_dir)

    # Initialize background estimator
    bg_estimator = background_estimator.BackgroundEstimator2D(
        block_size=128,
        order=2,
        normalize=False,
    )
    reporter = report.ReportWriter()
    well_xlsx_path = os.path.join(
        constants.RUN_PATH,
        'stats_per_well.xlsx',
    )
    well_xlsx_writer = pd.ExcelWriter(well_xlsx_path)
    antigen_df = reporter.get_antigen_df()
    antigen_df.to_excel(well_xlsx_writer, sheet_name='antigens')

    # ================
    # loop over images => good place for multiproc?  careful with columns in report
    # ================
    well_images = io_utils.get_image_paths(input_dir)

    for well_name, im_path in well_images.items():
        start = time.time()
        image = io_utils.read_gray_im(im_path)

        spot_props_array = txt_parser.create_array(
            constants.params['rows'],
            constants.params['columns'],
            dtype=object,
        )
        bgprops_array = txt_parser.create_array(
            constants.params['rows'],
            constants.params['columns'],
            dtype=object,
        )

        # finding center of well and cropping
        well_center, well_radi, well_mask = image_parser.find_well_border(image, detmethod='region', segmethod='otsu')
        im_crop, _ = img_processing.crop_image_at_center(
            image,
            well_center,
            2 * well_radi,
            2 * well_radi
        )

        # find center of spots from crop
        spot_mask = img_processing.thresh_and_binarize(im_crop, method='bright_spots')
        spot_props = image_parser.generate_props(spot_mask, intensity_image=im_crop)

        # if debug:

        crop_coords = image_parser.grid_from_centroids(
            spot_props,
            constants.params['rows'],
            constants.params['columns']
        )

        # convert to float64
        im_crop = im_crop / np.iinfo(im_crop.dtype).max
        background = bg_estimator.get_background(im_crop)
        spots_df, spot_props = array_gen.get_spot_intensity(
            coords=crop_coords,
            im=im_crop,
            background=background,
            params=constants.params,
        )
        # Write metrics for each spot in grid in current well
        spots_df.to_excel(well_xlsx_writer, sheet_name=well_name)
        # Assign well OD, intensity, and background stats to plate
        reporter.assign_well_to_plate(well_name, spots_df)

        stop = time.time()
        print(f"\ttime to process={stop-start}")

        # SAVE FOR DEBUGGING
        if constants.DEBUG:
            # Save spot and background intensities.
            output_name = os.path.join(constants.RUN_PATH, well_name)

            # # Save mask of the well, cropped grayscale image, cropped spot segmentation.
            io.imsave(output_name + "_well_mask.png",
                      (255 * well_mask).astype('uint8'))
            io.imsave(output_name + "_crop.png",
                      (255 * im_crop).astype('uint8'))
            io.imsave(output_name + "_crop_binary.png",
                      (255 * spot_mask).astype('uint8'))

            # Evaluate accuracy of background estimation with green (image), magenta (background) overlay.
            im_bg_overlay = np.stack([background, im_crop, background], axis=2)

            io.imsave(output_name + "_crop_bg_overlay.png",
                      (255 * im_bg_overlay).astype('uint8'))

            # This plot shows which spots have been assigned what index.
            debug_plots.plot_centroid_overlay(
                im_crop,
                constants.params,
                spot_props,
                output_name,
            )
            debug_plots.plot_od(
                spots_df=spots_df,
                nbr_grid_rows=constants.params['rows'],
                nbr_grid_cols=constants.params['columns'],
                output_name=output_name,
            )
            # save a composite of all spots, where spots are from source or from region prop
            debug_plots.save_composite_spots(
                spot_props,
                output_name,
                image=im_crop,
            )
            debug_plots.save_composite_spots(
                spot_props,
                output_name,
                image=im_crop,
                from_source=True,
            )
            stop2 = time.time()
            print(f"\ttime to save debug={stop2-stop}")

    # After running all wells, write plate reports
    well_xlsx_writer.close()
    reporter.write_reports()
