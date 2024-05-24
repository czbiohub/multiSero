import logging
import os
import pandas as pd
import time

import multiSero.array_analyzer.extract.background_estimator as background_estimator
import multiSero.array_analyzer.extract.img_processing as img_processing
import multiSero.array_analyzer.extract.metadata as metadata
import multiSero.array_analyzer.extract.txt_parser as txt_parser
import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.load.debug_plots as debug_plots
import multiSero.array_analyzer.load.report as report
import multiSero.array_analyzer.transform.point_registration as registration
import multiSero.array_analyzer.transform.array_generation as array_gen
import multiSero.array_analyzer.utils.io_utils as io_utils


def point_registration(input_dir, output_dir):
    """
    For each image in input directory, detect spots using particle filtering
    to register fiducial spots to blobs detected in the image.

    :param str input_dir: Input directory containing images and an xml file
        with parameters
    :param str output_dir: Directory where output is written to
    """
    logger = logging.getLogger(constants.LOG_NAME)

    metadata.MetaData(input_dir, output_dir)
    nbr_outliers = constants.params['nbr_outliers']

    # Create reports instance for whole plate
    reporter = report.ReportWriter()
    # Create writer for stats per well
    well_xlsx_path = os.path.join(
        constants.RUN_PATH,
        'stats_per_well.xlsx',
    )
    well_xlsx_writer = pd.ExcelWriter(well_xlsx_path)
    antigen_df = reporter.get_antigen_df()
    antigen_df.to_excel(well_xlsx_writer, sheet_name='antigens')

    # Initialize background estimator
    bg_estimator = background_estimator.BackgroundEstimator2D(
        block_size=128,
        order=2,
        normalize=False,
    )

    # Get grid rows and columns from params
    nbr_grid_rows = constants.params['rows']
    nbr_grid_cols = constants.params['columns']
    fiducials_idx = constants.FIDUCIALS_IDX
    # Create spot detector instance
    spot_detector = img_processing.SpotDetector(
        imaging_params=constants.params,
    )

    well_images = io_utils.get_image_paths(input_dir)
    well_names = list(well_images)
    # If rerunning only a subset of wells
    if constants.RERUN:
        logger.info("Rerunning wells: {}".format(constants.RERUN_WELLS))
        txt_parser.rerun_xl_od(
            well_names=well_names,
            well_xlsx_path=well_xlsx_path,
            rerun_names=constants.RERUN_WELLS,
            xlsx_writer=well_xlsx_writer,
        )
        reporter.load_existing_reports()
        well_names = constants.RERUN_WELLS
        # remove debug images from old runs
        for f in os.listdir(constants.RUN_PATH):
            if f.split('_')[0] in well_names:
                os.remove(os.path.join(constants.RUN_PATH, f))
    else:
        reporter.create_new_reports()

    # ================
    # loop over well images
    # ================
    for well_name in well_names:
        start_time = time.time()
        im_path = well_images[well_name]
        image = io_utils.read_gray_im(im_path)
        logger.info("Extracting well: {}".format(well_name))
        # Get max intensity
        max_intensity = io_utils.get_max_intensity(image)
        logger.debug("Image max intensity: {}".format(max_intensity))
        # Crop image to well only
        """""
        try:
            well_center, well_radi, _ = image_parser.find_well_border(
                image,
                detmethod='region',
                segmethod='otsu',
            )
            im_well, _ = img_processing.crop_image_at_center(
                im=image,
                center=well_center,
                height=2 * well_radi,
                width=2 * well_radi,
            )
        except IndexError:
            logging.warning("Couldn't find well in {}".format(well_name))
            im_well = image
        """""
        im_well = image
        # Find spot center coordinates
        spot_coords = spot_detector.get_spot_coords(
            im=im_well,
            max_intensity=max_intensity,
        )
        if spot_coords.shape[0] < constants.MIN_NBR_SPOTS:
            logging.warning("Not enough spots detected in {},"
                            "continuing.".format(well_name))
            continue
        # Create particle filter registration instance
        register_inst = registration.ParticleFilter(
            spot_coords=spot_coords,
            im_shape=im_well.shape,
            fiducials_idx=fiducials_idx,
        )
        register_inst.particle_filter()
        if not register_inst.registration_ok:
            logger.warning("Registration failed for {}, "
                           "repeat with outlier removal".format(well_name))
            register_inst.particle_filter(nbr_outliers=nbr_outliers)
        # Transform grid coordinates
        registered_coords = register_inst.compute_registered_coords()
        # Check that registered coordinates are inside well
        registration_ok = register_inst.check_reg_coords()
        if not registration_ok:
            logger.warning("Final registration failed,"
                           "will not write OD for {}".format(well_name))
            if constants.DEBUG:
                debug_plots.plot_registration(
                    im_well,
                    spot_coords,
                    register_inst.fiducial_coords,
                    registered_coords,
                    os.path.join(constants.RUN_PATH, well_name + '_failed'),
                    max_intensity=max_intensity,
                )
            continue

        # Crop image
        im_crop, crop_coords = img_processing.crop_image_from_coords(
            im=im_well,
            coords=registered_coords,
        )
        im_crop = im_crop / max_intensity
        # Estimate background
        background = bg_estimator.get_background(im_crop)
        # Find spots near grid locations and compute properties
        spots_df, spot_props = array_gen.get_spot_intensity(
            coords=crop_coords,
            im=im_crop,
            background=background,
        )
        # Write metrics for each spot in grid in current well
        spots_df.to_excel(well_xlsx_writer, sheet_name=well_name)
        # Assign well OD, intensity, and background stats to plate
        reporter.assign_well_to_plate(well_name, spots_df)

        time_msg = "Time to extract OD in {}: {:.3f} s".format(
            well_name,
            time.time() - start_time,
        )
        print(time_msg)
        logger.info(time_msg)

        # ==================================
        # SAVE FOR DEBUGGING
        if constants.DEBUG:
            start_time = time.time()
            # Save spot and background intensities
            output_name = os.path.join(constants.RUN_PATH, well_name)
            # Save OD plots, composite spots and registration
            debug_plots.plot_od(
                spots_df=spots_df,
                nbr_grid_rows=nbr_grid_rows,
                nbr_grid_cols=nbr_grid_cols,
                output_name=output_name,
            )
            debug_plots.save_composite_spots(
                spot_props=spot_props,
                output_name=output_name,
                image=im_crop,
            )
            debug_plots.plot_background_overlay(
                im_crop,
                background,
                output_name,
            )
            debug_plots.plot_registration(
                image=im_well,
                spot_coords=spot_coords,
                grid_coords=register_inst.fiducial_coords,
                reg_coords=registered_coords,
                output_name=output_name,
                max_intensity=max_intensity,
            )
            logger.debug("Time to save debug images: {:.3f} s".format(
                time.time() - start_time),
            )

    # After running all wells, write plate reports
    well_xlsx_writer.close()
    reporter.write_reports()
