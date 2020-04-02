# bchhun, {2020-03-23}

"""
Here we will call the methods from the ETL folders

============================
txt_parser workflow:
--------------------
1) xml_to_dict the xml file
2) create ID-array
3) create antigen-array

image_parser workflow:
----------------------
4) read_to_grey(supplied images)

# find center of well
5) thresh and binarize from 4
6) find well border from 5
7) crop image from 4

# find center of spots from crop
8) thresh and binarize from 4
9) clean spot binary from 5 (if using bimodal in 8)
10) generate props from 6
11) generate props dict from 7
12) assign props dict to array from 8

xlsx report generation workflow:
--------------------------------
13) "create base template"
14) "populate main tab" using :
        - workbook from 13
        - "ID-array" from 2
        - "props-array" from 12
        - "well" from "read_to_grey" from 4
15) "populate main replictes" :
        - workbook from 13
        - "props-array" from 12
        - "antigen-array" from 3
        - "well" from "read_to_grey" from 4
16) (populate well-tab) (in progress)
17) (populate well-replicate-tab) (in progress)

18) *repeat 14-17* using next image and well name
19) save .xlsx
==============================

==============================
FULL WORKFLOW

cli
---
- input folder
- output folder

extract
-------
A) search folder for all images, all .xmls to list
B) xlsx_report.create_base_template() step 13

C) txt_parse workflow above to create ID-array, antigen-array
D) image_parser workflow above to loop 4 (read_to_grey)
    within loop:
    E) image_parser steps 5-12

    transform
    ---------
    F) (ANY "transform" methods that will further analyze the data from E)
        (this is set aside as a place to make diagnosis calls and for downstream calculations using spot properties)

    load
    ----
    G) xlsx_report generation workflow steps 14-17

"""
import getopt
import glob
import os
import sys

import array_analyzer.extract.image_parser as image_parser
import array_analyzer.extract.txt_parser as txt_parser
from array_analyzer.load.xlsx_report import *
import array_analyzer.extract.img_processing as img_processing
from array_analyzer.load.debug_images import *
from array_analyzer.transform.property_filters import *
from array_analyzer.workflows import icp_wf, segmentation_wf

import time
from datetime import datetime
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd


def main(argv):
    inputfolder = ''
    outputfolder = ''
    debug = False
    method = 'fit'
    try:
        options, remainder = getopt.getopt(argv, "hi:o:dm:",
                                           ["help","ifile=", "ofile=", "debug=", "method="])
    except getopt.GetoptError:
        print('run_array_analyzer.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)

    for opt, arg in options:
        if opt == '-h':
            print('run_array_analyzer.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfolder = arg
        elif opt in ("-o", "--ofile"):
            outputfolder = arg
        elif opt in ("-m", "--method"):
            method = arg
            assert method in ['fit', 'interp'], \
                ValueError('"method" has to be "fit" or "interp"')
        elif opt in ("-d", "--debug"):
            print('debug mode on, saving well and spot images')
            debug = True

    if not os.path.isdir(inputfolder):
        raise ValueError("input folder is not a folder or not supplied")

    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    if method == 'fit':
        icp_wf.icp(inputfolder, outputfolder, debug)
    elif method == 'interp':
        segmentation_wf.seg(inputfolder, outputfolder, debug)
    else:
        raise KeyError(f"method {method} is not implemented")

    # workflow(inputfolder, outputfolder, method=method, debug=debug)


# def workflow(input_folder_, output_folder_, method='fit', debug=False):
#
#     xml_path = glob.glob(input_folder_ + '*.xml')
#     if len(xml_path) > 1 or not xml_path:
#         raise IOError("Did not find unique xml")
#     xml_path = xml_path[0]
#
#     # parsing .xml
#     fiduc, spots, repl, params = txt_parser.create_xml_dict(xml_path)
#
#     # creating our arrays
#     spot_ids = txt_parser.create_array(params['rows'], params['columns'])
#     antigen_array = txt_parser.create_array(params['rows'], params['columns'])
#
#     # adding .xml info to these arrays
#     spot_ids = txt_parser.populate_array_id(spot_ids, spots)
#     # spot_ids = populate_array_fiduc(spot_ids, fiduc)
#
#     antigen_array = txt_parser.populate_array_antigen(antigen_array, spot_ids, repl)
#
#     # save a sub path for this processing run
#     run_path = os.path.join(
#         output_folder_,
#         '_'.join([str(datetime.now().month),
#                   str(datetime.now().day),
#                   str(datetime.now().hour),
#                   str(datetime.now().minute),
#                   str(datetime.now().second)]),
#     )
#
#     # Write an excel file that can be read into jupyter notebook with minimal parsing.
#     xlwriterOD = pd.ExcelWriter(os.path.join(run_path, 'ODs.xlsx'))
#     pdantigen = pd.DataFrame(antigen_array)
#     pdantigen.to_excel(xlwriterOD, sheet_name='antigens')
#
#     if not os.path.isdir(run_path):
#         os.mkdir(run_path)
#
#     # ================
#     # loop over images => good place for multiproc?  careful with columns in report
#     # ================
#     images = [file for file in os.listdir(input_folder_) if '.png' in file or '.tif' in file or '.jpg' in file]
#
#     # remove any images that are not images of wells.
#     wellimages = [file for file in images if re.match(r'[A-P][0-9]{1,2}', file)]
#
#     # sort by letter, then by number (with '10' coming AFTER '9')
#     wellimages.sort(key=lambda x: (x[0], int(x[1:-4])))
#     #TODO: select wells based to analyze based on user input (Bryant)
#
#     # for image_name in wellimages:
#     #     start_time = time.time()
#     #     image = image_parser.read_gray_im(os.path.join(input_folder_, image_name))
#     # wellimages = ['H10.png','H11.png','H12.png']
#     # wellimages = ['B8.png', 'B9.png', 'B10.png']
#     # wellimages = ['A12.png', 'A11.png', 'A8.png', 'A1.png']
#     # wellimages = ['A9.png']
#     # wellimages = ['E5.png']
#     if debug:
#         well_path = os.path.join(run_path)
#         os.makedirs(well_path, exist_ok=True)
#
#     for well in wellimages:
#         start = time.time()
#         image, image_name = image_parser.read_gray_im(input_folder_)
#
#         props_array = txt_parser.create_array(
#             params['rows'],
#             params['columns'],
#             dtype=object,
#         )
#         bgprops_array = txt_parser.create_array(
#             params['rows'],
#             params['columns'],
#             dtype=object,
#         )
#         # finding center of well and cropping
#         cx, cy, r, well_mask = image_parser.find_well_border(
#             image,
#             segmethod='otsu',
#             detmethod='region',
#         )
#         im_crop = image_parser.crop_image(image, cx, cy, r, border_=0)
#
#         # Remove background
#         background = img_processing.get_background(im_crop, fit_order=2)
#         im_crop = (im_crop / background * np.mean(background)).astype(np.uint8)
#         spot_props = generate_props(spot_mask, intensity_image_=im_crop)
#
#         output_name = os.path.join(well_path, image_name[:-4])
#         spot_props_array = txt_parser.create_array(params['rows'], params['columns'], dtype=object)
#         bgprops_array = txt_parser.create_array(params['rows'], params['columns'], dtype=object)
#
#         # finding center of well and cropping
#         # cx, cy, r, well_mask = find_well_border(image, detmethod='region', segmethod='otsu')
#         im_crop = crop_image(image, cx, cy, r, border_=0)
#
#
#         # find center of spots from crop
#         spot_mask = thresh_and_binarize(im_crop, method='bright_spots')
#         if debug:
#             io.imsave(output_name + "_well_mask.png",
#                       (255 * well_mask).astype('uint8'))
#             io.imsave(output_name + "_crop.png",
#                       (255 * im_crop).astype('uint8'))
#             io.imsave(output_name + "_crop_binary.png",
#                   (255 * spot_mask).astype('uint8'))
#         background = get_background(im_crop, fit_order=2)
#
#         if debug:
#             im_bg_overlay = np.stack([background, im_crop, background], axis=2)
#             io.imsave(output_name + "_crop_bg_overlay.png",
#                       (255 * im_bg_overlay).astype('uint8'))
#
#
#
#
#         if method == 'fit':
#             spot_props = select_props(spot_props, attribute="area", condition="greater_than", condition_value=300)
#             fiducial_locations = [(0, 0), (0, 1), (0, 5), (7, 0), (7, 5)]
#             pix_size = 0.0049 # in mm
#             props_by_loc = find_fiducials_markers(spot_props,
#                                                   fiducial_locations,
#                                                   params['rows'],
#                                                   params['columns'],
#                                                   params['v_pitch'],
#                                                   params['h_pitch'],
#                                                   im_crop.shape,
#                                                   pix_size)
#
#             spot_props_array = assign_props_to_array_2(spot_props_array, props_by_loc)
#
#             # use the spot_props_array to find fiducials, create a new spot_mask "placed" on the array
#             placed_spotmask = build_and_place_block_array(spot_props_array, spot_mask, params, return_type='region')
#
#             spot_props = generate_props(placed_spotmask, intensity_image_=im_crop)
#             bg_props = generate_props(placed_spotmask, intensity_image_=background)
#
#             spot_labels = [p.label for p in spot_props]
#             bg_props = select_props(bg_props, attribute="label", condition="is_in", condition_value=spot_labels)
#
#             props_placed_by_loc = generate_props_dict(spot_props,
#                                                       params['rows'],
#                                                       params['columns'],
#                                                       min_area=100)
#             bgprops_by_loc = generate_props_dict(bg_props,
#                                                  params['rows'],
#                                                  params['columns'],
#                                                  min_area=100)
#         elif method == 'interp':
#             bg_props = generate_props(spot_mask, intensity_image_=background)
#             eccentricities = np.array([prop.eccentricity for prop in spot_props])
#             eccent_ub = eccentricities.mean() + 2 * eccentricities.std()
#             # spot_props = select_props(spot_props, attribute="area", condition="greater_than", condition_value=300)
#             spot_props = select_props(spot_props, attribute="eccentricity", condition="less_than",
#                                       condition_value=eccent_ub)
#             spot_labels = [p.label for p in spot_props]
#             bg_props = select_props(bg_props, attribute="label", condition="is_in", condition_value=spot_labels)
#
#             props_placed_by_loc = grid_from_centroids(spot_props,
#                                                im_crop,
#                                                params['rows'],
#                                                params['columns'],
#                                                )
#             # This call to generate_props_dict is excessive.
#             # Both spot_props and bgprops can be assigned locations in previous call.
#
#             bgprops_by_loc = grid_from_centroids(bg_props,
#                                                  background,
#                                                  params['rows'],
#                                                  params['columns'],
#                                                  )
#         props_array_placed = assign_props_to_array(spot_props_array, props_placed_by_loc)
#         bgprops_array = assign_props_to_array(bgprops_array, bgprops_by_loc)
#
#         nbr_grid_rows, nbr_grid_cols = props_array.shape
#         spot_coords = image_parser.get_spot_coords(
#             im_crop,
#             min_area=250,
#             min_thresh=25,
#         )
#         # todo: further calculations using bgprops, spot_props here
#         # TODO: compute spot and background intensities,
#         #  and then show them on a plate like graphic (visualize_elisa_spots).
#         od_well, i_well, bg_well = compute_od(props_array_placed, bgprops_array)
#
#         im_roi = im_crop.copy()
#         im_roi = cv.cvtColor(im_roi, cv.COLOR_GRAY2RGB)
#         for c in range(spot_coords.shape[0]):
#             coord = tuple(spot_coords[c, :].astype(np.int))
#             cv.circle(im_roi, coord, 2, (255, 0, 0), 10)
#         write_name = image_name[:-4] + '_spots.jpg'
#         # cv.imwrite(os.path.join(run_path, write_name), im_roi)
#         # plt.imshow(im_roi)
#         # plt.axis('off')
#         # plt.show()
#
#         mean_point, spot_dist = image_parser.grid_estimation(
#             im=im_crop,
#             spot_coords=spot_coords,
#         )
#         grid_coords = image_parser.create_reference_grid(
#             mean_point=mean_point,
#             nbr_grid_rows=nbr_grid_rows,
#             nbr_grid_cols=nbr_grid_cols,
#             spot_dist=spot_dist,
#         )
#
#         # SAVE FOR DEBUGGING
#         if debug:
#         for c in range(grid_coords.shape[0]):
#             coord = tuple(grid_coords[c, :].astype(np.int))
#             cv.circle(im_roi, coord, 2, (0, 0, 255), 10)
#         write_name = image_name[:-4] + '_grid.jpg'
#         # cv.imwrite(os.path.join(run_path, write_name), im_roi)
#         # plt.imshow(im_roi)
#         # plt.axis('off')
#         # plt.show()
#
#         # Optimize estimated coordinates with iterative closest point
#         t_matrix = image_parser.icp(
#             source=grid_coords,
#             target=spot_coords,
#         )
#         grid_coords = np.squeeze(cv.transform(np.expand_dims(grid_coords, 0), t_matrix))
#         print("Time to register grid to {}: {:.3f} s".format(image_name,
#                                                              time.time() - start_time))
#
#         for c in range(grid_coords.shape[0]):
#             coord = tuple(grid_coords[c, :].astype(np.int))
#             cv.circle(im_roi, coord, 2, (0, 255, 0), 10)
#         write_name = image_name[:-4] + '_icp.jpg'
#         cv.imwrite(os.path.join(run_path, write_name), im_roi)
#         # plt.imshow(im_roi)
#         # plt.axis('off')
#         # plt.show()
#
#             #   save spots
#             # save_all_wells(spot_props_array, spot_ids, well_path, image_name[:-4])
#
#             #   save a composite of all spots, where spots are from source or from region prop
#             # save_composite_spots(im_crop, props_array_placed, well_path, image_name[:-4], from_source=True)
#             # save_composite_spots(im_crop, props_array_placed, well_path, image_name[:-4], from_source=False)
#
#     #     # SAVE FOR DEBUGGING
#     #     if debug:
#     #         well_path = os.path.join(run_path)
#     #         os.makedirs(well_path, exist_ok=True)
#     #         output_name = os.path.join(well_path, image_name[:-4])
#     #         im_bg_overlay = np.stack([background, im_crop, background], axis=2)
#     #
#     #         #   save cropped image and the binary
#     #         io.imsave(output_name + "_crop.png",
#     #                   (255*im_crop).astype('uint8'))
#     #         io.imsave(output_name + "_crop_binary.png",
#     #                   (255 * spot_mask).astype('uint8'))
#     #         io.imsave(output_name + "_well_mask.png",
#     #                   (255 * well_mask).astype('uint8'))
#     #         io.imsave(output_name + "_crop_bg_overlay.png",
#     #                   (255 * im_bg_overlay).astype('uint8'))
#     #
#     #         # This plot shows which spots have been assigned what index.
#     #         plot_spot_assignment(od_well, i_well, bg_well,
#     #                              im_crop, props_placed_by_loc, bgprops_by_loc,
#     #                              image_name, output_name, params)
#     #
#     #         #   save spots
#     #         save_all_wells(props_array, spot_ids, well_path, image_name[:-4])
#     #
#     #         #   save a composite of all spots, where spots are from source or from region prop
#     #         save_composite_spots(im_crop, props_array_placed, well_path, image_name[:-4], from_source=True)
#     #         save_composite_spots(im_crop, props_array_placed, well_path, image_name[:-4], from_source=False)
#     #
#     #         stop2 = time.time()
#     #         print(f"\ttime to save debug={stop2-stop}")
#     #
#     # xlwriterOD.close()


if __name__ == "__main__":
    # input_path = '/Volumes/GoogleDrive/My Drive/ELISAarrayReader/' \
    #              'images_scienion/Plates_given_to_manu/2020-01-15_plate4_AEP_Feb3_6mousesera'
    input_path = "/Volumes/GoogleDrive/My Drive/ELISAarrayReader/images_octopi/20200325AdamsPlate/Averaged/500us"
    # output_path = '/Users/shalin.mehta/Documents/images_local/2020-01-15_plate4_AEP_Feb3_6mousesera/'

    output_path = '/Users/ivan.ivanov/Documents/images_local/' \
                  'Plates_given_to_manu/2020-01-15_plate4_AEP_Feb3_6mousesera'
    method = 'interp' # 'fit' or 'interp'
    flags = ['-i', input_path, '-o', output_path, '-d', '-m', method]
    # main(flags)
    main(sys.argv[1:])
