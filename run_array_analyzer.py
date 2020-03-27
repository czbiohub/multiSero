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
import sys, getopt

from array_analyzer.extract.image_parser import *
from array_analyzer.extract.txt_parser import *
from array_analyzer.load.xlsx_report import *
from array_analyzer.load.debug_images import *

import time
from datetime import datetime
import skimage as si
import skimage.io as io


def main(argv):
    inputfolder = ''
    outputfolder = ''
    debug = False
    try:
        opts, args = getopt.getopt(argv, "hdi:o:", ["ifile=", "ofile=", "debug"])
    except getopt.GetoptError:
        print('run_array_analyzer.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('run_array_analyzer.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfolder = arg
        elif opt in ("-o", "--ofile"):
            outputfolder = arg
        elif opt in ("-d", "--debug"):
            print('debug mode on, saving well and spot images')
            debug = True

    if not os.path.isdir(inputfolder):
        raise ValueError("input folder is not a folder or not supplied")

    if not os.path.isdir(outputfolder):
        raise ValueError("output folder is not a folder or not supplied")

    workflow(inputfolder, outputfolder, debug=debug)


def workflow(input_folder_, output_folder_, debug=False):

    xml = [f for f in os.listdir(input_folder_) if '.xml' in f]
    if len(xml) > 1:
        raise IOError("more than one .xml file found, aborting")

    xml_path = input_folder_+os.sep+xml[0]

    # parsing .xml
    fiduc, spots, repl, params = create_xml_dict(xml_path)

    # creating our arrays
    spot_ids = create_array(params['rows'], params['columns'])
    antigen_array = create_array(params['rows'], params['columns'])
    props_array = create_array(params['rows'], params['columns'], dtype=object)

    # adding .xml info to these arrays
    spot_ids = populate_array_id(spot_ids, spots)
    antigen_array = populate_array_antigen(antigen_array, spot_ids, repl)

    xlsx_workbook = create_base_template()

    # save a sub path for this processing run
    run_path = output_folder_ + os.sep + f'run_{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}'
    if not os.path.isdir(run_path):
        os.mkdir(run_path)

    # ================
    # loop over images => good place for multiproc?  careful with columns in report
    # ================
    for image, image_name in read_to_grey(input_folder_):
        start = time.time()
        print(image_name)

        # finding center of well and cropping
        cx, cy, r = find_well_center(image, method='otsu')
        im_crop = crop_image(image, cx, cy, r, border_=0)

        # ===================
        # Spot detection
        # ====================
        # rosin
        spotmask = thresh_and_binarize(im_crop, method='rosin')

        # alternative method: use adaptive threshold pipeline
        #  *** I found this includes too much signal and ends up confusing spot assignment later ***
        # spotmask = adaptive_threshold(im_crop)

        # TODO: Syuan-Ming implement background correction by surface fit
        spot_background = generate_spot_background(spotmask)

        props = generate_props(spotmask, intensity_image_=im_crop)
        bgprops = generate_props(spot_background, intensity_image_=im_crop)

        # apply filters to the region props lists
        props = filter_props(props, attribute="area", condition="greater_than", condition_value=200)
        props = filter_props(props, attribute="eccentricity", condition="less_than", condition_value=0.5)

        # insert some grid generation
        # function that takes bounds and outputs binary grid mask
        # feed this mask into labels, then region props

        bgprops = filter_props(bgprops, attribute="area", condition="greater_than", condition_value=200)
        bgprops = filter_props(bgprops, attribute="eccentricity", condition="less_than", condition_value=0.5)

        centroid_map = generate_props_dict(props,
                                           params['rows'],
                                           params['columns'],
                                           min_area=100)
        props_array = assign_props_to_array(props_array, centroid_map)

        # use the props_array to find boundaries, fit a new periodic grid to the image
        fitted_spotmask = build_block_array(props_array, spotmask, 6, 8)
        props_fit = generate_props(fitted_spotmask, intensity_image_=im_crop)
        props_fit_map = generate_props_dict(props_fit,
                                            params['rows'],
                                            params['columns'],
                                            min_area=0)
        props_array_fit = assign_props_to_array(props_array, props_fit_map)

        # todo: further calculations using bgprops, props here

        # xlsx report generation
        xlsx_workbook = populate_main_tab(xlsx_workbook, spot_ids, props_array, image_name[:-4])
        xlsx_workbook = populate_main_replicates(xlsx_workbook, props_array, antigen_array, image_name[:-4])

        stop = time.time()
        print(f"\ttime to process={stop-start}")

        # SAVE FOR DEBUGGING
        if debug:
            well_path = run_path+os.sep+image_name[:-4]
            os.mkdir(well_path)

            #   save cropped image and the binary
            si.io.imsave(well_path+os.sep+image_name[:-4]+"_crop.png", (255*im_crop).astype('uint8'))
            si.io.imsave(well_path + os.sep + image_name[:-4] + "_crop_binary.png", (255*spotmask).astype('uint8'))
            si.io.imsave(well_path + os.sep + image_name[:-4] + "_crop_binary_filt.png", (255 * fitted_spotmask).astype('uint8'))


            #   save spots
            save_all_wells(props_array, spot_ids, well_path, image_name[:-4])

            #   save a composite of all spots, where spots are from source or from region prop
            save_composite_spots(im_crop, props_array_fit, well_path, image_name[:-4], from_source=True)
            save_composite_spots(im_crop, props_array_fit, well_path, image_name[:-4], from_source=False)

            #
            # t = np.mean(im_crop)*np.ones(shape=(spotmask.shape[0]+64, spotmask.shape[1]+64))
            # t = create_composite_spots(t, props_array, im_crop)
            # si.io.imsave(well_path + os.sep + image_name[:-4] + f"_composite_spots_img.png",
            #           (255 * t).astype('uint8'))
            #
            # t = np.mean(im_crop)*np.ones(shape=(spotmask.shape[0]+64, spotmask.shape[1]+64))
            # t = create_composite_spots(t, props_array)
            # si.io.imsave(well_path + os.sep + image_name[:-4] + f"_composite_spots_prop.png",
            #           (255 * t).astype('uint8'))

    # SAVE COMPLETED WORKBOOK
    xlsx_workbook.save(run_path + os.sep +
                       f'testrun_{datetime.now().year}_'
                       f'{datetime.now().month}{datetime.now().day}_'
                       f'{datetime.now().hour}{datetime.now().minute}.xlsx')


if __name__ == "__main__":

    path = '/Users/bryant.chhun/PycharmProjects/array-imager/Plates_given_to_manu/2020-01-15_plate4_AEP_Feb3_6mousesera'
    flags = ['-i', path, '-o', path, '-d']
    main(flags)

    # main(sys.argv[1:])

# todo:
#   consider new workflow
#   - get spot binary
#   - get spot labels, get spot region props
#   - using spot region props, filter and determine boundaries (average of bounding boxes or fixed bounding box for now)
#   - create binary mask containing multiple square regions - use fiducials to orient spacing, bounds
#   - create sub-binary mask that is the intersection of "spot binary" and "region binary"
#   - do region props on this sub-binary mask - then feed through rest of pipeline
