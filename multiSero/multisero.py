import argparse
import logging
import os

import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.utils.io_utils as io_utils
import multiSero.array_analyzer.workflows.registration_workflow as registration_wf
from multiSero import array_analyzer as interpolation_wf
import multiSero.array_analyzer.workflows.well_wf as well_wf
import multiSero.interpretation.od_analyzer as od_analyzer
import matplotlib
matplotlib.use('Agg')

def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    # make sure that only extract_od or analyze_od stages are passed.
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        '-e', '--extract_od',
        action= 'store_const',
        const=True,
        help="Segment spots and compute ODs",
    )
    stage.add_argument(
        '-a', '--analyze_od',
        action='store_const',
        const=True,
        help="Generate OD analysis plots",
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Input directory path",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="Output directory path, where a timestamped subdir will be generated. "
             "In case of rerun, give path to timestamped run directory",
    )
    parser.add_argument(
        '-wf', '--workflow',
        type=str,
        choices=['well_segmentation', 'well_crop', 'array_interp', 'array_fit'],
        default='array_fit',
        help="Workflow to automatically identify and extract intensities from experiment. "
             "'Well' experiments are for standard ELISA.  "
             "'Array' experiments are for ELISA assays using antigen arrays printed with Scienion Array Printer "
             "Default: array_fit",
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        help="Write debug plots of well and spots. Default: False",
    )
    parser.set_defaults(debug=False)
    parser.add_argument(
        '-r', '--rerun',
        dest='rerun',
        action='store_true',
        help="Rerun wells listed in 'rerun_wells sheets of metadata file. Default: False",
    )

    parser.add_argument(
        '-m', '--metadata',
        type=str,
        default='multisero_output_data_metadata.xlsx',
        help="specify the file name for the experiment metadata. "
             "Assumed to be in the same directory as images. "
             "Default: 'multisero_output_data_metadata.xlsx'"
    )
    parser.set_defaults(load_report=False)
    parser.add_argument(
        '-l', '--load_report',
        dest='load_report',
        action='store_true',
        help="Load the saved master report in the output directory "
             "rather than the original OD reports in the config file"
             " which is slower. Default: False",
    )
    return parser.parse_args()


def extract_od(input_dir, output_dir, workflow):
    """
    For each image in input directory, run either interpolation
    or registration of fiducials (default) workflow.
    An xlsx file (and potentially debug plots) will be written to output directory.

    :param str input_dir: Input directory path
    :param str output_dir: Output directory path
    :param str workflow: str, one of 'array_interp', 'array_fit', 'well_segmentation', 'well_crop'
        <plate>_<method> format:
            <plate> describes the printing style of the antigen (array or ELISA)
            <method> describes the spot segmentation and extraction approach
    """

    if workflow == 'array_interp':
        interpolation_wf.interp(
            input_dir,
            output_dir,
        )
    elif workflow == 'array_fit':
        registration_wf.point_registration(
            input_dir,
            output_dir,
        )
    elif workflow == 'well_segmentation':
        well_wf.well_analysis(
            input_dir,
            output_dir,
            method='segmentation',
        )
    elif workflow == 'well_crop':
        well_wf.well_analysis(
            input_dir,
            output_dir,
            method='crop',
        )


def run_multisero(args):
    """
    Main function, handling logic for all subroutines

    :param args: Argparse arguments
    """
    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        raise ValueError("input directory is not a directory or doesn't exist")

    os.makedirs(output_dir, exist_ok=True)

    constants.METADATA_FILE = args.metadata
    constants.DEBUG = args.debug
    constants.RERUN = args.rerun
    constants.LOAD_REPORT = args.load_report

    constants.RUN_PATH = io_utils.make_run_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        rerun=constants.RERUN,
    )
    # Default log level is info, otherwise debug
    log_level = 20
    if constants.DEBUG:
        log_level = 10
    logger = io_utils.make_logger(
        log_dir=constants.RUN_PATH,
        logger_name=constants.LOG_NAME,
        log_level=log_level,
    )
    logger.info("input dir: {}".format(input_dir))
    logger.info("output dir: {}".format(output_dir))
    logger.info("run dir: {}".format(constants.RUN_PATH))

    if args.extract_od:
        logging.info("Extract OD workflow: {}".format(args.workflow))
        extract_od(
            input_dir=input_dir,
            output_dir=output_dir,
            workflow=args.workflow,
        )
    elif args.analyze_od:
        od_analyzer.analyze_od(
            input_dir=input_dir,
            output_dir=output_dir,
            load_report=args.load_report,
        )


if __name__ == '__main__':
    args = parse_args()
    run_multisero(args)
