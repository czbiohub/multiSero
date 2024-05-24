"""
The constants.py namespace contains constants used throughout the repo
    These values are assigned at the start of analysis using the "metadata.MetaData" class
    Set the values by populating the .xml or .xlsx file as per documentation
"""
# === runtime arguments ===
EXTRACT_OD = None
ANALYZE_OD = None
INPUT_FOLDER = None
OUTPUT_FOLDER = None
WORKFLOW = None
METADATA_FILE = None
DEBUG = None
LOAD_REPORT = None

# === constants parsed from metadata ===
#   the constants below are all dictionaries
params = {
    'rows': None,
    'columns': None,
    'v_pitch': None,
    'h_pitch': None,
    'spot_width': None,
    'bg_offset': None,
    'bg_thickness': None,
    'max_diam': None,
    'min_diam': None,
    'pixel_size_scienion': 0.0049,
    'pixel_size_octopi': 0.00185,
    'pixel_size': None,
    'nbr_outliers': 1
}

# a map between Image Name : well (row, col)
IMAGE_TO_WELL = dict()
# If there's a sheet in xlsx call 'rerun_wells', only those well names will be run
RERUN = False
RERUN_WELLS = []

# Column names for dataframe that holds all spot properties
SPOT_DF_COLS = ['grid_row',
                'grid_col',
                'centroid_row',
                'centroid_col',
                'intensity_mean',
                'intensity_median',
                'bg_mean',
                'bg_median',
                'od_norm',
                'bbox_row_min',
                'bbox_row_max',
                'bbox_col_min',
                'bbox_col_max'
                ]

# === array-constants ===
#   the constants below are all np.ndarrays whose elements are "U100" strings
SPOT_ID_ARRAY = None
SPOT_TYPE_ARRAY = None
FIDUCIAL_ARRAY = None
ANTIGEN_ARRAY = None

# ndarrays representing 96-well plates for each analysis type
WELL_OD_ARRAY = None
WELL_INT_ARRAY = None
WELL_BG_ARRAY = None

# === constants needed for workflows ===
#   these values are extracted from the above ARRAYs

# list of (column, row) fiducial locations
FIDUCIALS = []

# list of int fiducial location in array.  Example for 6x6 array: (0,0) = 0, (0,1) = 6, (5,5) = 35
FIDUCIALS_IDX = []

# values used by point_registration.py
SPOT_DIST_PIX = int()
SPOT_DIST_UM = int()
STDS = [100, 100, 2, .01]  # x, y, angle, scale
NBR_PARTICLES = 4000
REG_DIST_THRESH = 100
MEAN_POINT = (0, 0)
SCALE_MEAN = 1.
ANGLE_MEAN = 0.

# Requirement of minimum number of detected spots
MIN_NBR_SPOTS = 5
# Minimum detected spot percentage of spot ROI area
SPOT_MIN_PERCENT_AREA = .1

# constants for saving
RUN_PATH = ''

# Logger
LOG_NAME = 'multisero.log'

# template for writing OD, INT, BG worksheets
WELL_OUTPUT_TEMPLATE = {'A': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'B': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'C': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'D': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'E': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'F': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'G': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None},
                        'H': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
                              7: None, 8: None, 9: None, 10: None, 11: None, 12: None}
                        }
