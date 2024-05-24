import os
import numpy as np
import pandas as pd
import shutil

import multiSero.array_analyzer.extract.txt_parser as txt_parser
import multiSero.array_analyzer.extract.constants as constants


class MetaData:

    def __init__(self, input_folder_, output_folder_):
        """
        Parses metadata spreadsheets then populates all necessary ARRAY data structures
        Extracts all necessary constants and assigns them in the constants.py namespace

        :param input_folder_: str full path to metadata spreadsheet
        :param output_folder_: str full path to output folder for reports and diagnostics
        """
        self.fiduc = None
        self.spots = None
        self.repl = None
        self.params = None
        self.xlsx_path = None
        self.xml_path = None
        constants.INPUT_FOLDER = input_folder_
        constants.OUTPUT_FOLDER = output_folder_
        metadata_split = constants.METADATA_FILE.split('.')
        # In case of a 'well' run
        if len(metadata_split) == 1:
            assert metadata_split[0] == 'well',\
                "Only metadata without extension allowed is 'well,"\
                "not {}".format(metadata_split[0])
            return
        elif len(metadata_split) == 2:
            self.metadata_extension = metadata_split[-1]
        else:
            raise IOError("Metadata file must be of type"
                          "file_name.extension or 'well'"
                          "not {}".format(constants.METADATA_FILE))

        # parse fiducials, spot types, antigens, and hardware parameters from metadata
        if self.metadata_extension == 'xml':
            # check that .xml exists
            if constants.METADATA_FILE not in os.listdir(input_folder_):
                raise IOError("xml file not found, aborting")
            self.xml_path = os.path.join(input_folder_, constants.METADATA_FILE)
            # parsing .xml
            self.fiduc, self.spots, self.repl, self.params = txt_parser.create_xml_dict(self.xml_path)

        elif self.metadata_extension == 'csv':
            # check that three .csvs exist
            three_csvs = ['array_format_antigen', 'array_format_type', 'array_parameters']
            csvs = [f for f in os.listdir(input_folder_) if '.csv' in f]
            if len(csvs) != 3:
                raise IOError("incorrect number of .csv files found, aborting")
            for target in three_csvs:
                if True not in [target in file for file in csvs]:
                    raise IOError(f".csv file with substring {target} is missing")

            csv_paths = [os.path.join(input_folder_, one_csv) for one_csv in csvs]
            # parsing .csv
            self.fiduc, _, self.repl, self.params = txt_parser.create_csv_dict(csv_paths)

        elif self.metadata_extension == 'xlsx':
            self.xlsx_path = os.path.join(input_folder_, constants.METADATA_FILE)

            if constants.METADATA_FILE not in os.listdir(input_folder_):
                raise IOError("xlsx file not found, aborting")

            # check that the xlsx file contains necessary worksheets
            sheets = pd.read_excel(self.xlsx_path, sheet_name=None)
            if 'imaging_and_array_parameters' not in sheets.keys():
                raise IOError("sheet by name 'imaging_and_array_parameters' not present in excel file, aborting")
            if 'antigen_array' not in sheets.keys():
                raise IOError("sheet by name 'array_antigens' not present in excel file, aborting")
            # Collect well names for rerun, if sheet exists
            if constants.RERUN:
                if 'rerun_wells' in sheets.keys():
                    constants.RERUN_WELLS = list(sheets['rerun_wells']['well_name'])
                    assert len(constants.RERUN_WELLS) > 0,\
                        "No rerun well names found"
                else:
                    raise IOError("Rerun flag given but no rerun_wells sheet")
            # parsing .xlsx
            self.fiduc, self.repl, self.params = txt_parser.create_xlsx_dict(sheets)

            # parsing .xlsx using pandas !! not tested or finished yet
            # self.fiduc, _, self.repl, self.params = txt_parser.create_xlsx_array(xlsx_path)
            # c.FIDUCIAL_ARRAY = self.fiduc
            # c.ANTIGEN_ARRAY = self.repl

        else:
            raise NotImplementedError(
                f"metadata with extension {self.metadata_extension} is not supported"
            )

        # set hardware and array parameters
        self._assign_params()

        # setting constant arrays
        if self.metadata_extension == 'xml':
            self._create_spot_id_array()
            self._create_spot_type_array()
        self._create_fiducials_array()
        self._create_antigen_array()

        # setting location of fiducials and other useful parameters
        self._calculate_fiduc_coords()
        self._calculate_fiduc_idx()
        self._calc_spot_dist()
        if constants.RERUN:
            # Rerun certain wells in existing run path
            assert os.path.isdir(constants.RUN_PATH),\
                "Can't find re-run dir {}".format(constants.RUN_PATH)
            # Make sure it's a multisero directory
            base_path = os.path.basename(os.path.normpath(constants.RUN_PATH))
            assert base_path[:10] == 'multisero_',\
                "Rerun path should be a multisero_... path, not".format(constants.RUN_PATH)

        self._copy_metadata_to_output()

    def _assign_params(self):
        constants.params['rows'] = int(self.params['rows'])
        constants.params['columns'] = int(self.params['columns'])
        constants.params['v_pitch'] = float(self.params['v_pitch'])
        constants.params['h_pitch'] = float(self.params['h_pitch'])
        constants.params['spot_width'] = float(self.params['spot_width'])
        if self.metadata_extension == 'xml':
            constants.params['pixel_size'] = constants.params['pixel_size_scienion']
        else:
            constants.params['pixel_size'] = float(self.params['pixel_size'])
        if 'nbr_outliers' in self.params:
            constants.params['nbr_outliers'] = int(self.params['nbr_outliers'])

    def _create_spot_id_array(self):
        """
        Creates an empty ndarray of strings, whose rows and columns match the printed array's rows/cols
            Sets the ARRAY constant corresponding to "Spot-ID" based on metadata
        *** note: this is used ONLY for .xml metadata files generated by the sciReader ***
        :return:
        """
        self.spot_ids = np.empty(
            shape=(constants.params['rows'], constants.params['columns']),
            dtype='U100',
        )
        constants.SPOT_ID_ARRAY = txt_parser.populate_array_id(
            self.spot_ids,
            self.spots,
        )

    def _create_spot_type_array(self):
        """
        Creates an empty ndarray of strings, whose rows and columns match the printed array's rows/cols
            Sets the ARRAY constant corresponding to "Spot Type" based on metadata
            "Spot Type" are values like "Positive Control", "Diagnostic"
        :return:
        """
        self.spot_type = np.empty(
            shape=(constants.params['rows'], constants.params['columns']),
            dtype='U100',
        )
        constants.SPOT_TYPE_ARRAY = txt_parser.populate_array_spots_type(
            self.spot_type,
            self.spots,
            self.fiduc,
        )

    def _create_fiducials_array(self):
        """
        Creates an empty ndarray of strings, whose rows and columns match the printed array's rows/cols
            Sets the ARRAY constant corresponding to "Fiducial" based on metadata
            "Fiducial" are values like "Fiducial" or "Reference, Diagnostic" and are used to align array spots
        :return:
        """
        self.fiducials_array = np.empty(
            shape=(constants.params['rows'], constants.params['columns']),
            dtype='U100',
        )
        constants.FIDUCIAL_ARRAY = txt_parser.populate_array_fiduc(
            self.fiducials_array,
            self.fiduc,
        )

    def _create_antigen_array(self):
        """
        Creates an empty ndarray of strings, whose rows and columns match the printed array's rows/cols
            Assigns the corresponding "Antigen" based on metadata
            "Antigens" are descriptive values of the antigen at each spot location.

        This is the only array creator that requires one of the above arrays (SPOT_ID_ARRAY, .xml meta ONLY)
            multiple "spot-ids" can contain the same "antigen".  Thus it's required to pass the "SPOT_ID_ARRAY"
        :return:
        """
        self.antigen_array = np.empty(
            shape=(constants.params['rows'], constants.params['columns']),
            dtype='U100',
        )
        if self.metadata_extension == 'xml':
            if constants.SPOT_ID_ARRAY.size == 0:
                raise AttributeError("attempting to create antigen array "
                                     "before SPOT_ID_ARRAY is assigned")
            constants.ANTIGEN_ARRAY = txt_parser.populate_array_antigen_xml(
                self.antigen_array,
                constants.SPOT_ID_ARRAY,
                self.repl,
            )
        elif self.metadata_extension == 'csv' or self.metadata_extension == 'xlsx':
            constants.ANTIGEN_ARRAY = txt_parser.populate_array_antigen(
                self.antigen_array,
                self.repl,
            )

    @staticmethod
    def _calculate_fiduc_coords():
        """
        Calculate and set the fiducial coordinates like:
            FIDUCIALS = [(0, 0), (0, 1), (0, 5), (7, 0), (7, 5)]
            fiducial coordinates are labeled as "Reference, Diagnostic"
        :return:
        """
        x, y = np.where(constants.FIDUCIAL_ARRAY == 'Reference, Diagnostic')
        if x.size == 0 or y.size == 0:
            x, y = np.where(constants.FIDUCIAL_ARRAY == 'Fiducial')
        constants.FIDUCIALS = list(zip(x, y))

    @staticmethod
    def _calculate_fiduc_idx():
        """
        Calculate fiducial index like
            FIDUCIALS_IDX = [0, 5, 6, 30, 35]\
            FIDUCIALS_IDX = [0, 7, 8, 40, 47] for 8 columns
        :return:
        """
        constants.FIDUCIALS_IDX = list(np.where(
            constants.FIDUCIAL_ARRAY.flatten() == 'Reference, Diagnostic')[0])
        if len(constants.FIDUCIALS_IDX) == 0:
            constants.FIDUCIALS_IDX = list(np.where(
                constants.FIDUCIAL_ARRAY.flatten() == 'Fiducial')[0])

    @staticmethod
    def _calc_spot_dist():
        """
        Calculate distance between spots in both pixels and microns
        :return:
        """
        v_pitch_mm = constants.params['v_pitch']
        h_pitch_mm = constants.params['h_pitch']
        pix_size = constants.params['pixel_size']

        # assuming similar v_pitch and h_pitch, average to get the SPOT_DIST
        v_pitch_pix = v_pitch_mm/pix_size
        h_pitch_pix = h_pitch_mm/pix_size
        constants.SPOT_DIST_PIX = np.mean([v_pitch_pix, h_pitch_pix]).astype('uint8')

        # convert the SPOT_DIST to microns, 0 - 255
        constants.SPOT_DIST_UM = np.mean([v_pitch_mm * 1000, h_pitch_mm * 1000]).astype('uint8')

    def _copy_metadata_to_output(self):
        if self.metadata_extension == 'xlsx':
            shutil.copy2(self.xlsx_path, constants.RUN_PATH)
        elif self.metadata_extension == 'xml':
            shutil.copy2(self.xml_path, constants.RUN_PATH)
