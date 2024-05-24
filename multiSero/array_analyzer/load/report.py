import collections
from copy import deepcopy
import logging
import numpy as np
import os
import pandas as pd

import multiSero.array_analyzer.extract.constants as constants


class ReportWriter:
    """
    Class for handling plate reports.
    It pairs antigen names with well grid locations, and creates reports
    where each sheet correspond to a given antigen at a given grid location.
    Each sheet is a dataframe corresponding to all wells in a plate.
    Plates are traditionally represented with numerical columns and
    alphabetical rows.
    """
    def __init__(self):
        """
        Create dataframe with antigen names and well grid locations.
        """
        self.logger = logging.getLogger(constants.LOG_NAME)

        # Dataframe for antigen positions on grid
        self.antigen_df = pd.DataFrame(columns=['antigen', 'grid_row', 'grid_col'])
        for antigen_position, antigen in np.ndenumerate(constants.ANTIGEN_ARRAY):
            if antigen == '' or antigen is None:
                continue
            # Abbreviate antigen name if too long
            sheet_name = f'{antigen_position[0]}_{antigen_position[1]}_{antigen}'
            if len(sheet_name) >= 31:
                self.logger.warning(
                    "Sheet name: {} is too long, truncating".format(sheet_name),
                )
                sheet_name = sheet_name[:31]
            # Add grid row and column for antigen
            idx_row = {'antigen': sheet_name,
                       'grid_row': antigen_position[0],
                       'grid_col': antigen_position[1]}
            self.antigen_df = self.antigen_df.append(idx_row, ignore_index=True)

        self.antigen_names = list(self.antigen_df['antigen'].values)
        self.report_int = None
        self.report_bg = None
        self.report_od = None
        # Report paths
        self.od_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
        self.int_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
        self.bg_path = os.path.join(constants.RUN_PATH, 'median_backgrounds.xlsx')

    def get_antigen_df(self):
        """
        Returns dataframe with antigen names and their locations on the grid.

        :return pd.DataFrame antigen_df: Antigen names and grid rows, cols
        """
        return self.antigen_df

    def create_new_reports(self):
        """
        Creates three new reports with sheets corresponding to antigen names.
        """
        # Dataframe for a whole plate
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        plate_df = pd.DataFrame(None, index=rows, columns=cols)

        report_dict = collections.OrderedDict()
        for sheet_name in self.antigen_names:
            # Add empty plate dataframe to antigen
            report_dict[sheet_name] = plate_df.copy()

        self.report_int = deepcopy(report_dict)
        self.report_bg = deepcopy(report_dict)
        self.report_od = deepcopy(report_dict)

    def load_existing_reports(self):
        """
        If doing a rerun, load existing reports and make sure the sheet names
        (keys) match the ones from the current run.
        Rerun wells will be added to existing reports and rewritten.
        """
        assert os.path.isfile(self.od_path), \
            "OD report doesn't exist: {}".format(self.od_path)
        assert os.path.isfile(self.int_path), \
            "Intensity report doesn't exist: {}".format(self.int_path)
        assert os.path.isfile(self.bg_path), \
            "Background report doesn't exist: {}".format(self.bg_path)
        # Read reports and make sure they have the right keys
        ordered_dict = pd.read_excel(self.od_path, sheet_name=None, index_col=0)
        assert list(ordered_dict) == self.antigen_names, \
            "Existing report keys don't match current keys"
        self.report_od = ordered_dict
        self.logger.debug('Loaded existing OD report')
        ordered_dict = pd.read_excel(self.int_path, sheet_name=None, index_col=0)
        assert list(ordered_dict) == self.antigen_names, \
            "Existing report keys don't match current keys"
        self.report_int = ordered_dict
        self.logger.debug('Loaded existing intensity report')
        ordered_dict = pd.read_excel(self.bg_path, sheet_name=None, index_col=0)
        assert list(ordered_dict) == self.antigen_names, \
            "Existing report keys don't match current keys"
        self.report_bg = ordered_dict
        self.logger.debug('Loaded existing background report')

    def assign_well_to_plate(self, well_name, spots_df):
        """
        Takes intensity, background and OD values for a well and
        reassigns them by antigens for the plate.

        :param str well_name: Well name (e.g. 'B12')
        :param pd.DataFrame spots_df: Metrics for all spots in a well
        """
        plate_row = well_name[0]
        plate_col = well_name[1:]
        for idx, antigen_row in self.antigen_df.iterrows():
            spots_row = spots_df.loc[
                (spots_df['grid_row'] == antigen_row['grid_row']) &
                (spots_df['grid_col'] == antigen_row['grid_col']),
            ]
            self.report_int[antigen_row['antigen']].at[plate_row, plate_col] = \
                spots_row['intensity_median'].values[0]
            self.report_bg[antigen_row['antigen']].at[plate_row, plate_col] = \
                spots_row['bg_median'].values[0]
            self.report_od[antigen_row['antigen']].at[plate_row, plate_col] = \
                spots_row['od_norm'].values[0]
        self.logger.debug("Assigned well {} to plate reports".format(well_name))

    def write_reports(self):
        """
        After all wells are run, write plate based reports for OD,
        intensity, and background.
        """
        # Write OD report
        with pd.ExcelWriter(self.od_path) as writer:
            for antigen_name in self.antigen_names:
                sheet_df = self.report_od[antigen_name]
                sheet_df.to_excel(writer, sheet_name=antigen_name)
        self.logger.debug("Wrote OD plate report")
        # Write intensity report
        with pd.ExcelWriter(self.int_path) as writer:
            for antigen_name in self.antigen_names:
                sheet_df = self.report_int[antigen_name]
                sheet_df.to_excel(writer, sheet_name=antigen_name)
        self.logger.debug("Wrote intensity plate report")
        # Write background report
        with pd.ExcelWriter(self.bg_path) as writer:
            for antigen_name in self.antigen_names:
                sheet_df = self.report_bg[antigen_name]
                sheet_df.to_excel(writer, sheet_name=antigen_name)
        self.logger.debug("Wrote background plate report")
