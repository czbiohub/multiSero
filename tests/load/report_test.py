import collections
import numpy as np
import os
import pandas as pd
import pytest

import multiSero.array_analyzer.extract.constants as constants
import multiSero.array_analyzer.load.report as report


@pytest.fixture
def report_test(tmpdir_factory):
    input_dir = tmpdir_factory.mktemp("input_dir")
    constants.RUN_PATH = input_dir
    antigen_array = np.empty(shape=(2, 3), dtype='U100')
    antigen_array[0, 0] = 'antigen_0_0'
    antigen_array[1, 2] = 'antigen_1_2'
    constants.ANTIGEN_ARRAY = antigen_array
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    plate_df = pd.DataFrame(None, index=rows, columns=cols)
    report_test = collections.OrderedDict()
    for antigen_name in ['0_0_antigen_0_0', '1_2_antigen_1_2']:
        report_test[antigen_name] = plate_df.copy()
    # Assign some values to a report
    report_test['0_0_antigen_0_0'].at['A', '3'] = .75
    report_test['0_0_antigen_0_0'].at['H', '12'] = .5
    report_test['1_2_antigen_1_2'].at['A', '3'] = .1
    report_test['1_2_antigen_1_2'].at['H', '12'] = .2
    return report_test


def test_report_init():
    antigen_array = np.empty(shape=(2, 3), dtype='U100')
    antigen_array[0, 0] = 'antigen_0_0'
    antigen_array[1, 2] = 'antigen_1_2'
    antigen_array[0, 1] = 'suuuuuuuuper_loooooooong_antigen_name'
    constants.ANTIGEN_ARRAY = antigen_array
    constants.RUN_PATH = 'test_run_dir'
    # Create instance
    reporter = report.ReportWriter()
    # Check paths
    assert reporter.od_path == 'test_run_dir/median_ODs.xlsx'
    assert reporter.int_path == 'test_run_dir/median_intensities.xlsx'
    assert reporter.bg_path == 'test_run_dir/median_backgrounds.xlsx'
    # Check that antigens have correct names and indices
    antigen_df = reporter.antigen_df
    assert antigen_df.shape == (3, 3)
    antigen = antigen_df.loc[(antigen_df['grid_row'] == 0) &
                             (antigen_df['grid_col'] == 0), 'antigen'].values[0]
    assert antigen == '0_0_antigen_0_0'

    antigen = antigen_df.loc[(antigen_df['grid_row'] == 1) &
                             (antigen_df['grid_col'] == 2), 'antigen'].values[0]
    assert antigen == '1_2_antigen_1_2'

    antigen = antigen_df.loc[(antigen_df['grid_row'] == 0) &
                             (antigen_df['grid_col'] == 1), 'antigen'].values[0]
    assert antigen == '0_1_suuuuuuuuper_loooooooong_an'


def test_antigen_df():
    antigen_array = np.empty(shape=(2, 3), dtype='U100')
    antigen_array[0, 0] = 'antigen_0_0'
    antigen_array[1, 2] = 'antigen_1_2'
    constants.ANTIGEN_ARRAY = antigen_array
    # Create instance
    reporter = report.ReportWriter()
    antigen_df = reporter.get_antigen_df()
    assert antigen_df.shape == (2, 3)
    assert list(antigen_df) == ['antigen', 'grid_row', 'grid_col']
    assert list(antigen_df['antigen'].values) == \
           ['0_0_antigen_0_0', '1_2_antigen_1_2']


def test_create_new_reports(report_test):
    # Check the dicts that will be reports
    reporter = report.ReportWriter()
    reporter.create_new_reports()
    assert list(reporter.report_int) == [
        '0_0_antigen_0_0',
        '1_2_antigen_1_2',
    ]
    # Check first dataframe
    plate_df = reporter.report_int['0_0_antigen_0_0']
    assert plate_df.shape == (8, 12)
    assert list(plate_df) ==\
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    assert list(plate_df.index) == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


def test_load_existing_reports(report_test):
    # Write od, intensity and background reports
    antigen_names = ['0_0_antigen_0_0', '1_2_antigen_1_2']
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_backgrounds.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    # Load existing reports and make sure they're the same
    reporter = report.ReportWriter()
    reporter.load_existing_reports()
    for antigen_name in antigen_names:
        int_df = reporter.report_int[antigen_name]
        int_df.equals(report_test[antigen_name])
        bg_df = reporter.report_bg[antigen_name]
        bg_df.equals(report_test[antigen_name])
        od_df = reporter.report_od[antigen_name]
        od_df.equals(report_test[antigen_name])


def test_load_missing_reports(report_test):
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_load_missing_int_reports(report_test):
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in ['0_0_antigen_0_0', '1_2_antigen_1_2']:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_load_missing_bg_reports(report_test):
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in ['0_0_antigen_0_0', '1_2_antigen_1_2']:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in ['0_0_antigen_0_0', '1_2_antigen_1_2']:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_load_existing_reports_wrong_antigens_od(report_test):
    # Write od, intensity and background reports
    antigen_names = ['0_0_antigen_0_0', '1_2_antigen_1_2']
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_backgrounds.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name + 'wrong')
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_load_existing_reports_wrong_antigens_int(report_test):
    # Write od, intensity and background reports
    antigen_names = ['0_0_antigen_0_0', '1_2_antigen_1_2']
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name + 'wrong')
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_backgrounds.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_load_existing_reports_wrong_antigens_bg(report_test):
    # Write od, intensity and background reports
    antigen_names = ['0_0_antigen_0_0', '1_2_antigen_1_2']
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_intensities.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name + 'wrong')
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_backgrounds.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    xlsx_path = os.path.join(constants.RUN_PATH, 'median_ODs.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        for antigen_name in antigen_names:
            sheet_df = report_test[antigen_name]
            sheet_df.to_excel(writer, sheet_name=antigen_name)
    # Make sure we get an assertion error
    reporter = report.ReportWriter()
    with pytest.raises(AssertionError):
        reporter.load_existing_reports()


def test_assign_well_to_plate():
    well_name = 'D11'
    # Create some spots
    df_cols = ['grid_row', 'grid_col', 'intensity_median', 'bg_median', 'od_norm']
    spots_df = pd.DataFrame(columns=df_cols)
    df_row = {'grid_row': 0,
              'grid_col': 0,
              'intensity_median': 1.,
              'bg_median': .5,
              'od_norm': .75}
    spots_df = spots_df.append(df_row, ignore_index=True)
    df_row = {'grid_row': 0,
              'grid_col': 1,
              'intensity_median': 0.,
              'bg_median': .5,
              'od_norm': .2}
    spots_df = spots_df.append(df_row, ignore_index=True)
    df_row = {'grid_row': 1,
              'grid_col': 2,
              'intensity_median': .1,
              'bg_median': .2,
              'od_norm': .3}
    spots_df = spots_df.append(df_row, ignore_index=True)
    # Create antigens with matching indices
    antigen_array = np.empty(shape=(2, 3), dtype='U100')
    antigen_array[0, 0] = 'antigen_0_0'
    antigen_array[1, 2] = 'antigen_1_2'
    constants.ANTIGEN_ARRAY = antigen_array
    # Assign wells to plate and check values
    reporter = report.ReportWriter()
    reporter.create_new_reports()
    reporter.assign_well_to_plate(well_name=well_name, spots_df=spots_df)
    assert list(reporter.report_int['0_0_antigen_0_0']) ==\
           ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    assert list(reporter.report_int['0_0_antigen_0_0'].index) ==\
           ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    assert reporter.report_int['0_0_antigen_0_0'].at['D', '11'] == 1.
    assert reporter.report_bg['0_0_antigen_0_0'].at['D', '11'] == .5
    assert reporter.report_od['0_0_antigen_0_0'].at['D', '11'] == .75
    assert reporter.report_int['1_2_antigen_1_2'].at['D', '11'] == .1
    assert reporter.report_bg['1_2_antigen_1_2'].at['D', '11'] == .2
    assert reporter.report_od['1_2_antigen_1_2'].at['D', '11'] == .3


def test_write_reports(tmpdir_factory):
    output_dir = tmpdir_factory.mktemp("output_dir")
    constants.RUN_PATH = output_dir
    antigen_array = np.empty(shape=(2, 3), dtype='U100')
    antigen_array[0, 0] = 'antigen_0_0'
    antigen_array[1, 2] = 'antigen_1_2'
    constants.ANTIGEN_ARRAY = antigen_array
    # Create fake spots dataframe
    df_cols = ['grid_row', 'grid_col', 'intensity_median', 'bg_median', 'od_norm']
    spots_df = pd.DataFrame(columns=df_cols)
    df_row = {'grid_row': 0,
              'grid_col': 0,
              'intensity_median': 1.,
              'bg_median': .5,
              'od_norm': .75}
    spots_df = spots_df.append(df_row, ignore_index=True)
    df_row = {'grid_row': 1,
              'grid_col': 2,
              'intensity_median': .1,
              'bg_median': .2,
              'od_norm': .3}
    spots_df = spots_df.append(df_row, ignore_index=True)
    # Create report instance and write a few wells
    reporter = report.ReportWriter()
    reporter.create_new_reports()
    reporter.assign_well_to_plate('C11', spots_df)
    reporter.assign_well_to_plate('D4', spots_df)
    spots_df.loc[1, 'od_norm'] = 10
    reporter.assign_well_to_plate('A7', spots_df)
    reporter.write_reports()
    # Load reports and check values
    od_path = os.path.join(output_dir, 'median_ODs.xlsx')
    report_od = pd.read_excel(od_path, sheet_name=None, index_col=0)
    assert report_od['0_0_antigen_0_0'].at['C', '11'] == .75
    assert report_od['1_2_antigen_1_2'].at['C', '11'] == .3
    assert report_od['0_0_antigen_0_0'].at['D', '4'] == .75
    assert report_od['1_2_antigen_1_2'].at['D', '4'] == .3
    assert report_od['0_0_antigen_0_0'].at['A', '7'] == .75
    assert report_od['1_2_antigen_1_2'].at['A', '7'] == 10.
