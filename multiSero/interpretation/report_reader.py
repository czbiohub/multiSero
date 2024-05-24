import os
import numpy as np
import pandas as pd
import logging
import multiSero.array_analyzer.extract.constants as constants

def antigen2D_to_df1D(xlsx_path, sheet, data_col):
    """
    Convert old 2D output format (per antigen) to 1D dataframe
    :param str xlsx_path: path to the xlsx file
    :param str sheet: sheet name to load
    :param str data_col: new column name of the linearized values
    :return dataframe df: linearized dataframe
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    df = df.unstack().reset_index(name=data_col)  # linearize the table
    df.rename(columns={'level_1': 'antigen_row', 'level_0': 'antigen_col'}, inplace=True)
    df[['antigen_row', 'antigen_col']] = df[['antigen_row', 'antigen_col']].applymap(int)
    df = df[['antigen_row', 'antigen_col', data_col]]
    df.dropna(inplace=True)
    return df


def well2D_to_df1D(xlsx_path, sheet, data_col):
    """
    Convert new 2D output format (per well) to 1D dataframe
    :param str xlsx_path: path to the xlsx file
    :param str sheet: sheet name to load
    :param str data_col: new column name of the linearized values
    :return dataframe df: linearized dataframe
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    df = df.unstack().reset_index(name=data_col)  # unpivot (linearize) the table
    df.rename(columns={'level_1': 'row_id', 'level_0': 'col_id'}, inplace=True)
    df['well_id'] = df.row_id + df.col_id.map(str)
    df = df[['well_id', data_col]]
    return df


def read_plate_info(metadata_xlsx):
    """read plate info from the metadata"""
    print('Reading the plate info...')

    sheet_names = ['serum ID',
                   'serum dilution',
                   'serum type',
                   'serum cat',
                   'secondary ID',
                   'secondary dilution',
                   'visit value',
                   'sample type']
    plate_info_df = pd.DataFrame()
    # get sheet names that are available in metadata
    sheet_names = list(set(metadata_xlsx.sheet_names).intersection(sheet_names))
    for sheet_name in sheet_names:
        sheet_df = pd.read_excel(metadata_xlsx, sheet_name=sheet_name, index_col=0)
        sheet_df = sheet_df.unstack().reset_index(name=sheet_name)  # unpivot (linearize) the table
        sheet_df.rename(columns={'level_1': 'row_id', 'level_0': 'col_id'}, inplace=True)
        if plate_info_df.empty:
            plate_info_df = sheet_df
        else:
            plate_info_df = pd.merge(plate_info_df,
                                     sheet_df,
                                     how='left', on=['row_id', 'col_id'])
    plate_info_df['well_id'] = plate_info_df.row_id + plate_info_df.col_id.map(str)
    sheet_names.append('well_id')
    # convert to number and non-numeric to NaN
    plate_info_df['serum dilution'] = \
        plate_info_df['serum dilution'].apply(pd.to_numeric, errors='coerce')
    logger = logging.getLogger(constants.LOG_NAME)
    nan_cols = plate_info_df.columns[plate_info_df.isnull().any()].tolist()
    if nan_cols:
        logger.warning("Parsing metadata failed for some wells in tab {}. "
                       "Please check info in these tabs are all filled out and in the right format "
                       "(e.g. dilutions should not contain strings)".format(nan_cols))
    plate_info_df.dropna(inplace=True)
    plate_info_df.drop(['row_id', 'col_id'], axis=1, inplace=True)
    if 'sample type' not in sheet_names:
        plate_info_df['sample type'] = 'Serum'
    return plate_info_df


def read_antigen_info(metadata_path):
    """read antigen info from the metadata"""
    print('Reading antigen information...')
    antigen_df = antigen2D_to_df1D(xlsx_path=metadata_path, sheet='antigen_array', data_col='antigen')
    antigen_type_df = antigen2D_to_df1D(xlsx_path=metadata_path, sheet='antigen_type', data_col='antigen type')
    antigen_df = pd.merge(antigen_df, antigen_type_df, how='left', on=['antigen_row', 'antigen_col'])
    return antigen_df


def read_multisero_output(file_path, antigen_df, file_type='od'):
    """
    read and re-format multisero spot fitting output
    :param str file_path: path to the multisero output xlsx file
    :param dataframe antigen_df:
    :param str file_type: output file type. 'od', 'int', or 'bg'
    :return: linearized dataframe
    """
    print('Reading {}...'.format(file_type))
    data_col = {'od': 'OD', 'int': 'intensity', 'bg': 'background'}
    data_df = pd.DataFrame()

    with pd.ExcelFile(file_path) as file:
        sheet_names = file.sheet_names
        for _, row in antigen_df.iterrows():
            if sheet_names[0][0].isnumeric(): # new format
                sheet_name = '{}_{}_{}'.format(row['antigen_row'], row['antigen_col'], row['antigen'])
            else:
                sheet_name = '{}_{}_{}_{}'.format(file_type, row['antigen_row'], row['antigen_col'], row['antigen'])
            data_1_antiten_df = well2D_to_df1D(xlsx_path=file, sheet=sheet_name, data_col=data_col[file_type])
            data_1_antiten_df['antigen_row'] = row['antigen_row']
            data_1_antiten_df['antigen_col'] = row['antigen_col']
            data_1_antiten_df['antigen'] = row['antigen']
            data_df = data_df.append(data_1_antiten_df, ignore_index=True)
    return data_df


def read_scn_output(file_path, plate_info_df):
    """
    Read scienion intensity output and convert it to OD
    :param str file_path: path to the scienion output xlsx file
    :param dataframe plate_info_df: plate info dataframe
    :return dataframe: scienion OD dataframe
    """
    # Read analysis output from Scienion
    scienion_df = pd.DataFrame()
    with pd.ExcelFile(file_path) as scienion_xlsx:
        for well_id in plate_info_df['well_id']:
            OD_1_antiten_df = pd.read_excel(scienion_xlsx, sheet_name=well_id)
            OD_1_antiten_df['well_id'] = well_id
            scienion_df = scienion_df.append(OD_1_antiten_df, ignore_index=True)
    # parse spot ids
    spot_id_df = scienion_df['ID'].str.extract(r'spot-(\d)-(\d)')
    spot_id_df = spot_id_df.astype(int) - 1  # index starting from 0
    spot_id_df.rename(columns={0: 'antigen_row', 1: 'antigen_col'}, inplace=True)
    scienion_df = pd.concat([spot_id_df, scienion_df], axis=1)
    scienion_df.drop('ID', axis=1, inplace=True)
    # invert the intensity and compute ODs
    df_scn = scienion_df.loc[:, ['antigen_row', 'antigen_col', 'well_id']]
    df_scn['intensity'] = 1 - scienion_df['Median'] / 255
    df_scn['background'] = 1 - scienion_df['Background Median'] / 255
    df_scn['OD'] = np.log10(df_scn['background'] / df_scn['intensity'])
    return df_scn


def slice_df(df, slice_action, column, keys):
    """
    Return sliced dataframe given the colume and keys
    :param df: dataframe to slice
    :param slice_action: 'keep' or 'drop'
    :param column: column to slice based on
    :param keys: key values to keep or drop
    :return:
    """
    if column is None or column != column:
        return df
    if not isinstance(keys, (list, np.ndarray)):
        if keys != keys or keys is None: # nan
            return df
        keys = [keys]
    if slice_action is None or slice_action != slice_action:
        return df
    elif slice_action == 'keep':
        df = df.loc[df[column].isin(keys), :]
    elif slice_action == 'drop':
        df = df.loc[~df[column].isin(keys), :]
    else:
        raise ValueError('slice action has to be "keep" or "drop", not "{}"'.format(slice_action))
    return df


def normalize_od_helper(norm_antigen):
    def normalize(df):
        norm_antigen_df = slice_df(df, 'keep', 'antigen', [norm_antigen])
        norm_factor = norm_antigen_df['OD'].mean()
        df['OD'] = df['OD'] / norm_factor
        return df
    return normalize


def normalize_od(df, norm_antigen=None, group='plate'):
    """
    Normalize OD by OD of the reference antigen
    :param dataframe df: dataframe containing serum OD info
    :param str norm_antigen: reference antigen to normalize by
    :param str group: unit to normalize. 'plate' or 'well'
    :return dataframe df: dataframe with normalized serum OD info
    """
    if norm_antigen is None:
        return df
    if group == 'plate':
        groupby_cols = ['plate ID', 'pipeline', 'sample type']
    elif group == 'well':
        groupby_cols = ['plate ID', 'well_id', 'pipeline', 'sample type']
    else:
        ValueError('normalization group has to be plate or well, not {}'.format(group))
    for pipeline in df['pipeline'].unique():
        for sample_type in df['sample type'].unique():
            norm_antigen_df = slice_df(df, 'keep', 'pipeline', [pipeline])
            norm_antigen_df = slice_df(norm_antigen_df, 'keep', 'sample type', [sample_type])
            norm_antigen_df = slice_df(norm_antigen_df, 'keep', 'antigen', [norm_antigen])
            df.loc[(df['antigen'] == norm_antigen) &
                   (df['pipeline'] == pipeline) &
                   (df['sample type'] == sample_type), 'OD'] = \
                norm_antigen_df['OD'] / norm_antigen_df['OD'].mean()
    norm_fn = normalize_od_helper(norm_antigen)
    df = df.groupby(groupby_cols).apply(norm_fn)
    return df


def offset_od_helper(norm_antigen):
    def offset(df):
        norm_antigen_df = slice_df(df, 'keep', 'antigen', [norm_antigen])
        norm_factor = norm_antigen_df['OD'].mean()
        df['OD'] = df['OD'] - norm_factor
        df.loc[df['OD'] < 0, 'OD'] = 0
        return df
    return offset


def offset_od(df, norm_antigen=None, group='plate'):
    """offset OD by OD of the reference antigen
    """
    if norm_antigen is None:
        return df
    if group == 'plate':
        groupby_cols = ['plate ID']
    elif group == 'well':
        groupby_cols = ['plate ID', 'well_id']
    else:
        ValueError('normalization group has to be plate or well, not {}'.format(group))
    norm_fn = offset_od_helper(norm_antigen)
    df = df.groupby(groupby_cols).apply(norm_fn)
    return df


def read_scn_output_batch(scn_dirs_df):
    """
    batch read scienion outputs
    :param dataframe scn_dirs_df: dataframe loaded from the analysis config
    containing directories of scienion output xlsx file, assuming the file name always
    ends with '_analysis.xlsx'
    :return dataframe scn_df: combined scienion OD dataframe from multiple outputs
    """
    scn_df = pd.DataFrame()
    for scn_dir, plate_id, in zip(scn_dirs_df['directory'], scn_dirs_df['plate ID']):
        metadata_path = os.path.join(scn_dir, 'multisero_output_data_metadata.xlsx')
        with pd.ExcelFile(metadata_path) as meta_file:
            antigen_df = read_antigen_info(meta_file)
            plate_info_df = read_plate_info(meta_file)
        plate_info_df['plate ID'] = plate_id
        scn_fname = [f for f in os.listdir(scn_dir) if '_analysis.xlsx' in f]
        scn_path = os.path.join(scn_dir, scn_fname[0])
        scn_df_tmp = read_scn_output(scn_path, plate_info_df)
        # Join Scienion data with plateInfo
        scn_df_tmp = pd.merge(scn_df_tmp,
                          antigen_df,
                          how='left', on=['antigen_row', 'antigen_col'])
        scn_df_tmp = pd.merge(scn_df_tmp,
                          plate_info_df,
                          how='right', on=['well_id'])
        scn_df = scn_df.append(scn_df_tmp, ignore_index=True)
    scn_df['pipeline'] = 'scienion'
    scn_df.dropna(subset=['OD'], inplace=True)
    return scn_df


def read_multisero_output_batch(ntl_dirs_df):
    """
    batch read multisero outputs
    :param dataframe ntl_dirs_df: dataframe loaded from the analysis config
    containing directories of multisero output xlsx file
    :return dataframe scn_df: combined multisero OD dataframe from multiple outputs
    """
    multisero_df = pd.DataFrame()
    for data_folder, slice_action, well_id, plate_id in \
            zip(ntl_dirs_df['directory'], ntl_dirs_df['well action'],
                ntl_dirs_df['well ID'], ntl_dirs_df['plate ID']):
        print('Load {}...'.format(data_folder))
        metadata_path = os.path.join(data_folder, 'multisero_output_data_metadata.xlsx')
        OD_path = os.path.join(data_folder, 'median_ODs.xlsx')
        int_path = os.path.join(data_folder, 'median_intensities.xlsx')
        bg_path = os.path.join(data_folder, 'median_backgrounds.xlsx')

        with pd.ExcelFile(metadata_path) as meta_file:
            antigen_df = read_antigen_info(meta_file)
            plate_info_df = read_plate_info(meta_file)
        plate_info_df['plate ID'] = plate_id
        OD_df = read_multisero_output(OD_path, antigen_df, file_type='od')
        int_df = read_multisero_output(int_path, antigen_df, file_type='int')
        bg_df = read_multisero_output(bg_path, antigen_df, file_type='bg')
        OD_df = pd.merge(OD_df,
                         antigen_df[['antigen_row', 'antigen_col', 'antigen type']],
                         how='left', on=['antigen_row', 'antigen_col'])
        OD_df = pd.merge(OD_df,
                         plate_info_df,
                         how='right', on=['well_id'])
        multisero_df_tmp = pd.merge(OD_df,
                                 int_df,
                                 how='left', on=['antigen_row', 'antigen_col', 'well_id'])
        multisero_df_tmp = pd.merge(multisero_df_tmp,
                                 bg_df,
                                 how='left', on=['antigen_row', 'antigen_col', 'well_id'])
        multisero_df_tmp['pipeline'] = 'nautilus'
        multisero_df_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        multisero_df_tmp.dropna(subset=['OD'], inplace=True)
        multisero_df_tmp = slice_df(multisero_df_tmp, slice_action, 'well_id', well_id)
        multisero_df = multisero_df.append(multisero_df_tmp, ignore_index=True)
    return multisero_df


def read_output_batch(output_dir, ntl_dirs_df, scn_dirs_df, load_report):
    """
    batch read multisero and scienion outputs
    :param output_dir: directory to save the master report
    :param dataframe ntl_dirs_df: dataframe loaded from the analysis config
    containing directories of multisero output xlsx file
    :param dataframe scn_dirs_df: dataframe loaded from the analysis config
    containing directories of scienion output xlsx file, assuming the file name always
    ends with '_analysis.xlsx'
    :return dataframe stitched_multisero_df: combined multisero and scienion OD dataframe from multiple outputs
    """
    if not load_report:
        df_list = []
        scn_df = pd.DataFrame()
        if not scn_dirs_df.empty:
            scn_df = read_scn_output_batch(scn_dirs_df)

        if not ntl_dirs_df.empty:
           multisero_df = read_multisero_output_batch(ntl_dirs_df)

        df_list.append(multisero_df)
        df_list.append(scn_df)
        #% Concatenate dataframes
        stitched_multisero_df = pd.concat(df_list)
        stitched_multisero_df.reset_index(drop=True, inplace=True)
        # remove empty xkappa-biotin spots, round off dilution
        stitched_multisero_df = stitched_multisero_df[(stitched_multisero_df['antigen'] != 'xkappa-biotin') |
                                (stitched_multisero_df['antigen type'] == 'Fiducial')]
        stitched_multisero_df['serum dilution'] = stitched_multisero_df['serum dilution'].round(7)
        stitched_multisero_df.to_csv(os.path.join(output_dir, 'master_report.csv'))
    else:
        stitched_multisero_df = pd.read_csv(os.path.join(output_dir, 'master_report.csv'), index_col=0, low_memory=False)
    return stitched_multisero_df