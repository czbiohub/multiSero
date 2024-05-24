import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import seaborn as sns

from multiSero.plotting.plotting import roc_plot_grid
from multiSero.interpretation.report_reader import slice_df, normalize_od, read_output_batch
import multiSero.array_analyzer.extract.constants as constants


def read_config(input_dir):
    """
    Load analysis config from the input directory.
    :param str input_dir: input directory
    :return dataframe ntl_dirs_df: 'multisero output dirs' tab in the config file.
    :return dataframe scn_scn_df: 'scienion output dirs' tab in the config file.
    :return dataframe plot_setting_df: 'general plotting settings' tab in the config file.
    :return dataframe roc_param_df: 'ROC plot' tab in the config file.
    :return dataframe cat_param_df: 'categorical plot' tab in the config file.
    :return dataframe fit_param_df: 'standard curves' tab in the config file.
    """
    config = dict()
    config_path = os.path.join(input_dir, constants.METADATA_FILE)
    if constants.METADATA_FILE not in os.listdir(input_dir):
        raise IOError("analysis config file not found, aborting")
    # check that the xlsx file contains necessary worksheets
    ntl_dirs_df = scn_scn_df = roc_param_df = cat_param_df = fit_param_df = pd.DataFrame()
    with pd.ExcelFile(config_path) as config_file:
        plot_setting_df = pd.read_excel(config_file, sheet_name='general plotting settings',
                                        index_col=0, squeeze=True, usecols='A,B')
        plot_setting_df.where(plot_setting_df.notnull(), None, inplace=True)
        if 'ROC plot' in config_file.sheet_names:
            roc_param_df = pd.read_excel(config_file, sheet_name='ROC plot',
                                         index_col=0, squeeze=True, usecols='A,B')
            # replace NaN with None
            roc_param_df.where(roc_param_df.notnull(), None, inplace=True)
            if roc_param_df['serum ID'] is not None:
                roc_param_df['serum ID'] = re.split(r'\s*,\s*', roc_param_df['serum ID'])
        if 'categorical plot' in config_file.sheet_names:
            cat_param_df = pd.read_excel(config_file, sheet_name='categorical plot',
                                         index_col=0, squeeze=True, usecols='A,B')
            cat_param_df.where(cat_param_df.notnull(), None, inplace=True)
            if cat_param_df['serum ID'] is not None:
                cat_param_df['serum ID'] = re.split(r'\s*,\s*', cat_param_df['serum ID'])
        if 'standard curves' in config_file.sheet_names:
            fit_param_df = pd.read_excel(config_file, sheet_name='standard curves',
                                         index_col=0, squeeze=True, usecols='A,B')
            fit_param_df.where(fit_param_df.notnull(), None, inplace=True)
            if fit_param_df['serum ID'] is not None:
                fit_param_df['serum ID'] = re.split(r'\s*,\s*', fit_param_df['serum ID'])
        if not constants.LOAD_REPORT:
            assert ('multisero output dirs' in config_file.sheet_names) or \
            ('scienion output dirs' in config_file.sheet_names), \
            "sheet by name 'multisero output dirs' or 'scienion output dirs' are required " \
            "in analysis config file when load_report is False, aborting"
            if 'multisero output dirs' in config_file.sheet_names:
                ntl_dirs_df = pd.read_excel(config_file, sheet_name='multisero output dirs', comment='#')
                if not ntl_dirs_df.isna().loc[0, 'well ID']:
                    ntl_dirs_df['well ID'] = ntl_dirs_df['well ID'].str.split(pat=r'\s*,\s*')
            if 'scienion output dirs' in config_file.sheet_names:
                scn_scn_df = pd.read_excel(config_file, sheet_name='scienion output dirs', comment='#')
    return ntl_dirs_df, scn_scn_df, plot_setting_df, roc_param_df, cat_param_df, fit_param_df

def analyze_od(input_dir, output_dir, load_report):
    """
    Perform analysis on multisero or scienion OD outputs specified in the config files.
    Save the combined table as 'master report' in the output directory.
    :param str input_dir: Input directory
    :param str output_dir: Output directory
    :param bool load_report: If True, load the saved 'master report' in the output directory
    from the previous run. Load from the master report is much faster.
    """
    os.makedirs(output_dir, exist_ok=True)
    ntl_dirs_df, scn_scn_df, plot_setting_df, roc_param_df, cat_param_df, fit_param_df =\
        read_config(input_dir)
    stitched_multisero_df = read_output_batch(output_dir, ntl_dirs_df, scn_scn_df, load_report)
    if plot_setting_df['antigens to plot'] == 'all':
        plot_setting_df['antigens to plot'] = stitched_multisero_df['antigen'].unique()
    split_plots_by = plot_setting_df['split plots by']
    split_plots_vals = [None]
    if split_plots_by is not None:
        split_plots_vals = stitched_multisero_df[split_plots_by].unique()
    norm_antigen = plot_setting_df['normalize OD by']
    norm_group = 'plate'
    aggregate = 'mean'
    # aggregate = None
    antigen_list = [plot_setting_df['antigens to plot'], 'SARS CoV2 RBD 250', 'SARS CoV2 spike 62.5']

    suffix = ''
    df_norm = normalize_od(stitched_multisero_df.copy(), norm_antigen, group=norm_group)
    if norm_antigen is not None:
        suffix = '_'.join([suffix, norm_antigen, 'norm_per', norm_group])

    if aggregate is not None:
        df_norm = df_norm.groupby(['antigen', 'antigen type', 'serum ID', 'well_id', 'plate ID', 'sample type',
                                 'serum type', 'serum dilution', 'serum cat', 'pipeline', 'secondary ID',
                                 'secondary dilution', 'visit value'])['OD'].mean().reset_index()
        suffix = '_'.join([suffix, aggregate])

    for split_val in split_plots_vals:
        split_suffix = suffix
        if split_val is not None:
            split_suffix = '_'.join([split_suffix, split_val])
        df_norm_sub = slice_df(df_norm, 'keep', split_plots_by, split_val)
        slice_cols = [split_plots_by, 'antigen type', 'antigen']
        slice_keys = [[split_val], ['Diagnostic'], antigen_list] # might be useful to eventually select for in UI
        #slice_keys = [[split_val], ['Negative'], antigen_list]
        slice_actions = ['keep', 'keep', 'keep']
        # general slicing
        for col, action, key in zip(slice_cols, slice_actions, slice_keys):
            df_norm_sub = slice_df(df_norm_sub, action, col, key)

    # %% compute ROC curves and AUC
        if not roc_param_df.empty:
            sera_roc_list = roc_param_df['serum ID']
            slice_action = roc_param_df['serum ID action']
            # plot specific slicing
            roc_df = slice_df(df_norm_sub, slice_action, 'serum ID', sera_roc_list)
            fpr = 1 - roc_param_df['specificity']
            ci = roc_param_df['confidence interval']
            hue = roc_param_df['hue']
            # df_norm = offset_od(df_norm, offset_antigen, offset_group)
            roc_suffix = split_suffix
            if ci is not None:
                roc_suffix = '_'.join([roc_suffix, 'ci'])
            #%%
            print('{} unique positive sera'.format(len(roc_df.loc[roc_df['serum type']=='positive', 'serum ID'].unique())))
            print('{} unique negative sera'.format(len(roc_df.loc[roc_df['serum type'] == 'negative', 'serum ID'].unique())))
            roc_plot_grid(roc_df, constants.RUN_PATH, '_'.join(['ROC', roc_suffix]), 'png', ci=ci, fpr=fpr, hue=hue)
    #%% Plot categorical scatter plot for episurvey -- FIGURE 5
        if not cat_param_df.empty:
            # ms_tidy_df is created to create a tidy, concise, and contained dataframe for cat plotting (Fig 5a)
            ms_tidy_df = df_norm_sub[
                ['antigen', 'OD', 'visit value', 'serum cat', 'serum type', 'serum ID', 'serum dilution']]
            ms_tidy_df.drop(ms_tidy_df.loc[ms_tidy_df['serum dilution'] > 0.00005].index, inplace=True)
            ms_tidy_df.drop(ms_tidy_df.loc[(ms_tidy_df['serum cat'] != '[\'COVID-Vax+\']')
                                           & (ms_tidy_df['serum cat'] != '[\'COVID+Vax+\']')
                                           & (ms_tidy_df['serum cat'] != '[\'COVID+Vax-\']')].index, inplace=True)

            group = ms_tidy_df.groupby(
                'serum type')  # group data by patient ID and transfrom 'visit value' numbers into categorical values
            df2 = group.apply(lambda x: x['visit value'].unique())
            for element in df2.index:
                if (len(df2[element]) > 1):
                    if (df2[element][0][2:-2] > df2[element][1][2:-2]):
                        ms_tidy_df.loc[
                            ms_tidy_df['visit value'] == df2[element][0], 'vaccine availability'] = 'post-vax'
                        ms_tidy_df.loc[ms_tidy_df['visit value'] == df2[element][1], 'vaccine availability'] = 'pre-vax'
                    if (df2[element][0][2:-2] < df2[element][1][2:-2]):
                        ms_tidy_df.loc[ms_tidy_df['visit value'] == df2[element][0], 'vaccine availability'] = 'pre-vax'
                        ms_tidy_df.loc[
                            ms_tidy_df['visit value'] == df2[element][1], 'vaccine availability'] = 'post-vax'
                # else:
                # print(element)

            # delta_df serves a similar purpose to ms_tidy_df but contains the data for Fig 5b
            prevaxdf = ms_tidy_df.loc[ms_tidy_df['vaccine availability'] == 'pre-vax']
            prevaxdf['old OD'] = prevaxdf['OD']
            # prevaxdf.set_index(['serum type', 'antigen'], inplace=True)
            postvaxdf = ms_tidy_df.loc[ms_tidy_df['vaccine availability'] == 'post-vax']
            postvaxdf['new OD'] = postvaxdf['OD']
            # prevaxdf.set_index(['serum type', 'antigen'], inplace=True)
            delta_df = pd.merge(prevaxdf, postvaxdf, how='inner', on=['antigen', 'serum type'])
            delta_df['delta OD'] = delta_df['new OD'] - delta_df['old OD']
            # extract params from cat_param_df
            sera_cat_list = cat_param_df['serum ID']
            slice_action = cat_param_df['serum ID action']
            split_subplots_by = cat_param_df['split subplots by']
            hue = cat_param_df['hue']
            # plot specific slicing
            # cat_df = slice_df(df_norm_sub, slice_action, 'serum ID', sera_cat_list) #serum ID --> antigen
            ms_tidy_df.loc[ms_tidy_df['serum cat'] == '[\'COVID-Vax+\']', 'vaccine availability'] = 'post-vax'
            cat_df = slice_df(ms_tidy_df, slice_action, 'serum ID', sera_cat_list)
            assert not cat_df.empty, 'Plotting dataframe is empty. Please check the plotting keys'
            sns.set_context("talk")

            # FIGURE 5A
            fig_palette = ["#ee2cef", "#21e020", "#248ff9", "#e7e5e6"]
            g = sns.catplot(x="vaccine availability", y="OD", hue=hue, col=split_subplots_by, kind="swarm",
                            legend=False,
                            palette=fig_palette, order=["pre-vax", "post-vax"], dodge=True, data=cat_df, col_wrap=3)

            od_medians = cat_df.groupby(['antigen', 'serum cat', 'vaccine availability'])['OD'].median()
            median_palette = ["#a318a4", "#18a418", "#185ea4"]
            # adding median lines using boxplot functionality:
            g.map_dataframe(sns.boxplot, x="vaccine availability", y="OD", hue=hue,
                            # medianprops={'color': 'k', 'ls': '-', 'lw': 3},
                            whiskerprops={'visible': False},
                            showfliers=False, showbox=False, showcaps=False, zorder=10, order=["pre-vax", "post-vax"],
                            palette=median_palette,
                            dodge=0.55)  # change hue
            # specifying colors of median lines
            for ax in g.axes.flat:
                for line in ax.get_lines()[2:10:2]:
                    line.set_color(median_palette[0])
                for line in ax.get_lines()[10::2]:
                    line.set_color(median_palette[2])
                for line in ax.get_lines()[1::2]:
                    line.set_color(median_palette[1])
            # augmenting legend to make text more readable
            leg = plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, handleheight=0.05,
                             edgecolor="#000000")
            leg.get_texts()[0].set_text(
                f'COVID+/Vax+ median (pre-vax -- N: {od_medians[1]:.2f}, RBD: {od_medians[6]:.2f}, Spike: {od_medians[11]:.2f};'
                f' post-vax -- N: {od_medians[0]:.2f}, RBD: {od_medians[5]:.2f}, Spike: {od_medians[10]:.2f})')
            leg.get_texts()[1].set_text(
                f'COVID+/Vax- median (pre-vax -- N: {od_medians[3]:.2f}, RBD: {od_medians[8]:.2f}, Spike: {od_medians[13]:.2f};'
                f' post-vax -- N: {od_medians[2]:.2f}, RBD: {od_medians[7]:.2f}, Spike: {od_medians[12]:.2f})')
            leg.get_texts()[2].set_text(
                f'COVID-/Vax+ median (pre-vax not available; post-vax -- N: {od_medians[4]:.2f}, RBD: {od_medians[9]:.2f},'
                f' Spike: {od_medians[14]:.2f})')
            leg.get_texts()[3].set_text('COVID+/Vax+')
            leg.get_texts()[4].set_text('COVID+/Vax-')
            leg.get_texts()[5].set_text('COVID-/Vax+')
            # augment axis & tick labels
            g.set_axis_labels("vaccine availability", "OD")
            g.set_xticklabels(rotation=0, horizontalalignment='center')
            # save
            plt.savefig(os.path.join(constants.RUN_PATH, 'catplot_{}.svg'.format(split_suffix)),
                        dpi=300, bbox_inches='tight')
            if cat_param_df['zoom']:
                g.set(ylim=(-0.05, 0.4))
                plt.savefig(os.path.join(constants.RUN_PATH, 'catplot_zoom_{}.svg'.format(split_suffix)),
                            dpi=300, bbox_inches='tight')

            # FIGURE 5B -- PLOTTING DELTA OD
            # future fixes can involve making the plot fxns for 5a and 5b less redundant
            hue = 'serum cat_x'
            g = sns.catplot(x="serum cat_x", y="delta OD", hue=hue, col=split_subplots_by, kind="swarm",
                            palette=fig_palette,
                            data=delta_df, col_wrap=3, legend=False)
            g.map_dataframe(sns.boxplot, x="serum cat_x", y="delta OD",
                            # medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                            whiskerprops={'visible': False}, showfliers=False, hue=hue,
                            showbox=False, palette=median_palette,
                            showcaps=False, zorder=10,
                            dodge=False)

            dod_medians = delta_df.groupby(['antigen', 'serum cat_x'])['delta OD'].median()

            for ax in g.axes.flat:
                for line in ax.get_lines()[::2]:
                    line.set_color(median_palette[0])
                for line in ax.get_lines()[1::2]:
                    line.set_color(median_palette[1])

            leg = plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, handleheight=0.05,
                             edgecolor="#000000")  # changing the edge color to #000000 did not work but the goal is transparency
            leg.get_texts()[0].set_text(
                f'COVID+/Vax+ median (N: {dod_medians[0]:.2f}, RBD: {dod_medians[2]:.2f}, Spike: {dod_medians[4]:.2f})')
            leg.get_texts()[1].set_text(
                f'COVID+/Vax- median (N: {dod_medians[1]:.2f}, RBD: {dod_medians[3]:.2f}, Spike: {dod_medians[5]:.2f})')
            leg.get_texts()[2].set_text('COVID+/Vax+')
            leg.get_texts()[3].set_text('COVID+/Vax-')

            g.set_axis_labels("cohort", "ΔOD")
            ls = ['COVID+/Vax+', 'COVID+/Vax-']  # the same as hue_labels, a bit redundant
            g.set_xticklabels(ls, rotation=0)

            plt.savefig(os.path.join(constants.RUN_PATH, 'catplot_deltaod_{}.svg'.format(split_suffix)),
                        dpi=300, bbox_inches='tight')
            if cat_param_df['zoom']:
                g.set(ylim=(-0.05, 0.4))
                plt.savefig(os.path.join(constants.RUN_PATH, 'catplot_od_zoom_{}.svg'.format(split_suffix)),
                            dpi=300, bbox_inches='tight')
    #%% 4PL fit
        if not fit_param_df.empty:
            slice_action = fit_param_df['serum ID action']
            hue = fit_param_df['hue']
            dilution_df = slice_df(df_norm_sub, slice_action, 'serum ID', fit_param_df['serum ID'])
            split_subplots_by = fit_param_df['split subplots by']
            #total_plots(dilution_df, constants.RUN_PATH, 'fit_{}'.format(split_suffix), 'png', hue=hue,
                        #zoom=fit_param_df['zoom'], split_subplots_by=split_subplots_by, col_wrap=2)
        plt.close('all')

