import itertools
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
import warnings
import itertools
import re
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy import optimize as optimization
from sklearn.metrics import roc_auc_score
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import resample


def fourPL(x, A, B, C, D):
    """4 parameter logistic function"""
    return ((A-D)/(1.0+((x/C)**(B))) + D)

def fit2df(df, model, serum_group='serum ID'):
    """fit model to x, y data in dataframe.
    Return a dataframe with fit x, y for plotting
    """
    sera = df[serum_group].unique()
    antigens = df['antigen'].unique()
    secondaries = df['secondary ID'].unique()
    plate_ids = df['plate ID'].unique()

    # keys = itertools.product(sera, antigens, secondaries, plate_ids, prnt)
    keys = itertools.product(sera, antigens, secondaries, plate_ids)
    df_fit = pd.DataFrame(columns=df.columns)
    # for serum, antigen, secondary, plate_id, prnt in keys:
    for serum, antigen, secondary, plate_id in keys:
        print('Fitting {}, {}...'.format(serum, antigen))
        prnt_sub = df[df['serum ID'] == serum]
        prnt = prnt_sub['PRNT'].unique()
        sec_dilu_df = df[(df[serum_group] == serum) &
                         (df['antigen'] == antigen) &
                         (df['secondary ID'] == secondary) &
                         (df['plate ID'] == plate_id) &
                         (df['PRNT'] == prnt[0])]
        sec_dilutions = sec_dilu_df['secondary dilution'].unique()
        for sec_dilution in sec_dilutions:
            sub_df = sec_dilu_df[(sec_dilu_df['secondary dilution'] == sec_dilution)].reset_index(drop=True)
            df_fit_temp = pd.DataFrame()
            guess = [0, 1, 5e-4, 1]
            xdata = sub_df['serum dilution'].to_numpy()
            ydata = sub_df['OD'].to_numpy()
            ydata = ydata[xdata > 0]
            xdata = xdata[xdata > 0]  # concentration has to be positive
            params, params_covariance = optimization.curve_fit(model, xdata, ydata, guess, bounds=(0, np.inf), maxfev=1e5)
            x_input = np.logspace(np.log10(np.min(xdata)), np.log10(np.max(xdata)), 50)
            #y_fit = fivePL(x_input,*params)
            y_fit = fourPL(x_input, *params)

            df_fit_temp['serum dilution'] = x_input
            df_fit_temp['OD'] = y_fit
            df_fit_temp['b'] = params[1]
            df_fit_temp['c'] = params[2]
            df_fit_temp['d'] = params[3]
            sub_df_expand = pd.concat(
                [sub_df.loc[[0], [serum_group,
                                  'antigen',
                                 'serum type',
                                 'serum cat',
                                 'secondary ID',
                                 'secondary dilution',
                                 'pipeline',
                                  'PRNT',
                                  'plate ID']]] * len(df_fit_temp.index), axis=0).reset_index(drop=True)
            df_fit_temp = pd.concat([df_fit_temp, sub_df_expand], axis=1)
            df_fit = df_fit.append(df_fit_temp)
    print('4PL fitting finished')
    return df_fit

def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])

    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] * 1.01, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds



def roc_ci(df, ci):
    """Helper function to compute mean and confidence intervals
    from the bootstrapped distribution"""
    tpr_mean = df['tpr'].mean()
    cis = sns.utils.ci(df['tpr'], ci).tolist()
    return pd.Series([tpr_mean] + cis, ['True positive rate', 'ci_low', 'ci_high'])

def roc_from_df(df, ci=None):
    """
    Helper function to compute ROC curves using pandas.groupby(). Confidence intervals
    are computed using bootstrapping with stratified resampling
    :param dataframe df: dataframe containing serum OD info
    :param int or None ci: Confidence interval of the ROC curves in the unit of percent
    (95 would be 95%). If None, confidence intervals are not computed.
    :return dataframe rate_df: dataframe contains ROC curves for each condition
    """
    aucs = []
    n_btstp = 1000
    s = {}
    fprs = []
    tprs = []
    thrs = []
    y_test = df['serum type'] == 'positive'
    y_prob = df['OD']
    s['False positive rate'], s['True positive rate'], s['threshold'] = \
        roc_curve(y_test, y_prob, pos_label=1, drop_intermediate=False)
    try:
        s['AUC'] = [roc_auc_score(y_test, y_prob)] * len(s['False positive rate'])
    except ValueError as err:
        print('antigen {} only has {} serum type. {}'.
              format(df['antigen'].unique()[0], y_test.unique()[0], err))
        s['AUC'] = [np.nan] * len(s['False positive rate'])
    if ci is None:
        return pd.Series(s)
    else:
        for i in range(n_btstp):
            df_rsmpl = resample(df, n_samples=len(df), stratify=df['serum type'])
            y_test = df_rsmpl['serum type'] == 'positive'
            y_prob = df_rsmpl['OD']
            fpr_tmp, tpr_tmp, thr_tmp = \
                roc_curve(y_test, y_prob, pos_label=1, drop_intermediate=False)
            fprs += fpr_tmp.tolist()
            tprs += tpr_tmp.tolist()
            thrs += thr_tmp.tolist()
            try:
                aucs.append(roc_auc_score(y_test, y_prob))
            except ValueError as err:
                aucs.append(np.nan)
        rate_df = pd.DataFrame({'False positive rate': fprs, 'tpr': tprs})
        rate_df = rate_df.groupby('False positive rate').apply(
            lambda x: roc_ci(x, ci)).reset_index()
        if len(rate_df) > 0:
            rate_df = pd.concat([pd.DataFrame(data=np.zeros((1, 4)), columns=rate_df.columns), rate_df]) # add the origin corresponding to maximum threshold
        auc_low, auc_high = sns.utils.ci(aucs, ci)
        rate_df['auc_ci_low'] = auc_low
        rate_df['auc_ci_high'] = auc_high
        return rate_df

def get_roc_df(df, ci=None):
    """
    Generate ROC curves for serum samples
    :param dataframe df: dataframe containing serum OD info
    :param int or None ci: Confidence interval of the ROC curves in the unit of percent
    (95 would be 95%). If None, confidence intervals are not computed.
    :return dataframe roc_df: dataframe contains ROC curves for each condition
    """
    df = df[df['serum type'].isin(['positive', 'negative'])]
    roc_df = df[['antigen',
                 'serum type',
                 'secondary ID',
                 'secondary dilution',
                 'OD',
                 'pipeline']]
    roc_df = roc_df.groupby(['antigen',
                             'secondary ID',
                             'secondary dilution',
                             'pipeline']).apply(lambda x: roc_from_df(x, ci))
    # roc_df = roc_df.reset_index()
    roc_df = roc_df.apply(pd.Series.explode).astype(float).reset_index()
    roc_df.dropna(inplace=True)
    return roc_df

def roc_plot(x, y, **kwargs):
    """ helper function for making ROC plots with
    seaborn FacetGrid
    """

    df = kwargs.pop('data')
    ci = kwargs.pop('ci')
    fpr = kwargs.pop('fpr')
    ax = plt.gca()
    df.plot(x=x, y=y, ax=ax, legend=False)
    tpr = np.interp(fpr, df['False positive rate'], df['True positive rate'])
    ax.plot([fpr, fpr], [0, tpr], linewidth=1, color='g', linestyle='--', alpha=1)
    ax.plot([-0.05, fpr], [tpr, tpr], linewidth=1, color='g', linestyle='--', alpha=1)

    if ci is not None:
        ax.fill_between(df[x], df['ci_low'], df['ci_high'], alpha=0.2)
        auc_low = df['auc_ci_low'].unique()[0]
        auc_high = df['auc_ci_high'].unique()[0]
        tpr_low = np.interp(fpr, df['False positive rate'], df['ci_low'])
        tpr_high = np.interp(fpr, df['False positive rate'], df['ci_high'])
        ax.text(0.4, 0.38, 'AUC={:.3f}-{:.3f}'.format(auc_low, auc_high), fontsize=12)
        ax.text(fpr + 0.1, tpr - 0.0, 'sensitivity={:.3f}-{:.3f}\nspecificity={:.3f}'.
                format(tpr_low, tpr_high, 1 - fpr),
                fontsize=12, color='g')  # add text
    else:
        auc = df['AUC'].unique()[0]
        ax.text(0.6, 0.45, 'AUC={:.3f}'.format(auc), fontsize=12)
        ax.text(fpr + 0.05, tpr - 0.2, 'sensitivity={:.3f}\nspecificity={:.3f}'.format(tpr, 1 - fpr),
                fontsize=12, color='g')  # add text

def roc_plot_grid(df, fig_path, fig_name, ext='png', hue=None,
                  col_wrap=2, ci=95, tpr=None, fpr=None):
    """
    Generate ROC plots for each antigen
    :param dataframe df: dataframe containing serum OD info
    :param str fig_path: dir to save the plots
    :param str fig_name: name of the figure file
    :param str ext: figure file extension
    :param str hue: attribute to be plotted with different colors
    :param int col_wrap: number of columns in the facetgrid
    :param int ci: Confidence interval of the ROC curves in the unit of percent
    (95 would be 95%). If None, confidence intervals are not computed.
    :param float tpr: True positive rate at which the false positive rate is shown on the curve
    :param float fpr: False positive rate at which the true positive rate is shown on the curve
    :return:
    """
    assert tpr is None or fpr is None, \
        'Specify either true positive rate or false positive rate, not both.'
    # Plot ROC curves
    antigens = natsorted(df['antigen'].unique())
    #antigens = ['DENV2 50 RVP', 'DENV2 50 VLP', 'DENV2 100 RVP', 'DENV2 250 NS1']##
    sns.set_context("notebook")
    assert not df.empty, 'Plotting dataframe is empty. Please check the plotting keys'
    palette = sns.color_palette(n_colors=len(df[hue].unique()))
    print('Computing ROC curves...')
    roc_df = get_roc_df(df, ci=ci)
    g = sns.FacetGrid(roc_df, hue=hue, col="antigen", col_order=antigens, col_wrap=col_wrap, aspect=1,
                      xlim=(-0.05, 1), ylim=(0, 1.05))
                      # hue_kws={'linestyle': ['-', '--', '-.', ':']})
    g = (g.map_dataframe(roc_plot, 'False positive rate', 'True positive rate', ci=ci, fpr=fpr))
    plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])),
                             dpi=300, bbox_inches='tight')
    plt.close()
    return roc_df

def thr_plot_grid(roc_df, fig_path, fig_name, ext, col_wrap=3):
    """
    Generate ROC plots with thresholds for each antigen
    :param roc_df:
    :param fig_path:
    :param fig_name:
    :param ext:
    :param col_wrap:
    :return:
    """
    # Plot ROC curves
    hue = 'category'
    antigens = natsorted(roc_df['antigen'].unique())
    sns.set_context("notebook")
    assert not roc_df.empty, 'Plotting dataframe is empty. Please check the plotting keys'
    palette = sns.color_palette(n_colors=len(roc_df[hue].unique()))
    print('plotting ROC curves...')
    g = sns.FacetGrid(roc_df, hue=hue, col="antigen", col_order=antigens, col_wrap=col_wrap, aspect=1,
                      hue_kws={'linestyle': ['-', '--', '-.', ':']})
    g = (g.map(plt.plot, 'threshold', 'rate').add_legend())
    # g = (g.map_dataframe(roc_plot, 'False positive rate', 'True positive rate', y2='threshold'))
    plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])),
                             dpi=300, bbox_inches='tight')
    plt.close()


def scatter_plot(df,
                 x_col,
                 y_col,
                 title,
                 output_path,
                 output_fname,
                 xlim=None,
                 ylim=None,
                 alpha=1):
    """x, y scatter_plot"""
    diff_df = df[y_col] - df[x_col]
    me = diff_df.mean()
    mae = diff_df.abs().mean()
    fig = plt.figure()
    fig.set_size_inches((6, 6))
    ax = sns.scatterplot(x=x_col, y=y_col, data=df, alpha=alpha)
    plt.title(title)
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_xlim()
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    xfit = np.linspace(xlim[0], xlim[1], 2)
    plt.plot(xfit, xfit, linewidth=5, color='k', linestyle='--', alpha=0.5)
    ax.text(0.7 * xlim[1], 0.15 * ylim[1], 'Bias={:.3f}'.format(me), fontsize=16)
    ax.text(0.7 * xlim[1], 0.1 * ylim[1], 'Noise={:.3f}'.format(mae), fontsize=16)
    plt.savefig(os.path.join(output_path, ''.join([output_fname, '.jpg'])),
                dpi=300, bbox_inches='tight')
    plt.close()


def joint_plot(df_ori,
            x_col,
            y_col,
            hue,
            title,
            output_path,
            output_fname,
            bw='scott',
            n_levels=60,
            xlim=None,
            ylim=None,
            ):
    """ Join distribution plot"""

    # g = sns.JointGrid(x=x_col, y=y_col, data=df)
    #                   # xlim=(0, 50), ylim=(0, 8))
    # g = g.plot_joint(sns.kdeplot, cmap="Purples_d")
    # g = g.plot_marginals(sns.kdeplot, color="m", shade=True)
    df = df_ori.dropna(subset=[x_col, y_col])
    diff_df = df[y_col] - df[x_col]
    me = diff_df.mean()
    mae = diff_df.abs().mean()
    # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
    cmap = 'Blues'
    fig = plt.figure()
    fig.set_size_inches((9, 9))
    g = sns.JointGrid(x_col, y_col, df,
                xlim=xlim, ylim=ylim)
    hue_vals = []
    for hue_val, hue_df in df.groupby(hue):
        hue_vals.append(hue_val)
        sns.kdeplot(hue_df[x_col], ax=g.ax_marg_x, legend=False, bw=bw)
        sns.kdeplot(hue_df[y_col], ax=g.ax_marg_y, vertical=True, legend=False, bw=bw)
        sns.kdeplot(hue_df[x_col], hue_df[y_col], ax=g.ax_joint,
                     legend=True, gridsize=400, bw=bw, n_levels=n_levels, shade=True, cmap=cmap)
    xfit = np.linspace(xlim[0], xlim[1], 2)
    g.ax_joint.plot(xfit, xfit, linewidth=5, color='k', linestyle='--', alpha=0.5)
    g.ax_joint.text(0.7 * xlim[1], 0.15 * ylim[1], 'Bias={:.3f}'.format(me), fontsize=16)  # add text
    g.ax_joint.text(0.7 * xlim[1], 0.1 * ylim[1], 'Noise={:.3f}'.format(mae), fontsize=16)  # add text
    plt.title(title)
    plt.legend(hue_vals, loc='upper left')
    plt.savefig(os.path.join(output_path, ''.join([output_fname, '.jpg'])),
                dpi=300, bbox_inches='tight')

def standard_curve_plot(dilution_df, fig_path, fig_name, ext, hue=None,
                        zoom=False, split_subplots_by='antigen', col_wrap=2):
    """
    Plot standard curves for ELISA
    :param dataframe dilution_df: dataframe containing serum OD with serial diluition
    :param str fig_path: dir to save the plots
    :param str fig_name: name of the figure file
    :param str ext: figure file extension
    :param str hue: attribute to be plotted with different colors
    :param int col_wrap: number of columns in the facetgrid
    :param bool zoom: If true, output zoom-in of the low OD region
    """
    dilution_df_fit = dilution_df.copy()
    dilution_df_fit = fit2df(dilution_df_fit, fourPL)
    hue_list = dilution_df[hue].unique()
    # %% plot standard curves
    # hue_fit_list = [' '.join([x, 'fit']) for x in hue_list]
    split_subplots_vals = dilution_df[split_subplots_by].unique()
    # markers = 'o'
    mks = itertools.cycle(['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'])
    markers = [next(mks) for i in hue_list]
    style = 'serum type'
    # style = hue
    assert not dilution_df.empty, 'Plotting dataframe is empty. Please check the plotting keys'
    palette = sns.color_palette(n_colors=len(dilution_df[hue].unique()))
    print('plotting standard curves...')
    sns.set_context("talk")
    g = sns.lmplot(x="serum dilution", y="OD",
                   hue=hue, hue_order=hue_list, col=split_subplots_by, ci='sd', palette=palette, markers=markers,
                   data=dilution_df, col_wrap=col_wrap, fit_reg=False, x_estimator=np.mean)

    for val, ax in zip(split_subplots_vals, g.axes.flat):
        df_fit = dilution_df_fit[(dilution_df_fit[split_subplots_by] == val)]
        palette = sns.color_palette(n_colors=len(df_fit[hue].unique()))
        sns.set(font_scale=2)
        sns.lineplot(x="serum dilution", y="OD", hue=hue, hue_order=hue_list, data=df_fit,
                     style=style, palette=palette,
                     ax=ax, legend=False)
        ax.set(xscale="log")
    plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])), dpi=300, bbox_inches='tight')

    if zoom:
        for val, ax in zip(split_subplots_vals, g.axes.flat):
            ax.set(ylim=[-0.05, 0.4])
        fig_name += '_zoom'
        plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])), dpi=300, bbox_inches='tight')

def plot_heatmap(hmap,fig_path,ext,spot,type,vmin,vmax,x,y):
    """
    Generates heatmap of IC50 or slope at IC50 for various antigens
    :param dataframe hmap: DataFrame of slope, upper limit, and/or IC50 values per antigen per serum ID
    :param str fig_path: dir to save the plots
    :param str ext: figure file extension
    :param str spot: what type of antigen is being evaluated (ie: EDIII, NS1, etc)
    :param float vmin: minimum for cmap range
    :param float vmax: maximum for cmap range
    :param int x: width of heatmap plot
    :param int y: height of heatmap plot
    """
    #dftt = hmap.filter(like=spot)
    dftt = hmap
    df_t = dftt.transpose() #comment out if in serum ID mode
    fig, ax = plt.subplots(figsize=(x, y))
    sns.set_context('poster')
    sns.set(font_scale=4)
    sns.heatmap(df_t, annot=True, ax=ax, vmin=vmin, vmax=vmax)
    #sns.heatmap(df_t, annot=True, ax=ax, vmin=vmin, vmax=vmax)
    plt.xticks(rotation=45,fontsize=28)
    plt.yticks(rotation=0,fontsize=28)
    plt.title(f'{type} Values per Antigen per Serum ID ({spot})', fontsize=20)
    plt.savefig(os.path.join(fig_path, '.'.join([f'{spot}_{type}_map', ext])), dpi=300, bbox_inches='tight')

def delta_ic50(ic_df,prnt_val,fig_path,ext,spot,hue):
    """
    Generates a heatmap to look at the ratiometric difference between IC50 of various antigens
    :param dataframe ic_df: DataFrame IC50 values per antigen per serum ID
    :param str fig_path: dir to save the plots
    :param str ext: figure file extension
    :param str spot: what type of antigen is being evaluated (ie: EDIII, NS1, etc)
    """
    if hue =='antigen':
        [i, j, m, p] = [-6, -1, 0, 5]
    else:
        [i, j, m, p] = [0, 5, -6, -1] # hue == 'serum ID' mode
    #finding match
    new_df = pd.DataFrame()
    for col in ic_df.T:
        name = col  # name = serum type
        b = col[i:j] #presumes serum ID labeled with serotype in parantheses, ie: serum ID ABC135 (DENV1)
        for idx in ic_df.T.index:
            if idx[m:p] == b:
                match = ic_df.T[name].loc[idx]
                new_df[name] = np.abs(ic_df.T[name] / match)
    #actual plotting
    fig, (ax,ax2) = plt.subplots(figsize=(45,15) ,ncols=2)
    gs = gridspec.GridSpec(1, 2, width_ratios=[15, 1])
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    sns.set_context('poster')
    sns.set(font_scale=4)
    sns.heatmap(new_df, annot=True, ax=ax, vmin=0,vmax=2.5, cbar=False)
    ax.tick_params(labelrotation=45)
    ax.set_yticklabels(ic_df.T.index, rotation=0)
    sns.heatmap(prnt_val, annot=True, ax=ax2,vmin=0,vmax=2.5,yticklabels=False,cbar=False)
    ax.set(xlabel=hue)
    ax2.set(ylabel=None)
    fig.tight_layout()
    #plt.title(f'Binding Affinity Measurements per Antigen per Serum ID ({spot})', fontsize=20)
    plt.savefig(os.path.join(fig_path, '.'.join([f'deltaic{spot}map', ext])), dpi=300, bbox_inches='tight')

def plot_by_type(rvp_list,mks,dilution_df,dilution_df_fit,split_subplots_by,split_subplots_vals,fig_name,
                 fig_path,ext,hue,col_wrap,zoom=False):
    """
    For antigen evaluation: standard plots grouped by antigen type. Meant to be viewed in conjunction with heatmaps.
    :param str rvp_list: list of antigen types
    :param str mks: list of marker types for plotting
    :param dataframe dilution_df: dataframe containing serum OD with serial dilution
    :param dataframe dilution_df_fit: dataframe containing serum OD with serial dilution fitted to 4PL curve
    :param str split_subplots_by: breakdown of subplots, either by antigen or serum ID
    :param str split_subplots_vals: unique values of split_subplots_by
    :param str fig_name: name of the figure file
    :param str fig_path: dir to save the plots
    :param str ext: figure file extension
    :param str hue: attribute to be plotted with different colors
    :param int col_wrap: number of columns in the facetgrid
    :param bool zoom: If true, output zoom-in of the low OD region
    """
    markers = [next(mks) for i in rvp_list]
    style = 'serum type'
    # style = hue
    assert not dilution_df.empty, 'Plotting dataframe is empty. Please check the plotting keys'
    # palette = sns.color_palette(n_colors=len(dilution_df[hue].unique()))
    palette = sns.color_palette(n_colors=len(rvp_list))
    print('plotting standard curves...')
    g = sns.lmplot(x="serum dilution", y="OD",
                   hue=hue, hue_order=rvp_list, col=split_subplots_by, ci=None, palette=palette, markers=markers,
                   data=dilution_df, col_wrap=col_wrap, fit_reg=False, x_estimator=np.mean)

    for val, ax in zip(split_subplots_vals, g.axes.flat):
        df_fit = dilution_df_fit[(dilution_df_fit[split_subplots_by] == val)]
        # palette = sns.color_palette(n_colors=len(df_fit[hue].unique()))
        palette = sns.color_palette(n_colors=len(rvp_list))
        sns.set(font_scale=2)
        sns.lineplot(x="serum dilution", y="OD", hue=hue, hue_order=rvp_list, data=df_fit,
                     style=style, palette=palette,
                     ax=ax, legend=False)
        ax.set(xscale="log")

    plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])), dpi=300, bbox_inches='tight')

    if zoom:
        for val, ax in zip(split_subplots_vals, g.axes.flat):
            ax.set(ylim=[-0.05, 0.4])
        fig_name += '_zoom'
        plt.savefig(os.path.join(fig_path, '.'.join([fig_name, ext])), dpi=300, bbox_inches='tight')

def total_plots(dilution_df, fig_path, fig_name, ext, hue=None,
                        zoom=False, split_subplots_by='antigen', col_wrap=2):
    """
    Plot standard curves with heatmap plots for holistic antigen candidate evaluation
    :param dataframe dilution_df: dataframe containing serum OD with serial diluition
    :param str fig_path: dir to save the plots
    :param str fig_name: name of the figure file
    :param str ext: figure file extension
    :param str hue: attribute to be plotted with different colors
    :param int col_wrap: number of columns in the facetgrid
    :param bool zoom: If true, output zoom-in of the low OD region
    """
    dilution_df_fit = dilution_df.copy()
    dilution_df_fit = fit2df(dilution_df_fit,
                             fourPL)
    ic_50 = dilution_df_fit[['antigen', 'serum ID', 'c', 'b', 'd', 'PRNT', 'OD']]

    alt = ic_50.set_index('serum ID').drop_duplicates()

    # hue_list = dilution_df[hue].unique()

    omap = alt.pivot_table(index=hue, columns=split_subplots_by, values='OD')
    hmap = alt.pivot_table(index=hue, columns=split_subplots_by, values='c')
    bmap = alt.pivot_table(index=hue, columns=split_subplots_by, values='b')
    prnt = alt.pivot_table(index=hue, columns=split_subplots_by, values='PRNT')
    # dmap = alt.pivot(index=None, columns='antigen', values='d')

    lmap = np.log(hmap)
    spot_df = lmap
    prnt_val = ic_50[['serum ID', 'PRNT']].set_index('serum ID').drop_duplicates()

    slope_vmax = 3 #relevant if plotting b
    ic_vmax = 0
    y = ' '

    plot_heatmap(lmap, fig_path, ext, spot=y, type='Log of IC50', vmin=-10, vmax=ic_vmax, x=45, y=15)
    delta_ic50(spot_df, prnt_val, fig_path, ext, spot=y, hue=hue)
    standard_curve_plot(dilution_df, fig_path, fig_name, ext, hue, zoom, split_subplots_by, col_wrap)