"""
This script demonstrates how to build a xgboost classifiers to classify serotype of the sera using information from
multiple antigens. Two types of classifiers are currently supported (gradient boosting tree & logistic regression).
Usage:
python tran_classifier -c xgboost (or -c logistic_regression)

The script does the followings:
1. loads the example dataset from /examples/master_report.csv
2. normalizes ODs
3. tune hyperparameters of the classifier with cross-validation
4. refit the model with optimal hyperparameters
5. generate ROC plots for single antigen ODs and classifier outputs, and feature importance (xgboost model only)

Reference of the example data:
https://www.medrxiv.org/content/10.1101/2021.05.07.21249238v1.full.pdf
"""

import argparse
import logging
from matplotlib import pyplot as plt
import os
import pandas as pd
import unicodedata
import xgboost as xgb
import multiSero.array_analyzer.utils.io_utils as io_utils
from multiSero.plotting.plotting import get_roc_df, roc_plot_grid, thr_plot_grid
from multiSero.interpretation.report_reader import slice_df, normalize_od, offset_od
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
import examples
pd.options.mode.chained_assignment = None # disable chained assignment warning
LOG_NAME = 'train classifier.log'
# %%

def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace object containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--classifier',
        type=str,
        choices=['xgboost', 'logistic_regression'],
        default='xgboost',
        help="classifier to train for classifying the serotype"
             "Default: xgboost",
    )
    return parser.parse_args()

def model_fit(model, dtrain, features, target):
    """Vanilla model training without cross-validiation
    :param object model:
    :param dataframe dtrain: training data with rows being samples and columns being features
    :param list features: column names of features
    :param str target: column name of the target
    :return object model: fitted classifier object.
    :return float score: train AUC score.
    """
    model.fit(dtrain[features], dtrain[target])
    train_yhat = model.predict(dtrain[features])
    train_score = metrics.accuracy_score(dtrain[target].tolist(), train_yhat)
    return model, train_score

def xgb_fit(model, dtrain, features, target, cross_valid=True, folds=None, cv_folds=5, early_stopping_rounds=20):
    """Train xgboost model
    :param object model: XGBClassifier instance
    :param dataframe dtrain: training data with rows being samples and columns being features
    :param list features: column names of features
    :param str target: column name of the target
    :param bool cross_valid: if true, optimal number of estimators is first determined by cross-validation,
        then the model is fitted to the whole dataset with optimal parameters
    :param list or object folds: a KFold or StratifiedKFold instance or list of fold indices
    :param int cv_folds: Number of folds in CV.
    :param int early_stopping_rounds: Activates early stopping. Cross-Validation metric (average of validation
        metric computed over CV folds) needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
    :return object model: fitted classifier object.
    :return float score: test AUC score if "cross_valid=True" otherwise train AUC score is returned.
    """
    logger = logging.getLogger(LOG_NAME)
    if cross_valid:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[features].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'],
                          folds=folds, nfold=cv_folds, verbose_eval=True,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, stratified=True)
        model.set_params(n_estimators=cvresult.shape[0])
        test_score = cvresult['test-auc-mean'].iloc[-1]
        logger.info('# of estimators: %d' % cvresult.shape[0])
    # retrain the model with the whole dataset (no splitting)
    model, train_score = model_fit(model, dtrain, features, target)
    if cross_valid:
        score = test_score
        logger.info("AUC Score (Test): %f" % score)
    else:
        score = train_score
        logger.info("Accuracy : %.4g" % score)
    return model, score


def tune_cls_para(model, train, features, target, param_test, cross_valid=None, n_jobs=8):
    """Tune classifier parameters with sklearn GridSearchCV
    :param object model: estimator object
    :param dataframe train: training data with rows being samples and columns being features
    :param list features: column names of features
    :param str target: column name of the target
    :param dict or list of dictionaries param_test: Dictionary with parameters names (string)
        as keys and lists of parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    :param int, cross-validation generator or an iterable cross_valid: cross-validation splitting strategy. See "cv" argument
        in GridSearchCV for more info.
    :param int or None n_jobs: Number of jobs to run in parallel. "None" means 1 and "-1" means using all processors.
        If 1 is given, no parallel computing code is used at all, which is useful for debugging.
    :return object model: estimator object with optimal parameters.
    """
    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_test,
                           scoring='roc_auc',
                           n_jobs=n_jobs,
                           iid=False,
                           verbose=3,
                           cv=cross_valid)
    gsearch.fit(train[features], train[target])
    model.set_params(**gsearch.best_params_)
    means = gsearch.cv_results_['mean_test_score']
    stds = gsearch.cv_results_['std_test_score']
    logger = logging.getLogger(LOG_NAME)
    for mean, std, params in zip(means, stds, gsearch.cv_results_['params']):
        logger.info("%0.4f (+/-%0.04f) for %r"
              % (mean, std * 2, params))
    logger.info('best CV parameters:')
    logger.info(gsearch.best_params_)
    logger.info('best CV score: %f' % gsearch.best_score_)
    return model


def plot_xgb_fscore(model, output_dir, output_fname):
    """plot xgboost feature importance
    :param object model: XGBClassifier instance
    :param str output_dir: output directory to save the plot
    :param output_fname: output file name for the plot
    """
    fig = plt.figure()
    fig.set_size_inches((6, 6))
    xgb.plot_importance(model, importance_type='gain')
    plt.savefig(os.path.join(output_dir, ''.join([output_fname, '.jpg'])),
                dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    """
    :param object args: namespace object containing the arguments passed
    """
    clf_type = args.classifier
    # %% set file paths and load OD table
    data_dir = os.path.dirname(examples.__file__)
    output_dir = os.path.join(data_dir, 'classifier_outputs')
    os.makedirs(output_dir, exist_ok=True)
    logger = io_utils.make_logger(
        log_dir=output_dir,
        logger_name=LOG_NAME,
    )
    stitched_multisero_df = pd.read_csv(os.path.join(data_dir, 'master_report.csv'), index_col=0, low_memory=False)
    stitched_multisero_df['serum ID'] = stitched_multisero_df['serum ID'].apply(
        lambda x: unicodedata.normalize('NFKC', x)).str.strip()
    # serum ID to exclude from computing ROC
    sera_roc_list = ['Pool', 'mab', 'Blank', 'CR3022']
    df_norm = stitched_multisero_df.copy()
    norm_antigen = 'xIgG Fc'
    offset_antigen = None
    norm_group = 'plate'
    offset_group = 'well'
    suffix = '_'.join([norm_antigen, 'norm_per_plate'])
    pipeline = 'nautilus'
    suffix = pipeline
    if norm_antigen is not None:
        suffix = '_'.join([pipeline, norm_antigen, 'norm_per', norm_group])
    # %% slice OD table
    slice_cols = ['pipeline', 'serum ID', 'antigen']
    slice_keys = [[pipeline], sera_roc_list, ['xkappa-biotin']]
    slice_actions = ['keep', 'drop', 'drop']
    for col, action, key in zip(slice_cols, slice_actions, slice_keys):
        df_norm = slice_df(df_norm, action, col, key)
    # %% Normalize OD and transform the table to wide format for model training
    df_norm = normalize_od(df_norm, norm_antigen, norm_group)
    df_norm = offset_od(df_norm, offset_antigen, offset_group)
    df_norm['antigen_row'] = df_norm['antigen_row'].map(str)
    df_norm['antigen_col'] = df_norm['antigen_col'].map(str)
    multisero_df_pivot = df_norm.copy()
    multisero_df_pivot = pd.pivot_table(multisero_df_pivot,
                                     values='OD',
                                     index=['plate ID', 'well_id', 'serum ID',
                                            'serum type', 'serum dilution', 'secondary ID',
                                            'secondary dilution', 'pipeline'],
                                     columns=['antigen', 'antigen_row', 'antigen_col'])
    multisero_df_pivot.columns = ["_".join(cols) for cols in multisero_df_pivot.columns]
    features = multisero_df_pivot.columns.tolist()
    multisero_df_pivot.dropna(inplace=True)
    multisero_df_pivot.reset_index(inplace=True)
    # "positive" = 1, "negative" = 0
    multisero_df_pivot['target'] = (multisero_df_pivot['serum type'] == 'positive')
    # %% Split the dataset into train and test sets
    rand_seed = 0
    gss = GroupShuffleSplit(test_size=.4, n_splits=2, random_state=rand_seed)
    (train_ids, test_ids), _ = gss.split(multisero_df_pivot, groups=multisero_df_pivot['serum ID'])
    train = multisero_df_pivot.iloc[train_ids]
    test = multisero_df_pivot.iloc[test_ids]

    # %% set up CV folds by serum ID
    gkf = GroupKFold(n_splits=4)
    folds = []
    for fold in gkf.split(train, groups=train['serum ID']):
        folds.append(fold)
    #%% Initiate classifier instance
    if clf_type == 'xgboost':
        clf = xgb.sklearn.XGBClassifier(
                        learning_rate=0.01,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        nthread=8,
                        scale_pos_weight=1,
                        seed=0)
        #%% Tune xgboost parameter
        param_tests = [{
            'max_depth': range(1, 8, 1),
            'min_child_weight': range(1, 8, 1)
        },
            {
                'gamma': [i / 10.0 for i in range(0, 5)]
            },
            {
                'subsample': [i / 10.0 for i in range(6, 10)],
                'colsample_bytree': [i / 10.0 for i in range(6, 10)]
            },
            {
                'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]
            }
        ]
        for param_test in param_tests:
            clf = tune_cls_para(clf, train, features, target='target', param_test=param_test, cross_valid=folds, n_jobs=-1)

        #%% retrain the classifier with optimal parameters from tun_cls_para with lower learning rate and more steps
        param = {'learning_rate': 0.001,
                 'n_estimators': 10000}
        clf.set_params(**param)
        clf, score = xgb_fit(clf, train, features, target='target', folds=folds, early_stopping_rounds=10000)
        plot_xgb_fscore(clf, output_dir=output_dir, output_fname='xgb_feature_importance')

    elif clf_type == 'logistic_regression':
        clf = LogisticRegressionCV(
            Cs=10,
            intercept_scaling=1,
            max_iter=5000,
            random_state=rand_seed,
            solver='saga',
            dual=False,
            fit_intercept=True,
            penalty='l2',
            tol=0.0001,
            cv=folds,
            verbose=0)
        clf, score = model_fit(clf, train, features, target='target')
    else:
        raise ValueError('Classifier type {} is not supported.'.format(clf_type))
    #%% Model prediction
    for df in [train, test]:
        df.loc[:, 'OD'] = clf.predict_proba(df[features])[:, 1]
        df.loc[:, 'antigen'] = 'combined'
        df.loc[:, 'antigen type'] = 'Diagnostic'
    # %% transform the dataframe back to long format for plotting
    test_keys = test.drop(features + ['target'], axis=1)
    antigen_list = ['SARS CoV2 N 50', 'SARS CoV2 RBD 250', 'SARS CoV2 spike 62.5']
    suffix = '_'.join([pipeline, norm_antigen, 'norm_per_plate', 'mean_rands', str(rand_seed), 'low_c', 'ci'])
    test_keys = test_keys[['plate ID', 'well_id']].drop_duplicates()
    roc_df = df_norm.copy()
    slice_cols = ['serum ID', 'antigen type', 'antigen']
    slice_keys = [sera_roc_list, ['Diagnostic'], antigen_list]
    slice_actions = ['drop', 'keep', 'keep']
    fpr = 0.05
    ci = 95
    hue = 'pipeline'
    for col, action, key in zip(slice_cols, slice_actions, slice_keys):
        roc_df = slice_df(roc_df, action, col, key)
    roc_df = pd.merge(test_keys, roc_df, how='left', on=['plate ID', 'well_id'])
    # %% compute ROC curves and AUC
    roc_df = roc_df.groupby(['antigen', 'serum ID', 'well_id', 'plate ID',
                             'serum type', 'serum dilution', 'pipeline', 'secondary ID',
                             'secondary dilution'])['OD'].mean()
    roc_df = roc_df.reset_index()
    roc_df = pd.concat([test, roc_df])
    _ = roc_plot_grid(roc_df, output_dir, '_'.join(['ROC', clf_type, suffix]), 'pdf', col_wrap=4, ci=ci, fpr=fpr, hue=hue)
    # %% plot FPR & TPR v.s. threshold
    # threshold plot currently doesn't support CI
    ci = None
    roc_df = get_roc_df(roc_df, ci=ci)
    roc_df = roc_df.melt(id_vars=['antigen',
                                  'secondary ID',
                                  'secondary dilution',
                                  'pipeline',
                                  'threshold',
                                  'AUC'],
                         var_name='category',
                         value_name='rate'
                         )
    thr_plot_grid(roc_df, output_dir, '_'.join(['ROC_thr', clf_type, suffix]), 'pdf', col_wrap=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
