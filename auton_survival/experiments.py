# coding=utf-8
# MIT License

# Copyright (c) 2022 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utilities to perform cross-validation."""

import numpy as np
import pandas as pd

from auton_survival.estimators import SurvivalModel, CounterfactualSurvivalModel
from auton_survival.metrics import survival_regression_metric
from auton_survival.preprocessing import Preprocessor

from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle

from tqdm import tqdm
import warnings

class SurvivalRegressionCV:
  """Universal interface to train Survival Analysis models in a cross-
  validation or nested cross-validation fashion.

  Each of the model is trained in a CV fashion over the user specified
  hyperparameter grid. The best model(s) in terms of user-specified metric
  is selected.

  Parameters
  -----------
  model : str
      A string that determines the choice of the surival regression model.
      Survival model choices include:
      - 'dsm' : Deep Survival Machines [3] model
      - 'dcph' : Deep Cox Proportional Hazards [2] model
      - 'dcm' : Deep Cox Mixtures [4] model
      - 'rsf' : Random Survival Forests [1] model
      - 'cph' : Cox Proportional Hazards [2] model
  model : str, default='dcph'
      Survival regression model name.
  folds : list, default=None
      A list of fold assignment values for each sample.
      For regular (unnested) cross-validation, folds correspond to train
      and validation set.
      For nested cross-validation, folds correspond to train and test set.
  num_folds : int, default=5
      The number of folds.
      Ignored if folds is specified.
  num_nested_folds : int, default=None
      The number of folds to use for nested cross-validation.
      If None, then regular (unnested) CV is performed.
  random_seed : int, default=0
      Controls reproducibility of results.
  hyperparam_grid : dict
      A dictionary that contains the hyperparameters for grid search.
      The keys of the dictionary are the hyperparameter names and the
      values are lists of hyperparameter values.

  References
  -----------

  [1] Hemant Ishwaran et al. Random survival forests.
  The annals of applied statistics, 2(3):841–860, 2008.
  [2] Cox, D. R. (1972). Regression models and life-tables.
  Journal of the Royal Statistical Society: Series B (Methodological).
  [3] Chirag Nagpal, Xinyu Li, and Artur Dubrawski.
  Deep survival machines: Fully parametric survival regression and
  representation learning for censored data with competing risks. 2020.
  [4] Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K. (2021c).
  Deep cox mixtures for survival regression.
  In Machine Learning for Healthcare Conference, pages 674–708. PMLR

  """

  def __init__(self, model='dcph', folds=None, num_folds=5,
               num_nested_folds=None, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.folds = folds
    self.num_folds = num_folds
    self.num_nested_folds = num_nested_folds
    self.random_seed = random_seed
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))

  def fit(self, features, outcomes, metric='ibs', horizon=None):

    r"""Fits the survival regression model to the data in a cross-
    validation or nested cross-validation fashion.

    Parameters
    -----------
    features : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.
    outcomes : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively.
    metric : str, default='ibs'
        Metric used to evaluate model performance and tune hyperparameters.
        Options include:
        - 'auc': Dynamic area under the ROC curve
        - 'brs' : Brier Score
        - 'ibs' : Integrated Brier Score
        - 'ctd' : Concordance Index
    horizon : int or float, default=None
        Event-horizon at which to evaluate model performance.
        If None, then the maximum permissible event-time from the data is used.

    Returns
    -----------
    Trained survival regression model(s).

    """
    
    self.metric = metric
    self.horizon = horizon

    if (horizon is None) & (metric not in ['auc', 'ctd', 'brs']):
      warnings.warn("Horizon is not specificed for {} metric, so the maximum \
permissible time horizon from the data is used to evaluate model \
performance".format(metric))

    if self.folds is None:
      self.folds = self._get_stratified_folds(outcomes, 'event',
                                             self.num_folds,
                                             self.random_seed)

    if self.num_nested_folds is None: 
      best_params = self._cv_select_parameters(features, outcomes, self.folds)
      model = SurvivalModel(self.model, self.random_seed, **best_params)

      return model.fit(features, outcomes)

    else:
      return self._train_nested_cv_models(features, outcomes)

  def _train_nested_cv_models(self, features, outcomes):

    """Train models in a nested CV fashion.

    Parameters
    -----------
    features : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.
    outcomes : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively.

    Returns
    -----------
    Trained survival regression models.

    """

    models = {}
    for fi, fold in enumerate(set(self.folds)):
      x_tr = features.copy().loc[self.folds!=fold]
      y_tr = outcomes.loc[self.folds!=fold]

      self.nested_folds = self._get_stratified_folds(y_tr, 'event',
                                                   self.num_nested_folds,
                                                   self.random_seed)

      best_params = self._cv_select_parameters(x_tr, y_tr, self.nested_folds)
      model = SurvivalModel(self.model, self.random_seed, **best_params)
      models[fi] = model.fit(x_tr, y_tr)

    return models

  def _cv_select_parameters(self, features, outcomes, folds):

    """Evaluate model performance on validation set in a CV fashion and
    select hyperparameters.

    Parameters
    -----------
    features : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.
    outcomes : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively.
    folds : list, default=None
        A list of fold assignment values for each sample.

    Returns
    -----------
    Dictionary of the best model hyperparameters based on the user-specified
    metric.

    """

    unique_times = np.unique(outcomes.time.values)
    if self.horizon is not None:
      unique_times = unique_times[unique_times<self.horizon]
      unique_times = list(unique_times)+[self.horizon]
    unique_times = np.array(sorted(unique_times))
    self.times = self._check_times(outcomes, unique_times, folds)

    if (self.horizon is not None) & (self.horizon not in self.times):
      warnings.warn("Specified horizon is not permissible based on \
training and validation set times. %0.2f is used as the closest permissible \
event-horizon" %(max(self.times)))

    fold_results = pd.DataFrame()
    for fold in set(folds):
      x_tr = features.copy().loc[folds!=fold]
      x_val = features.copy().loc[folds==fold]
      y_tr = outcomes.loc[folds!=fold]
      y_val = outcomes.loc[folds==fold]

      param_results = self._fit_evaluate_model(x_tr, y_tr, x_val, y_val)

      # Hyperparameter results as row items and fold results as columns.
      fold_results = pd.concat([fold_results, pd.DataFrame(param_results)],
                               axis=1)

    agg_params_results = fold_results.mean(axis=1)
    if (self.metric == 'ibs') | (self.metric == 'brs'):
      best_params_idx = agg_params_results.reset_index(drop=True).idxmin()
    elif (self.metric == 'auc') | (self.metric == 'ctd'):
      best_params_idx = agg_params_results.reset_index(drop=True).idxmax()

    return self.hyperparam_grid[best_params_idx]

  def _fit_evaluate_model(self, features_tr, outcomes_tr,
                          features_val, outcomes_val):

    """Train the model and evaluate model performance on validation set.

    Parameters
    -----------
    features_tr : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates for the training set data.
    outcomes_tr : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively
        for the training set data.
    features_val : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates for the validation set data.
    outcomes_val : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively
        for the validation set data.

    Returns
    -----------
    Model performance results in terms of the user-specific metric for each
    hyperparameter combination.

    """

    # Cannot compute metrics for evaluation set samples with time > follow-up time
    max_follow_up = outcomes_tr.time.max()
    val_sample_idx = outcomes_val.time.values < max_follow_up
    outcomes_val = outcomes_val.loc[val_sample_idx]

    if self.metric == 'ibs':
      times = self.times
    else:
      times = [self.times[-1]]

    param_results = []
    for hyper_param in tqdm(self.hyperparam_grid):
      model = SurvivalModel(self.model, self.random_seed, **hyper_param)
      model.fit(features_tr, outcomes_tr)

      predictions_val = model.predict_survival(features_val, times)
      predictions_val = predictions_val[val_sample_idx]

      metric_val = survival_regression_metric(metric=self.metric,
                                              outcomes=outcomes_val,
                                              predictions=predictions_val,
                                              outcomes_train=outcomes_tr,
                                              times=times)
      param_results.append(metric_val)

    return param_results

  def _get_stratified_folds(self, dataset, event_label, n_folds, random_seed):

    """Get cross-validation fold value for each sample.

    Parameters
    -----------
    dataset : pd.DataFrame
        A pandas datafrom with with rows corresponding to individual samples
        and columns with covariates and 'event'
    event_label : str
        String of 'event' or outcome label
    n_folds : int
        Number of folds.
    random_seed : int
        Controls reproducibility of results.

    Returns
    -----------
    auton_survival.estimators.SurvivalModel:
        The selected survival model based on lowest integrated brier score.
    """

    pos_id = dataset.loc[lambda dataset: dataset[event_label]==1].index.values
    neg_id = dataset.loc[lambda dataset: dataset[event_label]==0].index.values
    fold_assignments_pos = np.array_split(shuffle(pos_id, random_state=random_seed), n_folds)
    fold_assignments_neg = np.array_split(shuffle(neg_id, random_state=random_seed), n_folds)

    fold_assignments = []
    for i in range(n_folds):
      fold_assignments.append(np.concatenate([fold_assignments_pos[i],
                                              fold_assignments_neg[i]]))

    df_folds = pd.DataFrame()
    for fi, ids in enumerate(fold_assignments):
      each_fold = pd.DataFrame({'idx': ids, 'fold': fi})
      df_folds = pd.concat([df_folds, each_fold], axis=0)

    df_folds.sort_values(by='idx', inplace=True)
    df_folds.drop(columns='idx', inplace=True)
    df_folds = df_folds.fold.values

    return df_folds

  def _check_times(self, outcomes, times, folds):

    """Verify times are within an appropriate range for model evaluation.

    Parameters
    -----------
    outcomes : pd.DataFrame
        A pandas dataframe with columns 'time' and 'event' that contain the
        survival time and censoring status \( \delta_i = 1 \), respectively.
    times : np.array
        A numpy array of times or an event horizon.
    folds : np.array, default=None
        A numpy array of fold assignment values for each sample.

    Returns
    -----------
    auton_survival.estimators.SurvivalModel:
        The selected survival model based on lowest integrated brier score.

    """

    time_max, time_min = max(times), min(times)
    for fold in set(folds):
      time_train = outcomes.loc[folds!=fold, 'time']
      time_test = outcomes.loc[folds==fold, 'time']
      time_test = time_test[time_test<time_train.max()]

      if time_test.min() > time_min:
        time_min = time_test.min()

      if (time_test.max() < time_max)|(time_train.max() < time_max):
        if time_test.max() > time_train.max():
          time_max = max(time_test[time_test < time_train.max()])
        else:
          time_max = max(time_test[time_test < time_test.max()])

    times = times[times>=time_min]
    times = times[times<time_max]

    return times.tolist()

class CounterfactualSurvivalRegressionCV:

  r"""Universal interface to train Counterfactual Survival Analysis models in a
  Cross Validation fashion.

  Each of the model is trained in a CV fashion over the user specified
  hyperparameter grid. The best model (in terms of integrated brier score)
  is then selected.

  Parameters
  -----------
  model : str
      A string that determines the choice of the surival analysis model.
      Survival model choices include:

      - 'dsm' : Deep Survival Machines [3] model
      - 'dcph' : Deep Cox Proportional Hazards [2] model
      - 'dcm' : Deep Cox Mixtures [4] model
      - 'rsf' : Random Survival Forests [1] model
      - 'cph' : Cox Proportional Hazards [2] model
  cv_folds : int
      Number of folds in the cross validation.
  random_seed : int
      Random seed for reproducibility.
  hyperparam_grid : dict
      A dictionary that contains the hyperparameters for grid search.
      The keys of the dictionary are the hyperparameter names and the
      values are lists of hyperparameter values.

  References
  -----------

  [1] Hemant Ishwaran et al. Random survival forests.
  The annals of applied statistics, 2(3):841–860, 2008.

  [2] Cox, D. R. (1972). Regression models and life-tables.
  Journal of the Royal Statistical Society: Series B (Methodological).

  [3] Chirag Nagpal, Xinyu Li, and Artur Dubrawski.
  Deep survival machines: Fully parametric survival regression and
  representation learning for censored data with competing risks. 2020.

  [4] Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K. (2021c).
  Deep cox mixtures for survival regression.
  In Machine Learning for Healthcare Conference, pages 674–708. PMLR

  """

  _VALID_CF_METHODS = ['dsm', 'dcph', 'dcm', 'rsf', 'cph']

  def __init__(self, model, cv_folds=5, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    self.random_seed = random_seed
    self.cv_folds = cv_folds

    self.treated_experiment = SurvivalRegressionCV(model=model,
                                                num_folds=cv_folds,
                                                random_seed=random_seed,
                                                hyperparam_grid=hyperparam_grid)

    self.control_experiment = SurvivalRegressionCV(model=model,
                                                num_folds=cv_folds,
                                                random_seed=random_seed,
                                                hyperparam_grid=hyperparam_grid)

  def fit(self, features, outcomes, interventions, metric):

    r"""Fits the Survival Regression Model to the data in a Cross
    Validation fashion.

    Parameters
    -----------
    features : pandas.DataFrame
        a pandas dataframe containing the features to use as covariates.
    outcomes : pandas.DataFrame
        a pandas dataframe containing the survival outcomes. The index of the
        dataframe should be the same as the index of the features dataframe.
        Should contain a column named 'time' that contains the survival time and
        a column named 'event' that contains the censoring status.
        \( \delta_i = 1 \) if the event is observed.
    interventions: pandas.Series
        A pandas series containing the treatment status of each subject.
        \( a_i = 1 \) if the subject is `treated`, else is considered control.

    Returns
    -----------
    auton_survival.estimators.CounterfactualSurvivalModel:
        The trained counterfactual survival model.

    """

    treated_model = self.treated_experiment.fit(features.loc[interventions==1],
                                                outcomes.loc[interventions==1],
                                                metric=metric)
    control_model = self.control_experiment.fit(features.loc[interventions!=1],
                                                outcomes.loc[interventions!=1],
                                                metric=metric)

    return CounterfactualSurvivalModel(treated_model, control_model)
