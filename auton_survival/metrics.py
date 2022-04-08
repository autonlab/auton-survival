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

"""Tools to compute metrics used to assess survival outcomes and survival
model performance."""

from sksurv import metrics, util
from lifelines import KaplanMeierFitter, CoxPHFitter

from sklearn.metrics import auc

import pandas as pd
import numpy as np

from tqdm import tqdm

import warnings

def survival_diff_metric(metric, outcomes, treatment_indicator,
                         weights=None, horizon=None, interpolate=True,
                         weights_clip=1e-2, n_bootstrap=None,
                         size_bootstrap=1.0, random_seed=0):

  """Compute metrics for comparing population level survival outcomes
  across treatment arms.

  Parameters
  ----------
  metric : str
    The metric to evalute for comparing survival outcomes.
    Options include:
      - `median`
      - `time_to`
      - `hazard_ratio`
      - `restricted_mean`
      - `survival_at`
  outcomes : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples
      and columns 'time' and 'event'.
  treatment_indicator : np.array
      Boolean numpy array of treatment indicators. True means individual
      was assigned a specific treatment.
  weights : pd.Series, default=None
      Treatment assignment propensity scores, \( \widehat{\mathbb{P}}(A|X=x) \).
      If `None`, all weights are set to \( 0.5 \). Default is `None`.
  horizon : float
      The time horizon at which to compare the survival curves.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.
  interpolate : bool, default=True
      Whether to interpolate the survival curves.
  weights_clip : float
      Weights below this value are clipped. This is to ensure IPTW
      estimation is numerically stable.
      Large weights can result in estimator with high variance.
  n_bootstrap : int, default=None
      The number of bootstrap samples to use.
      If None, bootrapping is not performed.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=0
      Controls the reproducibility random sampling for bootstrapping.

  Returns
  ----------
  float or list: The metric value(s) for the specified metric.

  """

  assert metric in ['median', 'hazard_ratio', 'restricted_mean',
                    'survival_at', 'time_to']

  if metric in ['restricted_mean', 'survival_at', 'time_to']:
    assert horizon is not None, "Please specify Event Horizon"

  if metric == 'hazard_ratio':
    warnings.warn("WARNING: You are computing Hazard Ratios.\n Make sure you have tested the PH Assumptions.")
  if (n_bootstrap is None) and (weights is not None):
    Warning("Treatment Propensity weights would be ignored, Since no boostrapping is performed."+
            "In order to incorporate IPTW weights please specify number of bootstrap iterations n_bootstrap>=1")
  # Bootstrapping ...
  if n_bootstrap is not None:
    assert isinstance(n_bootstrap, int), '`bootstrap` must be None or int'

  if isinstance(n_bootstrap, int):
    print('Bootstrapping... ', n_bootstrap,
          ' number of times. This may take a while. Please be Patient...')

  is_treated = treatment_indicator.astype(float)
  if weights is None:
    weights = 0.5*np.ones(len(outcomes))

  weights[weights>(1.-weights_clip)] = 1-weights_clip
  weights[weights<weights_clip] = weights_clip

  iptw_weights = 1./((is_treated*weights)+((1-is_treated)*(1-weights)))

  treated_outcomes = outcomes[treatment_indicator]
  control_outcomes = outcomes[~treatment_indicator]

  if metric == 'survival_at':
    _metric = _survival_at_diff
  elif metric == 'time_to':
    _metric = _time_to_diff
  elif metric == 'restricted_mean':
    _metric = _restricted_mean_diff
  elif metric == 'median':
    _metric = _time_to_diff
  elif metric == 'hazard_ratio':
    _metric = _hazard_ratio
  else: raise NotImplementedError()

  if n_bootstrap is None:
    return _metric(treated_outcomes,
                   control_outcomes,
                   horizon=horizon,
                   interpolate=interpolate,
                   treated_weights=iptw_weights[treatment_indicator],
                   control_weights=iptw_weights[~treatment_indicator])
  else:
    return [_metric(treated_outcomes,
                    control_outcomes,
                    horizon=horizon,
                    interpolate=interpolate,
                    treated_weights=iptw_weights[treatment_indicator],
                    control_weights=iptw_weights[~treatment_indicator],
                    size_bootstrap=size_bootstrap,
                    random_seed=i) for i in range(n_bootstrap)]

def survival_regression_metric(metric, outcomes_train, outcomes_test,
                               predictions, times):
  """Compute metrics to assess survival model performance.

  Parameters
  -----------
  metric: string
      Measure used to assess the survival regression model performance.
      Options include:
      - `brs` : brier score
      - `ibs` : integrated brier score
      - `auc`: cumulative dynamic area under the curve
      - `ctd` : concordance index inverse probability of censoring
                weights (ipcw)
  predictions: np.array
      A numpy array of survival time predictions for the samples.
  outcomes_train : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for test data.
  outcomes_test : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for training data.
  times: np.array
      The time points at which to compute metric value(s).

  Returns
  -----------
  float: The metric value for the specified metric.

  """

  survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
  survival_test = util.Surv.from_dataframe('event', 'time', outcomes_test)
  predictions_test = predictions

  if metric == 'brs':
    return metrics.brier_score(survival_train, survival_test,
                               predictions_test, times)[-1]
  elif metric == 'ibs':
    return metrics.integrated_brier_score(survival_train, survival_test,
                                          predictions_test, times)
  elif metric == 'auc':
    return metrics.cumulative_dynamic_auc(survival_train, survival_test,
                                            1-predictions_test, times)[0]
  elif metric == 'ctd':
    vals = []
    for i in range(len(times)):
      vals.append(metrics.concordance_index_ipcw(survival_train, survival_test,
                                                 1-predictions_test[:,i],
                                                 tau=times[i])[0])
    return vals

  else:
    raise NotImplementedError()

def phenotype_purity(phenotypes_train, outcomes_train,
                     phenotypes_test=None, outcomes_test=None,
                     strategy='instantaneous', horizon=None,
                     bootstrap=None):
  """Compute the brier score to assess survival model performance
  for phenotypes.

  Parameters
  -----------
  phenotypes_train: np.array
      A numpy array containing an array of integers that define subgroups
      for the train set.
  outcomes_train : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for the train set.
  phenotypes_test: np.array
      A numpy array containing an array of integers that define subgroups
      for the test set.
  outcomes_test : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for the test set.
  strategy : string, default='instantaneous'
      Options include:
      - `instantaneous` : Compute the brier score.
      - `integrated` : Compute the integrated brier score.
  horizon : float or int or np.array of floats or ints, default=None
      Event horizon(s) at which to compute the metric
  bootstrap : integer, default=None
      The number of bootstrap iterations.

  Returns
  -----------
  list:
      Columns are metric values computed for each event horizon.
      If bootstrapping, rows are bootstrap results.

  """

  # CODE UPDATE: enable phenotype purity to be computed for the test set...
  # without specifying folds to determine the train/test sets when folds are inapplicable (no CV)
  np.random.seed(0)

  if (outcomes_test is None) & (phenotypes_test is not None):
    raise Exception("Specify outcomes for test set.")
  if (outcomes_test is not None) & (phenotypes_test is None):
    raise Exception("Specify phenotypes for test set.")

  assert horizon is not None, "Please specify Event Horizon"

  if isinstance(horizon, float) | isinstance(horizon, int):
    horizon = [horizon]

  if outcomes_test is None:
    phenotypes_test = phenotypes_train
    outcomes_test = outcomes_train
    warnings.warn("You are are estimating survival probabilities for \
                    the same dataset used to estimate the censoring \
                    distribution.")

  survival_curves = {}
  for phenotype in np.unique(phenotypes_train):
    survival_curves[phenotype] = KaplanMeierFitter().fit(outcomes_train.time.iloc[phenotypes_train==phenotype],
                                                         outcomes_train.event.iloc[phenotypes_train==phenotype])

  survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
  survival_test  = util.Surv.from_dataframe('event', 'time', outcomes_test)
  n = len(survival_test)

  if strategy == 'instantaneous':

    predictions = np.zeros((len(survival_test), len(horizon)))
    for phenotype in set(phenotypes_test):
      predictions[phenotypes_test==phenotype, :] = survival_curves[phenotype].predict(times=horizon,
                                                                                    interpolate=True)
    if bootstrap is None:
      return metrics.brier_score(survival_train, survival_test,
                                 predictions, horizon)[1]
    else:
      scores = []
      for i in tqdm(range(bootstrap)):
        idx = np.random.choice(n, size=n, replace=True)
        score = metrics.brier_score(survival_train, survival_test[idx],
                                    predictions[idx], horizon)[1]
        scores.append(score)
      return scores

  elif strategy == 'integrated':

    horizon_scores = []
    for time in horizon:
      times = np.unique(outcomes_test['time'])
      times = times[times<time]
      predictions = np.zeros((len(survival_test), len(times)))
      for phenotype in set(phenotypes_test):
        predictions[phenotypes_test==phenotype, :] = survival_curves[phenotype].predict(times=times,
                                                                                     interpolate=True).values
      if bootstrap is None:
        horizon_scores.append(metrics.integrated_brier_score(survival_train,
                                                             survival_test,
                                                             predictions,
                                                             times))

      else:
        score = []
        for i in tqdm(range(bootstrap)):
          idx = np.random.choice(n, size=n, replace=True)
          score.append(metrics.integrated_brier_score(survival_train,
                                                      survival_test[idx],
                                                      predictions[idx],
                                                      times))
        horizon_scores.append(score)

    if bootstrap is None:
      return np.array(horizon_scores)
    else:
      # Format scores exactly like "instantaneous" option w/ bootstrapping for consistency
      return [np.array([j[i] for j in np.array(horizon_scores)]) for i in range(bootstrap)]

  else:
    raise NotImplementedError()

def __get_restricted_area(km_estimate, horizon):
  """Compute area under the Kaplan Meier curve (mean survival time) restricted
  by a specified time horizion.

  Parameters
  -----------
  km_estimate : Fitted Kaplan Meier estimator.
  horizon : float
      The time horizon at which to compare the survival curves.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.

  Returns
  -----------
  float : Area under the Kaplan Meier curve (mean survival time).

  """

  x = km_estimate.survival_function_.index.values
  idx = x < horizon
  x = x[idx].tolist()
  y = km_estimate.survival_function_.KM_estimate.values[idx].tolist()

  y = y + [float(km_estimate.predict(horizon))]
  x = x + [horizon]

  return auc(x, y)

def _restricted_mean_diff(treated_outcomes, control_outcomes, horizon,
                          treated_weights, control_weights,
                          size_bootstrap=1.0, random_seed=None, **kwargs):
  """Compute the difference in the area under the Kaplan Meier curve
  (mean survival time) between control and treatment groups.

  Parameters
  -----------
  treated_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      received a specific treatment.
  control_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      did not receive a specific treatment.
  horizon : float
      The time horizon at which to compare the survival curves.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.
  treated_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that received a specific treatment.
  control_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that did not receive a specific treatment.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=None
      Controls the reproducibility random sampling for bootstrapping.
  kwargs : dict
      Additional arguments for the Kaplan Meier estimator??

  Returns
  -----------
  float : The difference in the area under the Kaplan Meier curve
  (mean survival time) between control and treatment groups.

  """

  if random_seed is not None:
    treated_outcomes = treated_outcomes.sample(n=int(size_bootstrap*len(treated_outcomes)),
                                               weights=treated_weights,
                                               random_state=random_seed,
                                               replace=True)
    control_outcomes = control_outcomes.sample(n=int(size_bootstrap*len(control_outcomes)),
                                               weights=control_weights,
                                               random_state=random_seed,
                                               replace=True)

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'],
                                               treated_outcomes['event'])
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'],
                                             control_outcomes['event'])

  return __get_restricted_area(treatment_survival, horizon) - __get_restricted_area(control_survival, horizon)

def _survival_at_diff(treated_outcomes, control_outcomes, horizon,
                      treated_weights, control_weights,
                      interpolate=True, size_bootstrap=1.0, random_seed=None):
  """Compute the difference in Kaplan Meier survival function estimates
  between the control and treatment groups at a specified time horizon.

  Parameters
  -----------
  treated_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      received a specific treatment.
  control_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      did not receive a specific treatment.
  horizon : float
      The time horizon at which to compare the survival curves.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.
  treated_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that received a specific treatment.
  control_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that did not receive a specific treatment.
  interpolate : bool, default=True
      Whether to interpolate the survival curves.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=None
      Controls the reproducibility random sampling for bootstrapping.

  Returns
  -----------
  pd.Series : A pandas series of the difference in Kaplan Meier survival
  estimates between the control and treatment groups at a specified time
  horizon.

  """

  if random_seed is not None:
    treated_outcomes = treated_outcomes.sample(n=int(size_bootstrap*len(treated_outcomes)),
                                               weights=treated_weights,
                                               random_state=random_seed, replace=True)
    control_outcomes = control_outcomes.sample(n=int(size_bootstrap*len(control_outcomes)),
                                               weights=control_weights,
                                               random_state=random_seed, replace=True)

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'],
                                               treated_outcomes['event'])
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'],
                                             control_outcomes['event'])

  return treatment_survival.predict(horizon, interpolate=interpolate) - control_survival.predict(horizon, interpolate=interpolate)

def _time_to_diff(treated_outcomes, control_outcomes, horizon, interpolate=True):
  """Not implemented.

  Parameters
  -----------
  treated_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      received a specific treatment.
  control_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      did not receive a specific treatment.
  horizon : float
      The time horizon at which to compare the survival curves.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.
  interpolate : bool, default=True
      Whether to interpolate the survival curves.

  """

  raise NotImplementedError()

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'],
                                               treated_outcomes['event'])
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'],
                                             control_outcomes['event'])

def _hazard_ratio(treated_outcomes, control_outcomes,
                  treated_weights, control_weights,
                  size_bootstrap=1.0, random_seed=None, **kwargs):
  """Train an instance of the Cox Proportional Hazards model and return the
  exp(coefficients) (hazard ratios) of the model.

  Parameters
  -----------
  treated_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      received a specific treatment.
  control_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      did not receive a specific treatment.
  treated_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that received a specific treatment.
  control_weights : pd.Series
      A pandas series of the inverse probability of censoring weights for
      samples that did not receive a specific treatment.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=None
      Controls the reproducibility random sampling for bootstrapping.
  kwargs : dict
      Additional arguments for the Cox proportional hazards model.
      Please include dictionary key and item pairs specified by the following
      module: lifelines.fitters.coxph_fitter.CoxPHFitters

  Returns
  -----------
  pd.Series : The exp(coefficients) (hazard ratios) of the Cox Proportional Hazards model.

  """

  if random_seed is not None:
    treated_outcomes = treated_outcomes.sample(n=int(size_bootstrap*len(treated_outcomes)),
                                               weights=treated_weights,
                                               random_state=random_seed, replace=True)
    control_outcomes = control_outcomes.sample(n=int(size_bootstrap*len(control_outcomes)),
                                               weights=control_weights,
                                               random_state=random_seed, replace=True)

  treated_outcomes.insert(0, 'treated', 1.0)
  control_outcomes.insert(0, 'treated', 0.0)

  outcomes = pd.concat([treated_outcomes, control_outcomes])

  return CoxPHFitter().fit(outcomes,
                           duration_col='time',
                           event_col='event').hazard_ratios_['treated']
