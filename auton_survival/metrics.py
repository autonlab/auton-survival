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

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import qth_survival_time
import pandas as pd
import numpy as np
from sksurv import metrics, util
from scipy.optimize import fsolve
from sklearn.metrics import auc
from tqdm import tqdm
import warnings

def treatment_effect(metric, outcomes, treatment_indicator,
                     weights=None, horizons=None, risk_levels=None,
                     interpolate=True, weights_clip=1e-2,
                     n_bootstrap=None, size_bootstrap=1.0,
                     random_seed=0):

  """Compute metrics for comparing population level survival outcomes
  across treatment arms.

  Parameters
  ----------
  metric : str
    The metric to evalute for comparing survival outcomes.
    Options include:
      - `median`
      - `tar`
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
  horizons : float or int or array of floats or ints, default=None
      Time horizon(s) at which to compute the metric.
      Must be specified for metric 'restricted_mean' and 'survival_at'.
      For 'hazard_ratio' this is ignored.
  risk_levels : float or array of floats
      The risk level (0-1) at which to compare times between treatment arms.
      Must be specified for metric 'tar'.
      Ignored for other metrics.
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
                    'survival_at', 'tar']

  if metric in ['restricted_mean', 'survival_at']:
    assert horizons is not None, "Please specify Event Horizon"
    assert risk_levels is None, "Risks must be non for 'restricted_mean' and \
'survival_at' metrics"

  if metric in ['tar']:
    assert risk_levels is not None, "Please specify risk level(s) at \
which to compare time-to-event."
    assert horizons is None, "Horizons must be none for 'tar' metric."

  if metric == 'hazard_ratio':
    warnings.warn("WARNING: You are computing Hazard Ratios.\n Make sure you have tested the PH Assumptions.")
  if (n_bootstrap is None) and (weights is not None):
    Warning("Treatment Propensity weights would be ignored, Since no boostrapping is performed."+
            "In order to incorporate IPTW weights please specify number of bootstrap iterations n_bootstrap>=1")
  # Bootstrapping ...
  if n_bootstrap is not None:
    assert isinstance(n_bootstrap, int), '`bootstrap` must be None or int'

  if isinstance(horizons, (int, float)):
    horizons = [horizons]

  if isinstance(risk_levels, (int, float)):
    risk_levels = [risk_levels]

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
  elif metric == 'tar':
    _metric = _tar
  elif metric == 'restricted_mean':
    _metric = _restricted_mean_diff
  elif metric == 'median':
    _metric = _median  # Lifelines .median_survival_time_?
  elif metric == 'hazard_ratio':
    _metric = _hazard_ratio
  else: raise NotImplementedError()

  if n_bootstrap is None:
    return _metric(treated_outcomes,
                   control_outcomes,
                   horizons=horizons,
                   risk_levels=risk_levels,
                   interpolate=interpolate,
                   treated_weights=iptw_weights[treatment_indicator],
                   control_weights=iptw_weights[~treatment_indicator])
  else:
    return [_metric(treated_outcomes,
                    control_outcomes,
                    horizons=horizons,
                    risk_levels=risk_levels,
                    interpolate=interpolate,
                    treated_weights=iptw_weights[treatment_indicator],
                    control_weights=iptw_weights[~treatment_indicator],
                    size_bootstrap=size_bootstrap,
                    random_seed=i) for i in range(n_bootstrap)]

def survival_regression_metric(metric, outcomes, predictions,
                               times, outcomes_train=None, 
                               n_bootstrap=None, random_seed=0):
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
  outcomes : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for evaluation data.
  predictions: np.array
      A numpy array of survival time predictions for the samples.
  times: np.array
      The time points at which to compute metric value(s).
  outcomes_train : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for training data.
  n_bootstrap : int, default=None
      The number of bootstrap samples to use.
      If None, bootrapping is not performed.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=0
      Controls the reproducibility random sampling for bootstrapping.
  
  Returns
  -----------
  float: The metric value for the specified metric.

  """

  if isinstance(times, (float,int)):
    times = [times]

  if outcomes_train is None:
    outcomes_train = outcomes
    warnings.warn("You are are evaluating model performance on the \
same data used to estimate the censoring distribution.")

  assert max(times) < outcomes_train.time.max(), "Times should \
be within the range of event times to avoid exterpolation."
  assert max(times) <= outcomes.time.max(), "Times \
must be within the range of event times."

  survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
  survival_test = util.Surv.from_dataframe('event', 'time', outcomes)

  if metric == 'brs':
    _metric = _brier_score
  elif metric == 'ibs':
    _metric = _integrated_brier_score
  elif metric == 'auc':
    _metric = _cumulative_dynamic_auc
  elif metric == 'ctd':
    _metric = _concordance_index_ipcw 
  else:
    raise NotImplementedError()

  if n_bootstrap is None:
    return _metric(survival_train, survival_test, predictions, times)
  else:
    return [_metric(survival_train, survival_test, predictions, times, random_seed=i) for i in range(n_bootstrap)]


def _brier_score(survival_train, survival_test, predictions, times, random_seed=None):

  idx = np.arange(len(predictions))
  if random_seed is not None:
    np.random.seed(random_seed)
    idx = np.random.choice(idx, len(predictions), replace=True)

  return metrics.brier_score(survival_train, survival_test[idx], predictions[idx], times)[-1]  

def _integrated_brier_score(survival_train, survival_test, predictions, times, random_seed=None):

  idx = np.arange(len(predictions))
  if random_seed is not None:
    np.random.seed(random_seed)
    idx = np.random.choice(idx, len(predictions), replace=True)

  return metrics.integrated_brier_score(survival_train, survival_test[idx], predictions[idx], times)

def _cumulative_dynamic_auc(survival_train, survival_test, predictions, times, random_seed=None):

  idx = np.arange(len(predictions))
  if random_seed is not None:
    np.random.seed(random_seed)
    idx = np.random.choice(idx, len(predictions), replace=True)

  return metrics.cumulative_dynamic_auc(survival_train, survival_test[idx], 1-predictions[idx], times)[0]

def _concordance_index_ipcw(survival_train, survival_test, predictions, times, random_seed=None):

  idx = np.arange(len(predictions))
  if random_seed is not None:
    np.random.seed(random_seed)
    idx = np.random.choice(idx, len(predictions), replace=True)

  vals = []
  for i in range(len(times)):
    vals.append(metrics.concordance_index_ipcw(survival_train, survival_test[idx],
                                                1-predictions[idx][:,i],
                                                tau=times[i])[0])
  return vals

def phenotype_purity(phenotypes_train, outcomes_train,
                     phenotypes_test=None, outcomes_test=None,
                     strategy='instantaneous', horizons=None,
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
  horizons : float or int or an array of floats or ints, default=None
      Event horizon(s) at which to compute the metric
  bootstrap : integer, default=None
      The number of bootstrap iterations.

  Returns
  -----------
  list:
      Columns are metric values computed for each event horizon.
      If bootstrapping, rows are bootstrap results.

  """

  np.random.seed(0)

  if (outcomes_test is None) & (phenotypes_test is not None):
    raise Exception("Specify outcomes for test set.")
  if (outcomes_test is not None) & (phenotypes_test is None):
    raise Exception("Specify phenotypes for test set.")

  assert horizons is not None, "Please specify Event Horizon"

  if isinstance(horizons, (float,int)):
    horizons = [horizons]

  if outcomes_test is None:
    phenotypes_test = phenotypes_train
    outcomes_test = outcomes_train
    warnings.warn("You are are estimating survival probabilities for \
the same dataset used to estimate the censoring distribution.")

  assert outcomes_test.time.max() >= outcomes_train.time.max(), "Test \
set times must be within the range of training set follow-up times."

  survival_curves = {}
  for phenotype in np.unique(phenotypes_train):
    survival_curves[phenotype] = KaplanMeierFitter().fit(outcomes_train.time.iloc[phenotypes_train==phenotype],
                                                         outcomes_train.event.iloc[phenotypes_train==phenotype])

  survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
  survival_test  = util.Surv.from_dataframe('event', 'time', outcomes_test)
  n = len(survival_test)

  if strategy == 'instantaneous':

    predictions = np.zeros((len(survival_test), len(horizons)))
    for phenotype in set(phenotypes_test):
      predictions[phenotypes_test==phenotype, :] = survival_curves[phenotype].predict(times=horizons,
                                                                                    interpolate=True)
    if bootstrap is None:
      return metrics.brier_score(survival_train, survival_test,
                                 predictions, horizons)[1]
    else:
      scores = []
      for i in tqdm(range(bootstrap)):
        idx = np.random.choice(n, size=n, replace=True)
        score = metrics.brier_score(survival_train, survival_test[idx],
                                    predictions[idx], horizons)[1]
        scores.append(score)
      return scores

  elif strategy == 'integrated':

    horizon_scores = []
    for horizon in horizons:
      times = np.unique(outcomes_test['time'])
      times = times[times<horizon]
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
        horizon_scores.append(np.array(score))

    if bootstrap is None:
      return horizon_scores
    else:
      horizon_scores = np.transpose(horizon_scores)
      return [np.array(i) for i in horizon_scores]

  else:
    raise NotImplementedError()

def __get_restricted_area(km_estimate, horizon):
  """Compute area under the Kaplan Meier curve (mean survival time) restricted
  by a specified time horizion.

  Parameters
  -----------
  km_estimate : Fitted Kaplan Meier estimator.
  horizon : float or int
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

def _restricted_mean_diff(treated_outcomes, control_outcomes, horizons,
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
  horizons : float or int or array of floats or ints, default=None
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

  horizon_estimates = []
  for horizon in horizons:
    treatment_estimate = __get_restricted_area(treatment_survival, horizon)
    control_estimate = __get_restricted_area(control_survival, horizon)
    horizon_estimates.append(treatment_estimate-control_estimate)

  return np.array(horizon_estimates)

def _survival_at_diff(treated_outcomes, control_outcomes, horizons,
                      treated_weights, control_weights,
                      interpolate=True, size_bootstrap=1.0,
                      random_seed=None, **kwargs):
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
  horizons : float or int or array of floats or ints, default=None
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
  estimates between the control and treatment groups at the specified time
  horizons.

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

  treatment_estimate = treatment_survival.predict(horizons, interpolate=interpolate)
  control_estimate = control_survival.predict(horizons, interpolate=interpolate)

  return np.array(treatment_estimate-control_estimate)


def _tar(treated_outcomes, control_outcomes, risk_levels,
         treated_weights, control_weights, interpolate=True,
         size_bootstrap=1.0, random_seed=None, **kwargs):
  """Time at Risk (TaR) measures time-to-event at a specified level
  of risk.

  Parameters
  -----------
  treated_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      received a specific treatment.
  control_outcomes : pd.DataFrame
      A pandas dataframe with columns 'time' and 'event' for samples that
      did not receive a specific treatment.
  risk_levels : float or array of floats
      The risk level (0-1) at which to compare times between treatment arms.
      Must be specified for metric 'tar'.
      Ignored for other metrics.
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

  treated_survival = KaplanMeierFitter().fit(treated_outcomes['time'],
                                             treated_outcomes['event'])
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'],
                                             control_outcomes['event'])

  tar_diff = []
  for risk_level in risk_levels:
      treated_tar = treated_survival.percentile(1-risk_level)
      control_tar = control_survival.percentile(1-risk_level)
      tar_diff.append(treated_tar - control_tar)

  return np.array(tar_diff)

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
  pd.Series : The exp(coefficients) (hazard ratios) of the Cox Proportional 
  Hazards model.

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
