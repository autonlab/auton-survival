from sksurv import metrics, util
from lifelines import KaplanMeierFitter, CoxPHFitter

from sklearn.metrics import auc

import pandas as pd
import numpy as np

from tqdm import tqdm


def survival_regression_metric(metric, predictions, outcomes, times, folds=None, fold=None):

  if folds is None:
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes)
    survival_test  = survival_train
    predictions_fold = predictions

  else:
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes.iloc[folds!=fold])
    survival_test  = util.Surv.from_dataframe('event', 'time', outcomes.iloc[folds==fold])
    predictions_fold = predictions[folds==fold]

  if metric == 'brs':
    return metrics.brier_score(survival_train, survival_test, predictions_fold, times)[-1]
  elif metric == 'ibs':
    return metrics.integrated_brier_score(survival_train, survival_test, predictions_fold, times)
  elif metric == 'auc':
    return float(metrics.cumulative_dynamic_auc(survival_train, survival_test, 1-predictions_fold, times)[0])
  elif metric == 'ctd':
    return metrics.concordance_index_ipcw(survival_train, survival_test, 1-predictions_fold, tau=times)[0] 
  else:
    raise NotImplementedError()

def phenotype_purity(phenotypes, outcomes, strategy='instantaneous', folds=None, fold=None, time=None, bootstrap=None):

  np.random.seed(0)

  if folds is None: assert fold is None, "Please pass the data folds.."

  assert time is not None, "Please pass the time of evaluation!" 

  if folds is not None: 
    outcomes_train = outcomes.iloc[folds!=fold]
    outcomes_test = outcomes.iloc[folds==fold]
    phenotypes_train = phenotypes[folds!=fold]
    phenotypes_test = phenotypes[folds==fold] 
  else: 
    outcomes_train, outcomes_test = outcomes, outcomes
    phenotypes_train, phenotypes_test = phenotypes, phenotypes

  assert (time<outcomes_test['time'].max()) and (time>outcomes_test['time'].min())
  assert (time<outcomes_train['time'].max()) and (time>outcomes_train['time'].min())

  for phenotype in set(phenotypes_test):
    assert phenotype in phenotypes_train, "Testing on Phenotypes not found in the Training set!!"

  survival_curves = {}
  for phenotype in set(phenotypes_train):
    survival_curves[phenotype] = KaplanMeierFitter().fit(outcomes_train.iloc[phenotypes_train==phenotype]['time'], 
                                                         outcomes_train.iloc[phenotypes_test==phenotype]['event']) 
  
  survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
  survival_test  = util.Surv.from_dataframe('event', 'time', outcomes_test)

  n = len(survival_test)

  if strategy == 'instantaneous':

    predictions = np.zeros(len(survival_test))
    for phenotype in set(phenotypes):
      predictions[phenotypes==phenotype] = float(survival_curves[phenotype].predict(times=time, 
                                                                                    interpolate=True))
    if bootstrap is None:
      return float(metrics.brier_score(survival_train, survival_test, predictions, time)[1])
    else:
      scores = []
      for i in tqdm(range(bootstrap)):
        idx = np.random.choice(n, size=n, replace=True)
        score = float(metrics.brier_score(survival_train, survival_test[idx], predictions[idx], time)[1]) 
        scores.append(score)
      return scores

  elif strategy == 'integrated':

    times = np.unique(outcomes_test['time'])
    times = times[times<time]
    predictions = np.zeros((len(survival_test), len(times)))
    for phenotype in set(phenotypes):
      predictions[phenotypes==phenotype, :] = survival_curves[phenotype].predict(times=times,
                                                                                 interpolate=True).values

    if bootstrap is None:
      return metrics.integrated_brier_score(survival_train, survival_test, predictions, times)
    else:
      scores = []
      for i in tqdm(range(bootstrap)):
        idx = np.random.choice(n, size=n, replace=True)
        score = metrics.integrated_brier_score(survival_train, survival_test[idx], predictions[idx], times)
        scores.append(score)
      return scores

  else:
    raise NotImplementedError()


def __get_restricted_area(km_estimate, horizon):

  x = km_estimate.survival_function_.index.values 
  idx = x < horizon
  x = x[idx].tolist()
  y = km_estimate.survival_function_.KM_estimate.values[idx].tolist()
  
  y = y + [float(km_estimate.predict(horizon))]
  x = x + [horizon]

  return auc(x, y)


def _restricted_mean_diff(treated_outcomes, control_outcomes, horizon, seed=None, **kwargs):

  if seed is not None:
    treated_outcomes = treated_outcomes.sample(n=len(treated_outcomes), random_state=seed, replace=True) 
    control_outcomes = control_outcomes.sample(n=len(control_outcomes), random_state=seed, replace=True) 

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'], treated_outcomes['event'])
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'], control_outcomes['event'])

  return __get_restricted_area(treatment_survival, horizon) - __get_restricted_area(control_survival, horizon) 
 
def _survival_at_diff(treated_outcomes, control_outcomes, horizon, interpolate=True, seed=None):

  if seed is not None:
    treated_outcomes = treated_outcomes.sample(n=len(treated_outcomes), random_state=seed, replace=True) 
    control_outcomes = control_outcomes.sample(n=len(control_outcomes), random_state=seed, replace=True) 

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'], treated_outcomes['event']) 
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'], control_outcomes['event'])  

  return treatment_survival.predict(horizon, interpolate=interpolate) - control_survival.predict(horizon, interpolate=interpolate) 

def _time_to_diff(treated_outcomes, control_outcomes, horizon, interpolate=True):

  raise NotImplementedError()

  treatment_survival = KaplanMeierFitter().fit(treated_outcomes['time'], treated_outcomes['event']) 
  control_survival = KaplanMeierFitter().fit(control_outcomes['time'], control_outcomes['event'])  

def _hazard_ratio(treated_outcomes, control_outcomes, seed=None, **kwargs):

  if seed is not None:
    treated_outcomes = treated_outcomes.sample(n=len(treated_outcomes), random_state=seed, replace=True) 
    control_outcomes = control_outcomes.sample(n=len(control_outcomes), random_state=seed, replace=True) 

  treated_outcomes.insert(0, 'treated', 1.0)
  control_outcomes.insert(0, 'treated', 0.0)

  outcomes = pd.concat([treated_outcomes, control_outcomes])

  return CoxPHFitter().fit(outcomes, duration_col='time', event_col='event').hazard_ratios_['treated']


def survival_diff_metric(metric, outcomes, arms, treatment_arm, control_arm,  horizon=None, 
                           interpolate=True, bootstrap=None):

  assert metric in ['median', 'hazard_ratio' ,'restricted_mean', 'survival_at', 'time_to']
  if metric in ['restricted_mean', 'survival_at_time', 'time_to']: assert horizon is not None, "Please specify Event Horizon"

  if metric == 'hazard_ratio':
    print("WARNING: You are computing Hazard Ratios.\nMake sure you have tested the PH Assumptions.")

  # Bootstrapping ...
  if bootstrap is not None: assert isinstance(bootstrap, int), "`bootstrap` must be None or int"
  if isinstance(bootstrap, int): print("Bootstrapping... ", bootstrap, " number of times. This may take a while. Please be Patient...")

  treated_outcomes = outcomes[arms==treatment_arm]
  control_outcomes = outcomes[arms==control_arm] 

  if metric == 'survival_at': _metric = _survival_at_diff 
  elif metric == 'time_to': _metric = _time_to_diff
  elif metric == 'restricted_mean': _metric = _restricted_mean_diff 
  elif metric == 'median': _metric = _time_to_diff
  elif metric == 'hazard_ratio': _metric = _hazard_ratio
  else: raise NotImplementedError()

  if bootstrap is None: return _metric(treated_outcomes, control_outcomes, horizon=horizon, interpolate=interpolate)
  else: return [_metric(treated_outcomes, control_outcomes, horizon=horizon, interpolate=interpolate, seed=i) for i in range(bootstrap)]
