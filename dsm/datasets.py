# coding=utf-8
# Copyright 2020 Chirag Nagpal
#
# This file is part of Deep Survival Machines.

# Deep Survival Machines is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Deep Survival Machines is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Deep Survival Machines.  
# If not, see <https://www.gnu.org/licenses/>.


"""Utility functions to load standard datasets to train and evaluate the
Deep Survival Machines models.
"""


import io
import pkgutil

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def increase_censoring(e, t, p):

  uncens = np.where(e == 1)[0]
  mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
  toswitch = uncens[mask]

  e[toswitch] = 0
  t_ = t[toswitch]

  newt = []
  for t__ in t_:
    newt.append(np.random.uniform(1, t__))
  t[toswitch] = newt

  return e, t

def _load_pbc_dataset(sequential):
  """Helper function to load and preprocess the PBC dataset

  The Primary biliary cirrhosis (PBC) Dataset [1] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.

  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.


  References
  ----------
  [1] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.

  """

  data = pkgutil.get_data(__name__, 'datasets/pbc2.csv')
  data = pd.read_csv(io.BytesIO(data))

  data['histologic'] = data['histologic'].astype(str)
  dat_cat = data[['drug', 'sex', 'ascites', 'hepatomegaly',
                  'spiders', 'edema', 'histologic']]
  dat_num = data[['serBilir', 'serChol', 'albumin', 'alkaline',
                  'SGOT', 'platelets', 'prothrombin']]
  age = data['age'] + data['years']

  x1 = pd.get_dummies(dat_cat).values
  x2 = dat_num.values
  x3 = age.values.reshape(-1, 1)
  x = np.hstack([x1, x2, x3])

  time = (data['years'] - data['year']).values
  event = data['status2'].values

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
  x_ = StandardScaler().fit_transform(x)

  if not sequential:
    return x_, time, event
  else:
    x, t, e = [], [], []
    for id_ in sorted(list(set(data['id']))):
      x.append(x_[data['id'] == id_])
      t.append(time[data['id'] == id_])
      e.append(event[data['id'] == id_])
    return x, t, e

def _load_support_dataset():
  """Helper function to load and preprocess the SUPPORT dataset.

  The SUPPORT Dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].

  Please refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.

  References
  ----------
  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.

  """

  data = pkgutil.get_data(__name__, 'datasets/support2.csv')
  data = pd.read_csv(io.BytesIO(data))
  x1 = data[['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp',
             'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
             'urine', 'adlp', 'adls']]

  catfeats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
  x2 = pd.get_dummies(data[catfeats])

  x = np.concatenate([x1, x2], axis=1)
  t = data['d.time'].values
  e = data['death'].values

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
  x = StandardScaler().fit_transform(x)

  remove = ~np.isnan(t)
  return x[remove], t[remove], e[remove]


def load_dataset(dataset='SUPPORT', **kwargs):
  """Helper function to load datasets to test Survival Analysis models.

  Parameters
  ----------
  dataset: str
      The choice of dataset to load. Currently implemented is 'SUPPORT'.
  **kwargs: dict
      Dataset specific keyword arguments.

  Returns
  ----------
  tuple: (np.ndarray, np.ndarray, np.ndarray)
      A tuple of the form of (x, t, e) where x, t, e are the input covariates,
      event times and the censoring indicators respectively.

  """

  if dataset == 'SUPPORT':
    return _load_support_dataset()
  if dataset == 'PBC':
    sequential = kwargs.get('sequential', False)
    return _load_pbc_dataset(sequential)
  else:
    return NotImplementedError('Dataset '+dataset+' not implemented.')
