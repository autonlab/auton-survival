# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

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


"""Utility functions to load standard datasets to train and evaluate the
Deep Survival Machines models.
"""


import io
import pkgutil

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torchvision

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

def _load_framingham_dataset(sequential, competing = False):
  """Helper function to load and preprocess the Framingham dataset.

  The Framingham Dataset is a subset of 4,434 participants of the well known,
  ongoing Framingham Heart study [1] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.

  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.

  References
  ----------
  [1] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).

  """

  data = pkgutil.get_data(__name__, 'datasets/framingham.csv')
  data = pd.read_csv(io.BytesIO(data))

  if not sequential:
    # Consider only first event
    data = data.groupby('RANDID').first()

  dat_cat = data[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
                  'educ', 'PREVCHD', 'PREVAP', 'PREVMI',
                  'PREVSTRK', 'PREVHYP']]
  dat_num = data[['TOTCHOL', 'AGE', 'SYSBP', 'DIABP',
                  'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

  x1 = pd.get_dummies(dat_cat)
  x2 = dat_num
  x = np.hstack([x1.values, x2.values])

  time = (data['TIMEDTH'] - data['TIME']).values
  event = data['DEATH'].values

  if competing:
    time_cvd = (data['TIMECVD'] - data['TIME']).values
    event[data['CVD'] == 1] = 2
    time[data['CVD'] == 1] = time_cvd[data['CVD'] == 1]

  x_ = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)

  if not sequential:
    return x_, time + 1, event, np.concatenate([x1.columns, x2.columns])
  else:
    x_, data, time, event = x_[time > 0], data[time > 0], time[time > 0], event[time > 0]
    x, t, e = [], [], []
    for id_ in sorted(list(set(data['RANDID']))):
      x.append(x_[data['RANDID'] == id_])
      t.append(time[data['RANDID'] == id_] + 1)
      e.append(event[data['RANDID'] == id_])
    return (x, x_), t, e, np.concatenate([x1.columns, x2.columns])

def _load_pbc_dataset(sequential, competing = False):
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

  if not sequential:
    # Consider only first event
    data = data.groupby('id').first()

  data['histologic'] = data['histologic'].astype(str)
  dat_cat = data[['drug', 'sex', 'ascites', 'hepatomegaly',
                  'spiders', 'edema', 'histologic']]
  dat_num = data[['serBilir', 'serChol', 'albumin', 'alkaline',
                  'SGOT', 'platelets', 'prothrombin']]
  age = data['age'] + data['years']

  x1 = pd.get_dummies(dat_cat)
  x2 = dat_num
  x3 = age
  x = np.hstack([x1.values, x2.values, x3.values.reshape(-1, 1)])

  time = (data['years'] - data['year']).values
  event = (data['status'] == 'dead').values.astype(int)
  if competing:
    event[data['status'] == 'transplanted'] = 2

  x_ = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)

  if not sequential:
    return x_, time + 1, event, x1.columns.tolist() + x2.columns.tolist() + [x3.name]
  else:
    x, t, e = [], [], []
    for id_ in sorted(list(set(data['id']))):
      x.append(x_[data['id'] == id_])
      t.append(time[data['id'] == id_] + 1)
      e.append(event[data['id'] == id_])
    return (x, x_), t, e, x1.columns.tolist() + x2.columns.tolist() + [x3.name]

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

  remove = ~np.isnan(t)
  return x[remove], t[remove] + 1, e[remove], np.concatenate([x1.columns, x2.columns])

def _load_mnist():
  """Helper function to load and preprocess the MNIST dataset.

  The MNIST database of handwritten digits, available from this page, has a
  training set of 60,000 examples, and a test set of 10,000 examples.
  It is a good database for people who want to try learning techniques and
  pattern recognition methods on real-world data while spending minimal
  efforts on preprocessing and formatting [1].

  Please refer to http://yann.lecun.com/exdb/mnist/.
  for the original datasource.

  References
  ----------
  [1]: LeCun, Y. (1998). The MNIST database of handwritten digits.
  http://yann.lecun.com/exdb/mnist/.

  """


  train = torchvision.datasets.MNIST(root='datasets/',
                                     train=True, download=True)
  x = train.data.numpy()
  x = np.expand_dims(x, 1).astype(float)
  t = train.targets.numpy().astype(float) + 1

  e, t = increase_censoring(np.ones(t.shape), t, p=.5)

  return x, t + 1, e, train.data.columns

def load_dataset(dataset='SUPPORT', normalize = True, **kwargs):
  """Helper function to load datasets to test Survival Analysis models.

  Currently implemented datasets include:

  **SUPPORT**: This dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].
  (Refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.)

  **PBC**: The Primary biliary cirrhosis dataset [2] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.

  **FRAMINGHAM**: This dataset is a subset of 4,434 participants of the well
  known, ongoing Framingham Heart study [3] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.

  References
  -----------

  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.

  [2] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.

  [3] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).

  Parameters
  ----------
  dataset: str
      The choice of dataset to load. Currently implemented is 'SUPPORT',
      'PBC' and 'FRAMINGHAM'.
  **kwargs: dict
      Dataset specific keyword arguments.

  Returns
  ----------
  tuple: (np.ndarray, np.ndarray, np.ndarray)
      A tuple of the form of (x, t, e) where x, t, e are the input covariates,
      event times and the censoring indicators respectively.

  """
  sequential = kwargs.get('sequential', False)
  competing = kwargs.get('competing', False)

  if dataset == 'SUPPORT':
    x, t, e, covariates = _load_support_dataset()
  elif dataset == 'PBC':
    x, t, e, covariates = _load_pbc_dataset(sequential, competing)
  elif dataset == 'FRAMINGHAM':
    x, t, e, covariates = _load_framingham_dataset(sequential, competing)
  elif dataset == 'MNIST':
    x, t, e, covariates = _load_mnist()
  else:
    raise NotImplementedError('Dataset '+dataset+' not implemented.')

  if isinstance(x, tuple):
    (x, x_all) = x
    if normalize:
      scaler = StandardScaler().fit(x_all)
      x = [scaler.transform(x_) for x_ in x]
  elif normalize:
    x = StandardScaler().fit_transform(x)
  return x, t, e, covariates
