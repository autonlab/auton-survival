# coding=utf-8
# MIT License

# Copyright (c) 2021 Carnegie Mellon University, Auton Lab

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

"""Utility functions to load standard prognostics datasets to train and evaluate
the Deep Survival Machines models.

"""

import pandas as pd
import numpy as np

import dsm.prognostics.utilities as utilities 

def _increase_censoring(x, t, e, p, random_seed=0):

  np.random.seed(random_seed)

  n = len(x)
  to_censor = set(np.random.choice(np.arange(n), int(p*n), replace=False))
  
  for i in range(len(x)):
    if i in to_censor:
      
      d = len(x[i])
      cens_time = np.random.randint(1,d-1)

      x[i] = x[i][:cens_time]
      t[i] = np.arange(0,cens_time)[::-1]
      e[i] = 0*e[i][:cens_time]

  return x, t, e

def _remove_censored(x, t, e):

  idx = np.array([e_[0] for e_ in e]).astype('bool')
  return x[idx], t[idx], e[idx]


def _preprocess_turbofan(file, rul_file=None, windowsize=30, 
                         ft_cols=None, norm_tuple=None, flatten=False):

  data = pd.read_csv(file, sep=' ', header=None)

  flights = data[0]
  cycles = data[1]

  if rul_file is None:
    ruls = np.zeros(len(set(flights)))
  else:
    ruls = pd.read_csv(rul_file, sep=' ', header=None)[0].values

  if ft_cols is None:

    #ft_cols = data.columns
    ft_cols = [1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
    #ft_cols = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    ft_cols = list(set(ft_cols)-set([0, 26, 27]))

    same_cols = (data[ft_cols].max(axis=0) == data[ft_cols].min(axis=0)).values
    ft_cols = list(np.array(ft_cols)[~same_cols])

  features = data[ft_cols]

  if norm_tuple is None:
    norm_tuple = (features.min(), features.max())

  features = (features - norm_tuple[0])/(norm_tuple[1]-norm_tuple[0])
  features = 2*(features-0.5)

  x, t, e = [], [], []

  i = 0
  for flight in set(flights):

    features_ft = features[flights == flight]
    cycles_ft = cycles[flights == flight]

    d = cycles_ft.shape[0]
    featurized_cycles = d - windowsize

    x_, t_, e_ = [], [], []
    for j in range(featurized_cycles+1):

      cycles_ft_ = cycles_ft.iloc[j:j+windowsize].values
      features_ft_ = features_ft.iloc[j:j+windowsize, :].values.T
      target_ = d-cycles_ft_[-1]

      if flatten:
        x_.append(features_ft_.flatten())
      else:
        x_.append(features_ft_)

      t_.append(target_+ruls[i])
      e_.append(1)

    x.append(np.array(x_))
    t.append(np.array(t_))
    e.append(np.array(e_))

    i+=1

  return np.array(x), np.array(t), np.array(e), ft_cols, norm_tuple


def load_turbofan(cmapss_folder, experiment=1, windowsize=30,
                  flatten=False, sequential=True, test_last=True,
                  censoring=0.0, return_censored=True,
                  random_seed=0):

  """Helper function to load and preprocess the NASA Turbofan data.

  The NASA Trubofan Dataset is a popular dataset from the NASA Prognostics
  Center of Excellence consisting of synthetic dataset simulated using CMAPSS.

  TODO: Add synthetic censoring to the data.

  Parameters
  ----------
  cmapss_folder: str
    The location of the file consisting of the training dataset.
  experiment: int
    The CMAPSS Experiment to use. One of [1, 2, 3, 4]
  windowsize: int
    The size of the sliding window to extract features. (default: 30)
  flatten: bool
    Flag if the features at each time step are to be flattened into a vector.
    (Default: False)
  sequential: bool
    Flag if the data for each flight is to be stratified sequentially.
    (Default: True)
  test_last: bool
    (Default: True) Flag if only the last time step is to be output for the
    test data. Note: if sequential is True, this flag is ignored.
  censoring: float
    (Default: 0.) Proportion of training data points to be censored.
  return_censored: bool
    Flag to decide whether the censored data is returned or not.
  random_seed: 
    Seed for generating the censored data.
  References
  ----------
  [1] Saxena, Abhinav, Kai Goebel, Don Simon, and Neil Eklund.
  "Damage propagation modeling for aircraft engine run-to-failure simulation."
  International conference on prognostics and health management, IEEE, 2008.

  """

  tr_file = cmapss_folder+'train_FD00'+str(experiment)+'.txt'
  te_file = cmapss_folder+'test_FD00'+str(experiment)+'.txt'
  te_rul_file = cmapss_folder+'RUL_FD00'+str(experiment)+'.txt'

  x_tr, t_tr, e_tr, ft_cols, norm_tuple = _preprocess_turbofan(tr_file,
                                                               windowsize=windowsize,
                                                               flatten=flatten)
  x_te, t_te, e_te, _, _ = _preprocess_turbofan(te_file,
                                                rul_file=te_rul_file,
                                                windowsize=windowsize,
                                                ft_cols=ft_cols,
                                                norm_tuple=norm_tuple,
                                                flatten=flatten)

  if censoring>1e-10:
    x_tr, t_tr, e_tr = _increase_censoring(x_tr, t_tr, e_tr,
                                           p=censoring, random_seed=random_seed)   

  if not return_censored:
    x_tr, t_tr, e_tr = _remove_censored(x_tr, t_tr, e_tr)

  if not sequential:

    x_tr = utilities._unrollx(x_tr)
    t_tr = utilities._unrollt(t_tr)
    e_tr = utilities._unrollt(e_tr)

    if test_last:

      x_te = np.array([x_te_[-1:] for x_te_ in x_te])
      print("shape:", x_te.shape, x_te[0].shape)
      t_te = np.array([t_te_[-1:] for t_te_ in t_te])
      e_te = np.array([e_te_[-1:] for e_te_ in e_te])

    x_te = utilities._unrollx(x_te)
    t_te = utilities._unrollt(t_te)
    e_te = utilities._unrollt(e_te)

  return (x_tr, t_tr, e_tr), (x_te, t_te, e_te)
