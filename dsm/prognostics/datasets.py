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

def _preprocess_turbofan(data, ruls=None, run_col=0, windowsize=10,
                         r_mean=True, r_std=True,
                         standardize_with=None):

  flights = set(data[run_col])
  ft_cols = list(set(data.columns)-set([run_col]))

  feats = []
  targets = []
  events = []

  if ruls is None:
    ruls = np.zeros(len(flights))
  else:
    ruls = ruls.values[:, 0]

  i = 0

  for flight in flights:

    dat = data[ft_cols][data[run_col] == flight]
    feats_ = dat.values

    targets_ = list(range(len(feats_)))
    targets_.reverse()
    targets.append(np.array(targets_)+1e-6+ruls[i])
    events.append(np.ones(len(targets_)))

    if r_mean:
      m_feats = dat.rolling(windowsize).mean().bfill().values
      feats_ = np.hstack([feats_, m_feats])

    if r_std:
      s_feats = dat.rolling(windowsize).std().bfill().values
      feats_ = np.hstack([feats_, s_feats])

    feats.append(feats_)
    i += 1

  pop_ms = []
  pop_ss = []

  if standardize_with is not None:
    pop_ms, pop_ss = standardize_with

  else:
    for i in range(feats[0].shape[1]):
      pop_m = np.concatenate([out_[:, i] for out_ in feats]).mean()
      pop_s = np.concatenate([out_[:, i] for out_ in feats]).std()

      pop_ms.append(pop_m)
      pop_ss.append(pop_s)

  d = feats[0].shape[1]

  for i in range(len(feats)):
    for j in range(d):
      feats[i][:, j] = (feats[i][:, j]-pop_ms[j])/pop_ss[j]
      feats[i][np.isnan(feats[i])] = 0

  if standardize_with is None:
    return np.array(feats), np.array(targets), np.array(events), (pop_ms, pop_ss)
  else:
    return np.array(feats), np.array(targets), np.array(events)


def load_turbofan(train_data, test_data, run_col=0,
                  windowsize=10, r_mean=True, r_std=True):

  """Helper function to load and preprocess the NASA Turbofan data.

  The NASA Trubofan Dataset is a popular dataset from the NASA Prognostics
  Center of Excellence consisting of synthetic dataset simulated using CMAPSS.

  Parameters
  ----------
  train_data: str
    The location of the file consisting of the training dataset.
  test_data: tuple
    A tuple of (str, str) with the location of the file consisting
    of the testing dataset, including the testing data file and the
    remaining useful life.
  run_col: int
    An integer specifying the column that identifies each separate run.
  windowsize: int
    The size of the rolling window to extract features. (default: 10)
  r_mean: bool
    A boolean indicating if the rolling means are to be included in
    the feature set.
  r_std: bool
    A boolean indicating if the rolling standard deviation is to be
    included in the feature set.

  References
  ----------
  [1] Saxena, Abhinav, Kai Goebel, Don Simon, and Neil Eklund.
  "Damage propagation modeling for aircraft engine run-to-failure simulation."
  International conference on prognostics and health management, IEEE, 2008.


  """

  tr_data = pd.read_csv(train_data, sep=' ', header=None)
  te_data = pd.read_csv(test_data[0], sep=' ', header=None)
  te_ruls = pd.read_csv(test_data[1], sep=' ', header=None)

  x_tr, t_tr, e_tr, std_with = _preprocess_turbofan(tr_data, None,
                                                    run_col=run_col,
                                                    windowsize=windowsize,
                                                    r_mean=r_mean,
                                                    r_std=r_std)

  x_te, t_te, e_te = _preprocess_turbofan(te_data, te_ruls,
                                          run_col=run_col,
                                          windowsize=windowsize,
                                          r_mean=r_mean,
                                          r_std=r_std,
                                          standardize_with=std_with)

  return (x_tr, t_tr, e_tr), (x_te, t_te, e_te)
