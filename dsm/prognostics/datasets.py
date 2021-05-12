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

    ft_cols = list(set(data.columns)-set([0, 26, 27]))
    same_cols = (data[ft_cols].max(axis=0)== data[ft_cols].min(axis=0)).values
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

    x.append(x_)
    t.append(t_)
    e.append(e_)

    i+=1

  return np.array(x), np.array(t), np.array(e), ft_cols, norm_tuple


def load_turbofan(cmapss_folder, experiment=1, windowsize=30,
                  flatten=False, sequential=True, test_last=True):

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

  if not sequential:

    x_tr = utilities._unrollx(x_tr)
    t_tr = utilities._unrollt(t_tr)
    e_tr = utilities._unrollt(e_tr)

    if test_last:

      x_te = np.array([x_te_[-1:] for x_te_ in x_te])
      t_te = np.array([t_te_[-1:] for t_te_ in t_te])
      e_te = np.array([e_te_[-1:] for e_te_ in e_te])

    x_te = utilities._unrollx(x_te)
    t_te = utilities._unrollt(t_te)
    e_te = utilities._unrollt(e_te)

  return (x_tr, t_tr, e_tr), (x_te, t_te, e_te)
