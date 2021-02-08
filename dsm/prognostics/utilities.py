
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

"""Utility functions to evaluate DSM models on prognostics tasks.

"""

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score

import numpy as np


def _repackage_predictions(predictions, inputshape):

  repack = []

  idx = 0
  for row in inputshape:
    repack.append(predictions[idx:idx+len(row)])
    idx += len(row)

  return np.array(repack)

def _unrollt(data):
  return np.concatenate([dat for dat in data])

def _unrollx(data):
  return np.vstack([dat for dat in data])

def _eval_independent(predictions, t, horizon):

  fpr, tpr, _ = roc_curve(_unrollt(t) <= horizon, predictions)
  roc_auc = roc_auc_score(_unrollt(t) <= horizon, predictions)

  prec, rec, _ = precision_recall_curve(_unrollt(t) <= horizon, predictions)
  pr_auc = average_precision_score(_unrollt(t) <= horizon, predictions)

  return (fpr, tpr, roc_auc), (prec, rec, pr_auc)



def _eval_asbatch():
    return



def evaluate(predictions,
             test_data,
             typ='batch',
             horizon=None,
             eval_at=None):

  if horizon is None:
    raise Exception("Please provide horizon to evaluate on.")

  if typ == "independent":
    if eval_at is not None:
      print("WARNING: No threshold provided. Evaluation would be performed\
             at end of each cycle.")

    return _eval_independent(predictions, test_data, horizon)


  if typ == "batch":
    if eval_at is None:
      print("WARNING: No threshold provided. Evaluation would be performed\
             at end of each cycle.")






