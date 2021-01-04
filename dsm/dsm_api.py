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


"""
This module is a wrapper around torch implementations and
provides a convenient API to train Deep Survival Machines.
"""

from dsm.dsm_torch import DeepSurvivalMachinesTorch
from dsm.dsm_torch import DeepRecurrentSurvivalMachinesTorch
from dsm.dsm_torch import DeepConvolutionalSurvivalMachinesTorch

import dsm.losses as losses

from dsm.utilities import train_dsm
from dsm.utilities import _get_padded_features, _get_padded_targets
from dsm.utilities import _reshape_tensor_with_nans

import torch
import numpy as np

__pdoc__ = {}
__pdoc__["DeepSurvivalMachines.fit"] = True
__pdoc__["DeepSurvivalMachines._eval_nll"] = True
__pdoc__["DeepConvolutionalSurvivalMachines._eval_nll"] = True
__pdoc__["DSMBase"] = False


class DSMBase():
  """Base Class for all DSM models"""

  def __init__(self, k=3, layers=None, distribution="Weibull",
               temp=1000., discount=1.0):
    self.k = k
    self.layers = layers
    self.dist = distribution
    self.temp = temp
    self.discount = discount
    self.fitted = False

  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepSurvivalMachinesTorch(inputdim,
                                     k=self.k,
                                     layers=self.layers,
                                     dist=self.dist,
                                     temp=self.temp,
                                     discount=self.discount,
                                     optimizer=optimizer,
                                     risks=risks)

  def fit(self, x, t, e, vsize=0.15,
          iters=1, learning_rate=1e-3, batch_size=100,
          elbo=True, optimizer="Adam", random_state=100):

    r"""This method is used to train an instance of the DSM model.

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = 1 \) means the event took place.
    vsize: float
        Amount of data to set aside as the validation set.
    iters: int
        The maximum number of training iterations on the training dataset.
    learning_rate: float
        The learning rate for the `Adam` optimizer.
    batch_size: int
        learning is performed on mini-batches of input data. this parameter
        specifies the size of each mini-batch.
    elbo: bool
        Whether to use the Evidence Lower Bound for optimization.
        Default is True.
    optimizer: str
        The choice of the gradient based optimization method. One of
        'Adam', 'RMSProp' or 'SGD'.
    random_state: float
        random seed that determines how the validation set is chosen.

    """

    processed_data = self._prepocess_training_data(x, t, e, vsize,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    inputdim = x_train.shape[-1]
    maxrisk = int(e_train.max())
    model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
    model, _ = train_dsm(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val,
                         n_iter=iters,
                         lr=learning_rate,
                         elbo=elbo,
                         bs=batch_size)

    self.torch_model = model.eval()
    self.fitted = True
    
    return self    


  def compute_nll(self, x, t, e):
    r"""This function computes the negative log likelihood of the given data.
    In case of competing risks, the negative log likelihoods are summed over
    the different events' type.

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = r \) means the event r took place.
    
    Returns:
      float: Negative log likelihood.
    """
    if not(self.fitted):
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._prepocess_training_data(x, t, e, 0, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    x_val, t_val, e_val = x_val,\
        _reshape_tensor_with_nans(t_val),\
        _reshape_tensor_with_nans(e_val)
    loss = 0
    for r in range(self.torch_model.risks):
        loss += float(losses.conditional_loss(self.torch_model,
                      x_val, t_val, e_val, elbo=False,
                      risk=str(r+1)).detach().numpy())
    return loss

  def _prepocess_test_data(self, x):
    return torch.from_numpy(x)

  def _prepocess_training_data(self, x, t, e, vsize, random_state):

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

    vsize = int(vsize*x_train.shape[0])

    x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]
    x_train = x_train[:-vsize]
    t_train = t_train[:-vsize]
    e_train = e_train[:-vsize]

    return (x_train, t_train, e_train,
            x_val, t_val, e_val)

  def predict_mean(self, x, risk=1):
    r"""Returns the mean Time-to-Event \( t \)

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    Returns:
      np.array: numpy array of the mean time to event.

    """

    if self.fitted:
      x = self._prepocess_test_data(x)
      scores = losses.predict_mean(self.torch_model, x, risk=str(risk))
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_mean`.")
  def predict_risk(self, x, t, risk=1):
    r"""Returns the estimated risk of an event occuring before time \( t \)
      \( \widehat{\mathbb{P}}(T\leq t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the risks at each time in t.

    """

    if self.fitted:
      return 1-self.predict_survival(x, t, risk=str(risk))
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")


  def predict_survival(self, x, t, risk=1):
    r"""Returns the estimated survival probability at time \( t \),
      \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the survival probabilites at each time in t.

    """
    x = self._prepocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = losses.predict_cdf(self.torch_model, x, t, risk=str(risk))
      return np.exp(np.array(scores)).T
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")


class DeepSurvivalMachines(DSMBase):
  """A Deep Survival Machines model.

  This is the main interface to a Deep Survival Machines model.
  A model is instantiated with approporiate set of hyperparameters and
  fit on numpy arrays consisting of the features, event/censoring times
  and the event/censoring indicators.

  For full details on Deep Survival Machines, refer to our paper [1].

  References
  ----------
  [1] <a href="https://arxiv.org/abs/2003.01176">Deep Survival Machines:
  Fully Parametric Survival Regression and
  Representation Learning for Censored Data with Competing Risks."
  arXiv preprint arXiv:2003.01176 (2020)</a>

  Parameters
  ----------
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  distribution: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  Example
  -------
  >>> from dsm import DeepSurvivalMachines
  >>> model = DeepSurvivalMachines()
  >>> model.fit(x, t, e)

  """

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the Deep Survival Machines model")
    else:
      print("An unfitted instance of the Deep Survival Machines model")

    print("Number of underlying distributions (k):", self.k)
    print("Hidden Layers:", self.layers)
    print("Distribution Choice:", self.dist)


class DeepRecurrentSurvivalMachines(DSMBase):

  """The Deep Recurrent Survival Machines model to handle data with
  time-dependent covariates.

  """

  def __init__(self, k=3, layers=None, hidden=None,
               distribution="Weibull", temp=1000., discount=1.0, typ="LSTM"):
    super(DeepRecurrentSurvivalMachines, self).__init__(k=k,
                                                        layers=layers,
                                                        distribution=distribution,
                                                        temp=temp,
                                                        discount=discount)
    self.hidden = hidden
    self.typ = typ
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepRecurrentSurvivalMachinesTorch(inputdim,
                                              k=self.k,
                                              layers=self.layers,
                                              hidden=self.hidden,
                                              dist=self.dist,
                                              temp=self.temp,
                                              discount=self.discount,
                                              optimizer=optimizer,
                                              typ=self.typ,
                                              risks=risks)

  def _prepocess_test_data(self, x):
    return torch.from_numpy(_get_padded_features(x))

  def _prepocess_training_data(self, x, t, e, vsize, random_state):
    """RNNs require different preprocessing for variable length sequences"""

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)

    x = _get_padded_features(x)
    t = _get_padded_targets(t)
    e = _get_padded_targets(e)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

    vsize = int(vsize*x_train.shape[0])
    x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

    x_train = x_train[:-vsize]
    t_train = t_train[:-vsize]
    e_train = e_train[:-vsize]

    return (x_train, t_train, e_train,
            x_val, t_val, e_val)


class DeepConvolutionalSurvivalMachines(DSMBase):
  """The Deep Convolutional Survival Machines model to handle data with
  image-based covariates.

  """

  def __init__(self, k=3, layers=None, hidden=None, 
               distribution='Weibull', temp=1000., discount=1.0, typ='ConvNet'):
    super(DeepConvolutionalSurvivalMachines, self).__init__(k=k,
                                                            distribution=distribution,
                                                            temp=temp,
                                                            discount=discount)
    self.hidden = hidden
    self.typ = typ
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepConvolutionalSurvivalMachinesTorch(inputdim,
                                              k=self.k,
                                              hidden=self.hidden,
                                              dist=self.dist,
                                              temp=self.temp,
                                              discount=self.discount,
                                              optimizer=optimizer,
                                              typ=self.typ,
                                              risks=risks)
