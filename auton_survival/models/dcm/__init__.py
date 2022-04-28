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

r"""

Deep Cox Mixtures
------------------

The Cox Mixture involves the assumption that the survival function
of the individual to be a mixture of K Cox Models. Conditioned on each
subgroup \( Z=k \); the PH assumptions are assumed to hold and the baseline
hazard rates is determined non-parametrically using an spline-interpolated
Breslow's estimator.

For full details on Deep Cox Mixture, refer to the paper [1].

References
----------
[1] <a href="https://arxiv.org/abs/2101.06536">Deep Cox Mixtures
for Survival Regression. Machine Learning in Health Conference (2021)</a>

```
  @article{nagpal2021dcm,
  title={Deep Cox mixtures for survival regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  journal={arXiv preprint arXiv:2101.06536},
  year={2021}
  }
```

"""

import torch
import numpy as np

from .dcm_torch import DeepCoxMixturesTorch
from .dcm_utilities import train_dcm, predict_survival, predict_latent_z

from auton_survival.utils import _dataframe_to_array


class DeepCoxMixtures:
  """A Deep Cox Mixture model.

  This is the main interface to a Deep Cox Mixture model.
  A model is instantiated with approporiate set of hyperparameters and
  fit on numpy arrays consisting of the features, event/censoring times
  and the event/censoring indicators.

  For full details on Deep Cox Mixture, refer to the paper [1].

  References
  ----------
  [1] <a href="https://arxiv.org/abs/2101.06536">Deep Cox Mixtures
  for Survival Regression. Machine Learning in Health Conference (2021)</a>

  Parameters
  ----------
  k: int
      The number of underlying Cox distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  random_seed: int
      Controls the reproducibility of called functions.

  Example
  -------
  >>> from auton_survival.models.dcm import DeepCoxMixtures
  >>> model = DeepCoxMixtures()
  >>> model.fit(x, t, e)

  """

  def __init__(self, k=3, layers=None, gamma=10,
               smoothing_factor=1e-4, use_activation=False,
               random_seed=0):

    self.k = k
    self.layers = layers
    self.fitted = False
    self.gamma = gamma
    self.smoothing_factor = smoothing_factor
    self.use_activation = use_activation
    self.random_seed = random_seed 

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the Deep Cox Mixtures model")
    else:
      print("An unfitted instance of the Deep Cox Mixtures model")

    print("Number of underlying cox distributions (k):", self.k)
    print("Hidden Layers:", self.layers)

  def _preprocess_test_data(self, x):
    x = _dataframe_to_array(x)
    return torch.from_numpy(x).float()

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_seed):

    x = _dataframe_to_array(x)
    t = _dataframe_to_array(t)
    e = _dataframe_to_array(e)

    idx = list(range(x.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(idx)
    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).float()
    e_train = torch.from_numpy(e_train).float()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])
      x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

      x_train = x_train[:-vsize]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]

    else:

      x_val, t_val, e_val = val_data

      x_val = _dataframe_to_array(x_val)
      t_val = _dataframe_to_array(t_val)
      e_val = _dataframe_to_array(e_val)

      x_val = torch.from_numpy(x_val).float()
      t_val = torch.from_numpy(t_val).float()
      e_val = torch.from_numpy(e_val).float()

    return (x_train, t_train, e_train, x_val, t_val, e_val)

  def _gen_torch_model(self, inputdim, optimizer):
    """Helper function to return a torch model."""

    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)

    return DeepCoxMixturesTorch(inputdim,
                                k=self.k,
                                gamma=self.gamma,
                                use_activation=self.use_activation,
                                layers=self.layers,
                                optimizer=optimizer)

  def fit(self, x, t, e, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          optimizer="Adam"):

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
    val_data: tuple
        A tuple of the validation dataset. If passed vsize is ignored.
    iters: int
        The maximum number of training iterations on the training dataset.
    learning_rate: float
        The learning rate for the `Adam` optimizer.
    batch_size: int
        learning is performed on mini-batches of input data. this parameter
        specifies the size of each mini-batch.
    optimizer: str
        The choice of the gradient based optimization method. One of
        'Adam', 'RMSProp' or 'SGD'.

    """

    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   self.random_seed)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child

    inputdim = x_train.shape[-1]

    model = self._gen_torch_model(inputdim, optimizer)

    model, _ = train_dcm(model,
                         (x_train, t_train, e_train),
                         (x_val, t_val, e_val),
                         epochs=iters,
                         lr=learning_rate,
                         bs=batch_size,
                         return_losses=True,
                         smoothing_factor=self.smoothing_factor,
                         use_posteriors=True,
                         random_seed=self.random_seed)

    self.torch_model = (model[0].eval(), model[1])
    self.fitted = True

    return self


  def predict_survival(self, x, t):
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
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = predict_survival(self.torch_model, x, t)
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_latent_z(self, x):

    x = self._preprocess_test_data(x)

    if self.fitted:
      scores = predict_latent_z(self.torch_model, x)
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_latent_z`.")
