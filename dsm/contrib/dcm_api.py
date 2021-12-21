
import torch
import numpy as np

from dsm.contrib.dcm_torch import DeepCoxMixturesTorch
from dsm.contrib.dcm_utilities import *


class DeepCoxMixtures():
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
  Example
  -------
  >>> from dsm.contrib import DeepCoxMixtures
  >>> model = DeepCoxMixtures()
  >>> model.fit(x, t, e)

  """
  def __init__(self, k=3, layers=None, distribution="Weibull",
               temp=1000., discount=1.0):
    self.k = k
    self.layers = layers
    self.dist = distribution
    self.temp = temp
    self.discount = discount
    self.fitted = False

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the Deep Cox Mixtures model")
    else:
      print("An unfitted instance of the Deep Cox Mixtures model")

    print("Number of underlying cox distributions (k):", self.k)
    print("Hidden Layers:", self.layers)

  def _preprocess_test_data(self, x):
    return torch.from_numpy(x).float()

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_state):

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
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

      x_val = torch.from_numpy(x_val).float()
      t_val = torch.from_numpy(t_val).float()
      e_val = torch.from_numpy(e_val).float()

    return (x_train, t_train, e_train, x_val, t_val, e_val)

  def _gen_torch_model(self, inputdim, optimizer):
    """Helper function to return a torch model."""
    return DeepCoxMixturesTorch(inputdim,
                                k=self.k,
                                layers=self.layers,
                                optimizer=optimizer)

  def fit(self, x, t, e, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          optimizer="Adam", random_state=100):

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
    random_state: float
        random seed that determines how the validation set is chosen.

    """

    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
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
                         smoothing_factor=None,
                         use_posteriors=True,)

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

  def compute_nll(self, x, t, e):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    with torch.no_grad():
      return - get_posteriors(repair_probs(get_likelihood(self.torch_model[0], self.torch_model[1], x_val, t_val, e_val))).sum().item()

  def predict_alphas(self, x):
    x = self._preprocess_test_data(x)
    if self.fitted:
      alphas, _ = self.torch_model[0](x)
      return alphas.detach().exp().cpu().numpy()
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_alphas`.")