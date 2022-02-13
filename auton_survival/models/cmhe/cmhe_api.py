from .cmhe_torch import DeepCMHETorch
from .cmhe_utilities import train_cmhe, predict_survival
from .cmhe_utilities import predict_latent_phi, predict_latent_z

import torch
import numpy as np

class DeepCoxMixturesHeterogenousEffects:
  """A Deep Cox Mixtures with Heterogenous Effects model.

  This is the main interface to a Deep Cox Mixture with Heterogenous Effects.
  A model is instantiated with approporiate set of hyperparameters and
  fit on numpy arrays consisting of the features, event/censoring times
  and the event/censoring indicators.

  For full details on Deep Cox Mixture, refer to the paper [1].

  References
  ----------
  [1] Nagpal, C., Goswami M., Dufendach K., and Artur Dubrawski.
  "Counterfactual phenotyping for censored Time-to-Events" (2022).

  Parameters
  ----------
  k: int
      The number of underlying base survival phenotypes.
  g: int
      The number of underlying treatment effect phenotypes.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.

  Example
  -------
  >>> from auton_survival import DeepCoxMixturesHeterogenousEffects
  >>> model = DeepCoxMixturesHeterogenousEffects(k=2, g=3)
  >>> model.fit(x, t, e, a)

  """

  def __init__(self, k, g, layers=None):

    self.layers = layers
    self.fitted = False
    self.k = k
    self.g = g

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the CMHE model")
    else:
      print("An unfitted instance of the CMHE model")

    print("Hidden Layers:", self.layers)

  def _preprocess_test_data(self, x, a=None):
    if a is not None:
      return torch.from_numpy(x).float(), torch.from_numpy(a).float()
    else:
      return torch.from_numpy(x).float()

  def _preprocess_training_data(self, x, t, e, a, vsize, val_data,
                                random_state):

    idx = list(range(x.shape[0]))

    np.random.seed(random_state)
    np.random.shuffle(idx)

    x_tr, t_tr, e_tr, a_tr = x[idx], t[idx], e[idx], a[idx]

    x_tr = torch.from_numpy(x_tr).float()
    t_tr = torch.from_numpy(t_tr).float()
    e_tr = torch.from_numpy(e_tr).float()
    a_tr = torch.from_numpy(a_tr).float()

    if val_data is None:

      vsize = int(vsize*x_tr.shape[0])
      x_vl, t_vl, e_vl, a_vl = x_tr[-vsize:], t_tr[-vsize:], e_tr[-vsize:], a_tr[-vsize:]

      x_tr = x_tr[:-vsize]
      t_tr = t_tr[:-vsize]
      e_tr = e_tr[:-vsize]
      a_tr = a_tr[:-vsize]

    else:

      x_vl, t_vl, e_vl, a_vl = val_data

      x_vl = torch.from_numpy(x_vl).float()
      t_vl = torch.from_numpy(t_vl).float()
      e_vl = torch.from_numpy(e_vl).float()
      a_vl = torch.from_numpy(a_vl).float()

    return (x_tr, t_tr, e_tr, a_tr,
    	      x_vl, t_vl, e_vl, a_vl)

  def _gen_torch_model(self, inputdim, optimizer):
    """Helper function to return a torch model."""
    return DeepCMHETorch(self.k, self.g, inputdim,
                         layers=self.layers,
                         optimizer=optimizer)

  def fit(self, x, t, e, a, vsize=0.15, val_data=None,
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
    a: np.ndarray
        A numpy array of the treatment assignment indicators, \( a \).
        \( a = 1 \) means the individual was treated.
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

    processed_data = self._preprocess_training_data(x, t, e, a,
                                                    vsize, val_data,
                                                    random_state)

    x_tr, t_tr, e_tr, a_tr, x_vl, t_vl, e_vl, a_vl = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child

    inputdim = x_tr.shape[-1]

    model = self._gen_torch_model(inputdim, optimizer)

    model, _ = train_cmhe(model,
                          (x_tr, t_tr, e_tr, a_tr),
                          (x_vl, t_vl, e_vl, a_vl),
                          epochs=iters,
                          lr=learning_rate,
                          bs=batch_size,
                          return_losses=True)

    self.torch_model = (model[0].eval(), model[1])
    self.fitted = True

    return self

  def predict_risk(self, x, a, t=None):

    if self.fitted:
      return 1-self.predict_survival(x, a, t)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")

  def predict_survival(self, x, a, t=None):
    r"""Returns the estimated survival probability at time \( t \),
      \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    a: np.ndarray
        A numpy array of the treatmeant assignment, \( a \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the survival probabilites at each time in t.

    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

    x = self._preprocess_test_data(x, a)

    if t is not None:
      if not isinstance(t, list):
        t = [t]

    scores = predict_survival(self.torch_model, x, a, t)
    return scores

  def predict_latent_z(self, x):

    r"""Returns the estimated latent base survival group \( z \) given the confounders \( x \)."""

    x = self._preprocess_test_data(x)

    if self.fitted:
      scores = predict_latent_z(self.torch_model, x)
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_latent_z`.")

  def predict_latent_phi(self, x):

    r"""Returns the estimated latent treatment effect group \( \phi \) given the confounders \( x \)."""

    x = self._preprocess_test_data(x)

    if self.fitted:
      scores = predict_latent_phi(self.torch_model, x)
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_latent_phi`.")