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

Deep Survival Machines
----------------------

.. figure:: https://ndownloader.figshare.com/files/25259852
   :figwidth: 20 %
   :alt: Schematic Description of Deep Survival Machines

   Schematic Description of Deep Survival Machines.

**Deep Survival Machines (DSM)** is a fully parametric approach to model
Time-to-Event outcomes in the presence of Censoring first introduced in
[\[1\]](https://arxiv.org/abs/2003.01176).
In the context of Healthcare ML and Biostatistics, this is known as 'Survival
Analysis'. The key idea behind Deep Survival Machines is to model the
underlying event outcome distribution as a mixure of some fixed \( k \)
parametric distributions. The parameters of these mixture distributions as
well as the mixing weights are modelled using Neural Networks.

Example Usage
-------------

>>> from dsm import DeepSurvivalMachines
>>> from dsm import datasets
>>> # load the SUPPORT dataset.
>>> x, t, e = datasets.load_dataset('SUPPORT')
>>> # instantiate a DeepSurvivalMachines model.
>>> model = DeepSurvivalMachines()
>>> # fit the model to the dataset.
>>> model.fit(x, t, e)
>>> # estimate the predicted risks at the time
>>> model.predict_risk(x, 10)


Deep Recurrent Survival Machines
--------------------------------

.. figure:: https://ndownloader.figshare.com/files/28329918
   :figwidth: 20 %
   :alt: Schematic Description of Deep Survival Machines

   Schematic Description of Deep Survival Machines.

**Deep Recurrent Survival Machines (DRSM)** builds on the original **DSM**
model and allows for learning of representations of the input covariates using
**Recurrent Neural Networks** like **LSTMs, GRUs**. Deep Recurrent Survival
Machines is a natural fit to model problems where there are time dependendent
covariates. Examples include situations where we are working with streaming
data like vital signs, degradation monitoring signals in predictive
maintainance. **DRSM** allows the learnt representations at each time step to
involve historical context from previous time steps. **DRSM** implementation in
`dsm` is carried out through an easy to use API,
`DeepRecurrentSurvivalMachines` that accepts lists of data streams and
corresponding failure times. The module automatically takes care of appropriate
batching and padding of variable length sequences.


Deep Convolutional Survival Machines
------------------------------------

Predictive maintenance and medical imaging sometimes requires to work with
image streams. Deep Convolutional Survival Machines extends **DSM** and
**DRSM** to learn representations of the input image data using
convolutional layers. If working with streaming data, the learnt
representations are then passed through an LSTM to model temporal dependencies
before determining the underlying survival distributions.

..warning:: Not Implemented Yet!

References
----------

Please cite the following papers if you are using the `auton_survival` package.

[1] [Deep Survival Machines:
Fully Parametric Survival Regression and
Representation Learning for Censored Data with Competing Risks."
IEEE Journal of Biomedical and Health Informatics (2021)](https://arxiv.org/abs/2003.01176)</a>

```
  @article{nagpal2021dsm,
  title={Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks},
  author={Nagpal, Chirag and Li, Xinyu and Dubrawski, Artur},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={25},
  number={8},
  pages={3163--3175},
  year={2021},
  publisher={IEEE}
  }
```

[2] [Deep Parametric Time-to-Event Regression with Time-Varying Covariates. AAAI
Spring Symposium (2021)](http://proceedings.mlr.press/v146/nagpal21a.html)</a>

```
  @InProceedings{pmlr-v146-nagpal21a,
  title={Deep Parametric Time-to-Event Regression with Time-Varying Covariates},
  author={Nagpal, Chirag and Jeanselme, Vincent and Dubrawski, Artur},
  booktitle={Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
  }
```

[3] [Deep Cox Mixtures for Survival Regression. Conference on Machine Learning for
Healthcare (2021)](https://arxiv.org/abs/2101.06536)</a>

```
  @inproceedings{nagpal2021dcm,
  title={Deep Cox mixtures for survival regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  booktitle={Machine Learning for Healthcare Conference},
  pages={674--708},
  year={2021},
  organization={PMLR}
  }
```

[4] [Counterfactual Phenotyping with Censored Time-to-Events (2022)](https://arxiv.org/abs/2202.11089)</a>

```
  @article{nagpal2022counterfactual,
  title={Counterfactual Phenotyping with Censored Time-to-Events},
  author={Nagpal, Chirag and Goswami, Mononito and Dufendach, Keith and Dubrawski, Artur},
  journal={arXiv preprint arXiv:2202.11089},
  year={2022}
  }
```


"""

import torch
import numpy as np

from .dsm_torch import DeepSurvivalMachinesTorch
from .dsm_torch import DeepRecurrentSurvivalMachinesTorch
from .dsm_torch import DeepConvolutionalSurvivalMachinesTorch
from .dsm_torch import DeepCNNRNNSurvivalMachinesTorch

from . import losses

from .utilities import train_dsm
from .utilities import _get_padded_features, _get_padded_targets
from .utilities import _reshape_tensor_with_nans

from auton_survival.utils import _dataframe_to_array


__pdoc__ = {}
__pdoc__["DeepSurvivalMachines.fit"] = True
__pdoc__["DeepSurvivalMachines._eval_nll"] = True
__pdoc__["DeepConvolutionalSurvivalMachines._eval_nll"] = True
__pdoc__["DSMBase"] = False


class DSMBase():
  """Base Class for all DSM models"""

  def __init__(self, k=3, layers=None, distribution="Weibull",
               temp=1000., discount=1.0, random_seed=0):
    self.k = k
    self.layers = layers
    self.dist = distribution
    self.temp = temp
    self.discount = discount
    self.fitted = False
    self.random_seed = random_seed

  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)
    
    return DeepSurvivalMachinesTorch(inputdim,
                                     k=self.k,
                                     layers=self.layers,
                                     dist=self.dist,
                                     temp=self.temp,
                                     discount=self.discount,
                                     optimizer=optimizer,
                                     risks=risks)

  def fit(self, x, t, e, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          elbo=True, optimizer="Adam"):

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
    elbo: bool
        Whether to use the Evidence Lower Bound for optimization.
        Default is True.
    optimizer: str
        The choice of the gradient based optimization method. One of
        'Adam', 'RMSProp' or 'SGD'.

    """

    processed_data = self._preprocess_training_data(x, t, e,
                                                    vsize, val_data,
                                                    self.random_seed)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child
    if type(self).__name__ in ["DeepConvolutionalSurvivalMachines",
                               "DeepCNNRNNSurvivalMachines"]:
      inputdim = tuple(x_train.shape)[-2:]
    else:
      inputdim = x_train.shape[-1]

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
    model, _ = train_dsm(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val,
                         n_iter=iters,
                         lr=learning_rate,
                         elbo=elbo,
                         bs=batch_size,
                         random_seed=self.random_seed)

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
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
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

  def _preprocess_test_data(self, x):
    x = _dataframe_to_array(x)
    return torch.from_numpy(x)

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_seed):

    x = _dataframe_to_array(x)
    t = _dataframe_to_array(t)
    e = _dataframe_to_array(e)

    idx = list(range(x.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(idx)
    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

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

      x_val = torch.from_numpy(x_val).double()
      t_val = torch.from_numpy(t_val).double()
      e_val = torch.from_numpy(e_val).double()

    return (x_train, t_train, e_train, x_val, t_val, e_val)


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
      x = self._preprocess_test_data(x)
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
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = losses.predict_cdf(self.torch_model, x, t, risk=str(risk))
      return np.exp(np.array(scores)).T
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_pdf(self, x, t, risk=1):
    r"""Returns the estimated pdf at time \( t \),
      \( \widehat{\mathbb{P}}(T = t|X) \) for some input data \( x \). 

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which pdf is
        to be computed
    Returns:
      np.array: numpy array of the estimated pdf at each time in t.

    """
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = losses.predict_pdf(self.torch_model, x, t, risk=str(risk))
      return np.exp(np.array(scores)).T
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")


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

  For full details on Deep Recurrent Survival Machines, refer to our paper [1].

  References
  ----------
  [1] <a href="http://proceedings.mlr.press/v146/nagpal21a.html">
  Deep Parametric Time-to-Event Regression with Time-Varying Covariates
  AAAI Spring Symposium on Survival Prediction</a>

  """

  def __init__(self, k=3, layers=None, hidden=None,
               distribution="Weibull", temp=1000., discount=1.0, typ="LSTM",
               random_seed=0):
    super(DeepRecurrentSurvivalMachines, self).__init__(k=k,
                                                        layers=layers,
                                                        distribution=distribution,
                                                        temp=temp,
                                                        discount=discount,
                                                        random_seed=random_seed)
    self.hidden = hidden
    self.typ = typ
    self.random_seed = random_seed
  
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)
    
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

  def _preprocess_test_data(self, x):
    x = _dataframe_to_array(x)
    return torch.from_numpy(_get_padded_features(x))

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_seed):
    """RNNs require different preprocessing for variable length sequences"""

    x = _dataframe_to_array(x)
    t = _dataframe_to_array(t)
    e = _dataframe_to_array(e)

    idx = list(range(x.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(idx)

    x = _get_padded_features(x)
    t = _get_padded_targets(t)
    e = _get_padded_targets(e)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

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

      x_val = _get_padded_features(x_val)
      t_val = _get_padded_features(t_val)
      e_val = _get_padded_features(e_val)

      x_val = torch.from_numpy(x_val).double()
      t_val = torch.from_numpy(t_val).double()
      e_val = torch.from_numpy(e_val).double()

    return (x_train, t_train, e_train, x_val, t_val, e_val)


class DeepConvolutionalSurvivalMachines(DSMBase):
  """The Deep Convolutional Survival Machines model to handle data with
  image-based covariates.

  """

  def __init__(self, k=3, layers=None, hidden=None, 
               distribution="Weibull", temp=1000., discount=1.0, typ="ConvNet"):
    super(DeepConvolutionalSurvivalMachines, self).__init__(k=k,
                                                            distribution=distribution,
                                                            temp=temp,
                                                            discount=discount,
                                                            random_seed=0)
    self.hidden = hidden
    self.typ = typ
    self.random_seed = random_seed
    
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)
    
    return DeepConvolutionalSurvivalMachinesTorch(inputdim,
                                                  k=self.k,
                                                  hidden=self.hidden,
                                                  dist=self.dist,
                                                  temp=self.temp,
                                                  discount=self.discount,
                                                  optimizer=optimizer,
                                                  typ=self.typ,
                                                  risks=risks)


class DeepCNNRNNSurvivalMachines(DeepRecurrentSurvivalMachines):

  """The Deep CNN-RNN Survival Machines model to handle data with
  moving image streams.

  """

  def __init__(self, k=3, layers=None, hidden=None,
               distribution="Weibull", temp=1000., discount=1.0, typ="LSTM"):
    super(DeepCNNRNNSurvivalMachines, self).__init__(k=k,
                                                     layers=layers,
                                                     distribution=distribution,
                                                     temp=temp,
                                                     discount=discount,
                                                     random_seed=0)
    self.hidden = hidden
    self.typ = typ
    self.random_seed = random_seed

  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)
    
    return DeepCNNRNNSurvivalMachinesTorch(inputdim,
                                           k=self.k,
                                           layers=self.layers,
                                           hidden=self.hidden,
                                           dist=self.dist,
                                           temp=self.temp,
                                           discount=self.discount,
                                           optimizer=optimizer,
                                           typ=self.typ,
                                           risks=risks)
