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

r""" Deep Cox Proportional Hazards Model"""

from loguru import logger
import torch
import numpy as np

from .dcph_torch import DeepCoxPHTorch, DeepRecurrentCoxPHTorch
from .dcph_utilities import train_dcph, predict_survival

from auton_survival.utils import _dataframe_to_array
from auton_survival.models.utils.recurrent_nn_utils import _get_padded_features
from auton_survival.models.utils.recurrent_nn_utils import _get_padded_targets


class DeepCoxPH:
    """A Deep Cox Proportional Hazards model.

    This is the main interface to a Deep Cox Proportional Hazards model.
    A model is instantiated with approporiate set of hyperparameters and
    fit on numpy arrays consisting of the features, event/censoring times
    and the event/censoring indicators.

    For full details on Deep Cox Proportional Hazards, refer [1], [2].

    References
    ----------
    [1] <a href="https://arxiv.org/abs/1606.00931">DeepSurv: personalized
    treatment recommender system using a Cox proportional hazards
    deep neural network. BMC medical research methodology (2018)</a>

    [2] <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.4780140108">
    A neural network model for survival data. Statistics in medicine (1995)</a>

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
    >>> from auton_survival import DeepCoxPH
    >>> model = DeepCoxPH()
    >>> model.fit(x, t, e)

    """

    def __init__(self, layers=None, random_seed=0):
        self.layers = layers
        self.fitted = False
        self.initialized = False
        self.random_seed = random_seed

    @property
    def torch_module(self):
        if self.initialized:
            return self.torch_model[0]
        else:
            raise Exception("Torch module not initialized")

    def __call__(self):
        if self.fitted:
            logger.info("A fitted instance of the Deep Cox PH model")
        else:
            logger.info("An unfitted instance of the Deep Cox PH model")

        logger.info("Hidden Layers: {}", self.layers)

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
            vsize = int(vsize * x_train.shape[0])
            x_val, t_val, e_val = (
                x_train[-vsize:],
                t_train[-vsize:],
                e_train[-vsize:],
            )

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
        # Add random seed to get the same results like in dcm __init__.py
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        return DeepCoxPHTorch(inputdim, layers=self.layers, optimizer=optimizer)

    def init_torch_model(self, inputdim, optimizer):
        if not self.initialized:
            self.torch_model = (
                self._gen_torch_model(inputdim, optimizer),
                None,
            )
            self.initialized = True

    def fit(
        self,
        x,
        t,
        e,
        vsize=0.15,
        val_data=None,
        iters=1,
        learning_rate=1e-3,
        batch_size=100,
        optimizer="Adam",
        patience=3,
        breslow=True,
    ):
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

        processed_data = self._preprocess_training_data(
            x, t, e, vsize, val_data, self.random_seed
        )

        x_train, t_train, e_train, x_val, t_val, e_val = processed_data

        # Todo: Change this somehow. The base design shouldn't depend on child

        inputdim = x_train.shape[-1]

        self.init_torch_model(inputdim, optimizer)

        torch_module = self.torch_model[0]

        fitted_model, losses = train_dcph(
            torch_module,
            (x_train, t_train, e_train),
            (x_val, t_val, e_val),
            epochs=iters,
            lr=learning_rate,
            bs=batch_size,
            return_losses=True,
            random_seed=self.random_seed,
            patience=patience,
            breslow=breslow,
        )

        self.torch_model = (fitted_model[0].eval(), fitted_model[1])
        self.losses = losses
        self.fitted = True
        self.breslow = True if fitted_model[1] is not None else False

        return self

    @torch.inference_mode()
    def predict_time_independent_survival(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        if self.fitted:
            return 1 - self.predict_time_independent_risk(x)
        else:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_time_independent_survival`."
            )

    @torch.inference_mode()
    def predict_time_independent_risk(self, x: torch.Tensor) -> torch.Tensor:
        if self.fitted:
            x = self._preprocess_test_data(x)
            self.torch_model[0].eval()
            return self.torch_model[0](x)
        else:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_time_independent_risk`."
            )

    def predict_risk(self, x, t=None):
        if self.breslow and self.fitted:
            return 1 - self.predict_survival(x, t)
        else:
            raise Exception(
                "The model has not been fitted yet or has been fitted with `breslow=False`. Please fit the "
                + "model using the `fit` method on some training data and enable Breslow spline fit "
                + "before calling `predict_risk`."
            )

    def predict_survival(self, x, t=None):
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
        if not (self.fitted and self.breslow):
            raise Exception(
                "The model has not been fitted yet or has been fitted with `breslow=False`. Please fit the "
                + "model using the `fit` method on some training data and enable Breslow spline fit "
                + "before calling `predict_survival`."
            )

        x = self._preprocess_test_data(x)

        if t is not None:
            if not isinstance(t, list):
                t = [t]

        scores = predict_survival(self.torch_model, x, t)
        return scores


class DeepRecurrentCoxPH(DeepCoxPH):
    r"""A deep recurrent Cox PH model.

    This model is based on the paper:
    <a href="https://aclanthology.org/2021.naacl-main.358.pdf"> Leveraging
    Deep Representations of Radiology Reports in Survival Analysis for
    Predicting Heart Failure Patient Mortality. NAACL (2021)</a>

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
    >>> from dsm.contrib import DeepRecurrentCoxPH
    >>> model = DeepRecurrentCoxPH()
    >>> model.fit(x, t, e)

    """

    def __init__(self, layers=None, hidden=None, typ="LSTM", random_seed=0):
        super(DeepRecurrentCoxPH, self).__init__(layers=layers)

        self.typ = typ
        self.hidden = hidden
        self.random_seed = random_seed

    def __call__(self):
        if self.fitted:
            logger.info("A fitted instance of the Recurrent Deep Cox PH model")
        else:
            logger.info(
                "An unfitted instance of the Recurrent Deep Cox PH model"
            )

        logger.info("Hidden Layers: {}", self.layers)

    def _gen_torch_model(self, inputdim, optimizer):
        """Helper function to return a torch model."""

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        return DeepRecurrentCoxPHTorch(
            inputdim,
            layers=self.layers,
            hidden=self.hidden,
            optimizer=optimizer,
            typ=self.typ,
        )

    def _preprocess_test_data(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return torch.from_numpy(_get_padded_features(x)).float()

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

        x_train = torch.from_numpy(x_train).float()
        t_train = torch.from_numpy(t_train).float()
        e_train = torch.from_numpy(e_train).float()

        if val_data is None:
            vsize = int(vsize * x_train.shape[0])

            x_val, t_val, e_val = (
                x_train[-vsize:],
                t_train[-vsize:],
                e_train[-vsize:],
            )

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

            x_val = torch.from_numpy(x_val).float()
            t_val = torch.from_numpy(t_val).float()
            e_val = torch.from_numpy(e_val).float()

        return (x_train, t_train, e_train, x_val, t_val, e_val)
