# coding=utf-8
# Copyright 2020 Chirag Nagpal, Auton Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch model definitons for the Deep Survival Machines model

This includes definitons for the Torch Deep Survival Machines module.
The main interface is the DeepSurvivalMachines class which inherits
from torch.nn.Module.

Note: NOT DESIGNED TO BE CALLED DIRECTLY!!!
"""

import torch.nn as nn
import torch


def create_representation(inputdim, layers, activation):
  """Helper function to generate the representation function for DSM.

  Deep Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  layers: list
      A list consisting of the number of neurons in each hidden layer.
  activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

  Returns
  ----------
  an MLP with torch.nn.Module with the specfied structure.
  """

  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'SeLU':
    act = nn.SELU()

  modules = []
  prevdim = inputdim

  for hidden in layers:
    modules.append(nn.Linear(prevdim, hidden, bias=False))
    modules.append(act)
    prevdim = hidden

  return nn.Sequential(*modules)

class DeepSurvivalMachinesTorch(nn.Module):
  """A Torch implementation of Deep Survival Machines model.

  This is an implementation of Deep Survival Machines model in torch.
  It inherits from the torch.nn.Module class and includes references to the
  representation learning MLP, the parameters of the underlying distributions
  and the forward function which is called whenver data is passed to the
  module. Each of the parameters are nn.Parameters and torch automatically
  keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.dsm_api.DeepSurvivalMachines` !!!

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  activation: str
      Choice of activation function for the MLP representation.
      One of 'ReLU6', 'ReLU' or 'SeLU'.
      Default is 'ReLU6'.
  dist: str
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
  """

  def __init__(self, inputdim, k, layers=None, init=False, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam'):
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer

    if layers is None:
      layers = []
    self.layers = layers

    if self.dist == 'Weibull':
      self.act = nn.SELU()
      self.scale = nn.Parameter(-torch.ones(k))
      self.shape = nn.Parameter(-torch.ones(k))

    elif self.dist == 'LogNormal':
      self.act = nn.Tanh()
      self.scale = nn.Parameter(torch.ones(k))
      self.shape = nn.Parameter(torch.ones(k))

    self.embedding = create_representation(inputdim, layers, 'ReLU6')

    if len(layers) == 0:
      self.gate = nn.Sequential(nn.Linear(inputdim, k, bias=False))
      self.scaleg = nn.Sequential(nn.Linear(inputdim, k, bias=True))
      self.shapeg = nn.Sequential(nn.Linear(inputdim, k, bias=True))

    else:
      self.gate = nn.Sequential(nn.Linear(layers[-1], k, bias=False))
      self.scaleg = nn.Sequential(nn.Linear(layers[-1], k, bias=True))
      self.shapeg = nn.Sequential(nn.Linear(layers[-1], k, bias=True))

    if init is not False:
      self.shape.data.fill_(init[0])
      self.scale.data.fill_(init[1])

  def forward(self, x):
    """The forward function that is called when data is passed through DSM.

    Args:
      x:
        a torch.tensor of the input features.
    """
    xrep = self.embedding(x)
    return(self.act(self.shapeg(xrep))+self.shape.expand(x.shape[0], -1),
           self.act(self.scaleg(xrep))+self.scale.expand(x.shape[0], -1),
           self.gate(xrep)/self.temp)

  def get_shape_scale(self):
    return(self.shape,
           self.scale)
