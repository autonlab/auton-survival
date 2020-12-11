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


"""Torch model definitons for the Deep Survival Machines model

This includes definitons for the Torch Deep Survival Machines module.
The main interface is the DeepSurvivalMachines class which inherits
from torch.nn.Module.

Note: NOT DESIGNED TO BE CALLED DIRECTLY!!!

"""


import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

__pdoc__ = {}

for clsn in ['DeepSurvivalMachinesTorch', 
             'DeepRecurrentSurvivalMachinesTorch']:
  for membr in ['training', 'dump_patches']:

    __pdoc__[clsn+'.'+membr] = False


def create_representation(inputdim, layers, activation):
  r"""Helper function to generate the representation function for DSM.

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

  def __init__(self, inputdim, k, layers=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam',
               risks=1):
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.risks = risks

    if layers is None:
      layers = []
    self.layers = layers

    if self.dist in ['Weibull']:
      self.act = nn.SELU()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1):nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['Normal']:
      self.act = nn.Identity()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['LogNormal']:
      self.act = nn.Tanh()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    else:
      raise NotImplementedError('Distribution: '+self.dist+' not implemented'+
                                ' yet.')

    self.embedding = create_representation(inputdim, layers, 'ReLU6')

    if len(layers) == 0:
      lastdim = inputdim
    else:
      lastdim = layers[-1]

    self.gate = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, k, bias=False)
        ) for r in range(self.risks)})

    self.scaleg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, k, bias=True)
        ) for r in range(self.risks)})

    self.shapeg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, k, bias=True)
        ) for r in range(self.risks)})


  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
    xrep = self.embedding(x)
    dim = x.shape[0]
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])

class DeepRecurrentSurvivalMachinesTorch(nn.Module):
  """A Torch implementation of Deep Recurrent Survival Machines model.

  This is an implementation of Deep Recurrent Survival Machines model
  in torch. It inherits from `DeepSurvivalMachinesTorch` and replaces the
  input representation learning MLP with an LSTM or RNN, the parameters of the
  underlying distributions and the forward function which is called whenever
  data is passed to the module. Each of the parameters are nn.Parameters and
  torch automatically keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.dsm_api.DeepRecurrentSurvivalMachines`!!

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: int
      The number of hidden layers in the LSTM or RNN cell.
  hidden: int
      The number of neurons in each hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
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

  def __init__(self, inputdim, k, typ='LSTM', layers=1,
               hidden=None, dist='Weibull',
               temp=1000., discount=1.0,
               optimizer='Adam', risks=1):
    super(DeepRecurrentSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.hidden = hidden
    self.layers = layers
    self.typ = typ
    self.risks = risks

    if self.dist in ['Weibull']:
      self.act = nn.SELU()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1):nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['Normal']:
      self.act = nn.Identity()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['LogNormal']:
      self.act = nn.Tanh()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    else:
      raise NotImplementedError('Distribution: '+self.dist+' not implemented'+
                                ' yet.')

    self.gate = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=False)
        ) for r in range(self.risks)})

    self.scaleg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=True)
        ) for r in range(self.risks)})

    self.shapeg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=True)
        ) for r in range(self.risks)})

    if self.typ == 'LSTM':
      self.embedding = nn.LSTM(inputdim, hidden, layers,
                               bias=False, batch_first=True)
    if self.typ == 'RNN':
      self.embedding = nn.RNN(inputdim, hidden, layers,
                              bias=False, batch_first=True,
                              nonlinearity='relu')
    if self.typ == 'GRU':
      self.embedding = nn.GRU(inputdim, hidden, layers,
                              bias=False, batch_first=True)

  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Note: As compared to DSM, the input data for DRSM is a tensor. The forward
    function involves unpacking the tensor in-order to directly use the
    DSM loss functions.

    Args:
      x:
        a torch.tensor of the input features.

    """
    x = x.detach().clone()
    inputmask = ~torch.isnan(x[:, :, 0]).reshape(-1)
    x[torch.isnan(x)] = 0
    xrep, _ = self.embedding(x)
    xrep = xrep.contiguous().view(-1, self.hidden)
    xrep = xrep[inputmask]
    xrep = nn.ReLU6()(xrep)
    dim = xrep.shape[0]
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])
           
class DeepConvolutionalSurvivalMachinesTorch(nn.Module):
  """A Torch implementation of Deep Convolutional Survival Machines model.

  This is an implementation of Deep Convolutional Survival Machines model
  in torch. It inherits from `DeepSurvivalMachinesTorch` and replaces the
  input representation learning MLP with an simple convnet, the parameters of the
  underlying distributions and the forward function which is called whenever
  data is passed to the module. Each of the parameters are nn.Parameters and
  torch automatically keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.dsm_api.DeepConvolutionalSurvivalMachines`!!

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: int
      The number of hidden layers in the LSTM or RNN cell.
  hidden: int
      The number of neurons in each hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
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

  def __init__(self, inputdim, k, typ='ResNet', layers=1,
               hidden=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam', risks=1):
    super(DeepConvolutionalSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.hidden = hidden
    self.layers = layers
    self.typ = typ
    self.risks = risks

    if self.dist in ['Weibull']:
      self.act = nn.SELU()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1):nn.Parameter(-torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['Normal']:
      self.act = nn.Identity()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    elif self.dist in ['LogNormal']:
      self.act = nn.Tanh()
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(k))
                                     for r in range(self.risks)})
    else:
      raise NotImplementedError('Distribution: '+self.dist+' not implemented'+
                                ' yet.')

    self.gate = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=False)
        ) for r in range(self.risks)})

    self.scaleg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=True)
        ) for r in range(self.risks)})

    self.shapeg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(hidden, k, bias=True)
        ) for r in range(self.risks)})

    if self.typ == 'ConvNet':
#         self.cnn = torchvision.models.resnet18(pretrained=True).float()
#         self.cnn.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
#         self.linear = torch.nn.Linear(1000, hidden)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, hidden)


  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
#     xrep = self.linear(self.cnn(x))
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    xrep = self.fc3(x)
    
    dim = x.shape[0]
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])
