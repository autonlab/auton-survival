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

import torch
from auton_survival.models.dsm.dsm_torch import create_representation

class DeepCMHETorch(torch.nn.Module):
  """PyTorch model definition of the Cox Mixture with Hereogenous Effects Model.

  Cox Mixtures with Heterogenous Effects involves the assuming that the
  base survival rates are independent of the treatment effect.
  of the individual to be a mixture of K Cox Models. Conditioned on each
  subgroup Z=k; the PH assumptions are assumed to hold and the baseline
  hazard rates is determined non-parametrically using an spline-interpolated
  Breslow's estimator.

  """

  def _init_dcmhe_layers(self, lastdim):


    self.expert = IdentifiableLinear(lastdim, self.k, bias=False)
    self.z_gate = IdentifiableLinear(lastdim, self.k, bias=False)
    self.phi_gate = IdentifiableLinear(lastdim, self.g, bias=False)
    # self.expert = torch.nn.Linear(lastdim, self.k, bias=False)
    # self.z_gate = torch.nn.Linear(lastdim, self.k, bias=False)
    # self.phi_gate = torch.nn.Linear(lastdim, self.g, bias=False)
    self.omega = torch.nn.Parameter(torch.rand(self.g)-0.5)

  def __init__(self, k, g, inputdim, layers=None, gamma=100,
               smoothing_factor=1e-4, gate_l2_penalty=1e-4, 
               optimizer='Adam'):

    super(DeepCMHETorch, self).__init__()

    assert isinstance(k, int)

    if layers is None: layers = []

    self.optimizer = optimizer

    self.k = k # Base Physiology groups
    self.g = g # Treatment Effect groups

    self.gamma = gamma
    self.smoothing_factor = smoothing_factor
  
    if len(layers) == 0: lastdim = inputdim
    else: lastdim = layers[-1]

    self._init_dcmhe_layers(lastdim)

    self.gate_l2_penalty = gate_l2_penalty 

    self.embedding = create_representation(inputdim, layers, 'Tanh')


  def forward(self, x, a):

    x = self.embedding(x)
    a = 2*(a-0.5)

    log_hrs = torch.clamp(self.expert(x),
                          min=-self.gamma,
                          max=self.gamma)

    logp_z_gate = torch.nn.LogSoftmax(dim=1)(self.z_gate(x)) #
    logp_phi_gate = torch.nn.LogSoftmax(dim=1)(self.phi_gate(x))

    logp_jointlatent_gate = torch.zeros(len(x), self.k, self.g)

    for i in range(self.k):
      for j in range(self.g):
        logp_jointlatent_gate[:, i, j] = logp_z_gate[:, i] + logp_phi_gate[:, j]

    logp_joint_hrs = torch.zeros(len(x), self.k, self.g)

    for i in range(self.k):
      for j in range(self.g):
        logp_joint_hrs[:, i, j] = log_hrs[:, i] + (j!=2)*a*self.omega[j]

    return logp_jointlatent_gate, logp_joint_hrs

class IdentifiableLinear(torch.nn.Module):

  """
  Softmax and LogSoftmax with K classes in pytorch are over specfied and lead to
  issues of mis-identifiability. This class is a wrapper for linear layers that 
  are correctly specified with K-1 columns. The output of this layer for the Kth
  class is all zeros. This allows direct application of pytorch.nn.LogSoftmax
  and pytorch.nn.Softmax.
  """

  def __init__(self, in_features, out_features, bias=True):
  
    super(IdentifiableLinear, self).__init__()

    assert out_features>0; "Output features must be greater than 0"

    self.out_features = out_features
    self.in_features = in_features
    self.linear = torch.nn.Linear(in_features, max(out_features-1, 1), bias=bias)

  @property
  def weight(self):
    return self.linear.weight

  def forward(self, x):
    if self.out_features == 1:
      return self.linear(x).reshape(-1, 1)
    else:
      zeros = torch.zeros(len(x), 1, device=x.device)
      return torch.cat([self.linear(x), zeros], dim=1)