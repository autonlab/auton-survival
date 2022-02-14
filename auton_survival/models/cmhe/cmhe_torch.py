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
from dsm.dsm_torch import create_representation

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

    self.expert = torch.nn.Linear(lastdim, self.k, bias=False)
    self.z_gate = torch.nn.Linear(lastdim, self.k, bias=False)
    self.phi_gate = torch.nn.Linear(lastdim, self.g, bias=False)
    self.omega = torch.nn.Parameter(torch.rand(self.g)-0.5)

  def __init__(self, k, g, inputdim, layers=None, optimizer='Adam'):

    super(DeepCMHETorch, self).__init__()

    assert isinstance(k, int)

    if layers is None: layers = []

    self.optimizer = optimizer

    self.k = k # Base Physiology groups
    self.g = g # Treatment Effect groups

    if len(layers) == 0: lastdim = inputdim
    else: lastdim = layers[-1]

    self._init_dcmhe_layers(lastdim)

    self.embedding = create_representation(inputdim, layers, 'Tanh')


  def forward(self, x, a):

    x = self.embedding(x)
    a = 2*(a-0.5)

    log_hrs = torch.clamp(self.expert(x), min=-100, max=100)

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

# class DeepCoxMixtureHETorch(CoxMixtureHETorch):

#   def __init__(self, k, g, inputdim, hidden):

#     super(DeepCoxMixtureHETorch, self).__init__(k, g, inputdim, hidden)

#     # Get rich feature representations of the covariates
#     self.embedding = torch.nn.Sequential(torch.nn.Linear(inputdim, hidden),
#                                          torch.nn.Tanh(),
#                                          torch.nn.Linear(hidden, hidden),
#                                          torch.nn.Tanh())
