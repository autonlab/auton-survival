# coding=utf-8
# Copyright 2020 Chirag Nagpal
#
# This file is part of Deep Survival Machines.

# Deep Survival Machines is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Deep Survival Machines is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

"""Utility functions to train the Deep Survival Machines models"""

from dsm.losses import unconditional_loss, conditional_loss

from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np

import gc

from dsm.dsm_torch import DeepSurvivalMachinesTorch

def get_optimizer(model, lr):

  if model.optimizer == 'Adam':
    return torch.optim.Adam(model.parameters(), lr=lr)
  elif model.optimizer == 'SGD':
    return torch.optim.SGD(model.parameters(), lr=lr)
  elif model.optimizer == 'RMSProp':
    return torch.optim.RMSprop(model.parameters(), lr=lr)
  else:
    raise NotImplementedError("Optimizer "+model.optimizer+
                              " is not implemented")
    
def pretrain_dsm(model, t_train, e_train, t_valid, e_valid,
                 n_iter=10000, lr=1e-2, thres=1e-4):

  premodel = DeepSurvivalMachinesTorch(1, 1,
                                       init=False, dist=model.dist)
  premodel.double()

  optimizer = torch.optim.Adam(premodel.parameters(), lr=lr)
  oldcost = -float('inf')
  patience = 0

  costs = []
  for _ in tqdm(range(n_iter)):

    optimizer.zero_grad()

    loss = unconditional_loss(premodel, t_train, e_train)
    loss.backward()
    optimizer.step()

    valid_loss = unconditional_loss(premodel, t_valid, e_valid)
    valid_loss = valid_loss.detach().cpu().numpy()

    costs.append(valid_loss)

    if np.abs(costs[-1] - oldcost) < thres:
      patience += 1
      if patience == 3:
        break
    oldcost = costs[-1]

  return premodel


def train_dsm(model,
              x_train, t_train, e_train,
              x_valid, t_valid, e_valid,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100):

  print('Pretraining the Underlying Distributions...')

  premodel = pretrain_dsm(model,
                          t_train,
                          e_train,
                          t_valid,
                          e_valid,
                          n_iter=10000,
                          lr=1e-2,
                          thres=1e-4)
  model.shape.data.fill_(float(premodel.shape))
  model.scale.data.fill_(float(premodel.scale))

  # print(premodel.shape, premodel.scale)
  # print(model.shape, model.scale)

  # init=(float(premodel.shape[0]),
  # float(premodel.scale[0])),
  # print(torch.exp(-premodel.scale).cpu().data.numpy()[0],
  #       torch.exp(premodel.shape).cpu().data.numpy()[0])

  model.double()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  patience = 0
  oldcost = float('inf')

  nbatches = int(x_train.shape[0]/bs)+1

  dics = []
  costs = []
  i = 0
  for i in tqdm(range(n_iter)):
    for j in range(nbatches):

      optimizer.zero_grad()
      loss = conditional_loss(model,
                              x_train[j*bs:(j+1)*bs],
                              t_train[j*bs:(j+1)*bs],
                              e_train[j*bs:(j+1)*bs],
                              elbo=elbo)
      loss.backward()
      optimizer.step()

    valid_loss = conditional_loss(model,
                                  x_valid,
                                  t_valid,
                                  e_valid,
                                  elbo=False)

    valid_loss = valid_loss.detach().cpu().numpy()
    costs.append(float(valid_loss))
    dics.append(deepcopy(model.state_dict()))

    if (costs[-1] >= oldcost) is True:
      if patience == 2:
        maxm = np.argmax(costs)
        model.load_state_dict(dics[maxm])

        del dics
        gc.collect()
        return model, i
      else:
        patience += 1
    else:
      patience = 0

    oldcost = costs[-1]

  return model, i
