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


"""Loss function definitions for the Deep Survival Machines model

In this module we define the various losses for the censored and uncensored
instances of data corresponding to Weibull and LogNormal distributions.
These losses are optimized when training DSM.

.. todo::
  Use torch.distributions
.. warning::
  NOT DESIGNED TO BE CALLED DIRECTLY!!!

"""

import numpy as np
import torch
import torch.nn as nn


def _lognormal_loss(model, t, e):

  shape, scale = model.get_shape_scale()

  k_ = shape.expand(t.shape[0], -1)
  b_ = scale.expand(t.shape[0], -1)

  ll = 0.
  for g in range(model.k):

    mu = k_[:, g]
    sigma = b_[:, g]

    f = - sigma - 0.5*np.log(2*np.pi)
    f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
    s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
    s = 0.5 - 0.5*torch.erf(s)
    s = torch.log(s)

    uncens = np.where(e == 1)[0]
    cens = np.where(e == 0)[0]
    ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def _weibull_loss(model, t, e):

  shape, scale = model.get_shape_scale()

  k_ = shape.expand(t.shape[0], -1)
  b_ = scale.expand(t.shape[0], -1)

  ll = 0.
  for g in range(model.k):

    k = k_[:, g]
    b = b_[:, g]

    s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
    f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
    f = f + s

    uncens = np.where(e.cpu().data.numpy() == 1)[0]
    cens = np.where(e.cpu().data.numpy() == 0)[0]
    ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def unconditional_loss(model, t, e):

  if model.dist == 'Weibull':
    return _weibull_loss(model, t, e)

  elif model.dist == 'LogNormal':
    return _lognormal_loss(model, t, e)

def _conditional_lognormal_loss(model, x, t, e, elbo=True):

  alpha = model.discount
  shape, scale, logits = model.forward(x)

  lossf = []
  losss = []

  k_ = shape
  b_ = scale

  for g in range(model.k):

    mu = k_[:, g]
    sigma = b_[:, g]

    f = - sigma - 0.5*np.log(2*np.pi)
    f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
    s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
    s = 0.5 - 0.5*torch.erf(s)
    s = torch.log(s)

    lossf.append(f)
    losss.append(s)

  losss = torch.stack(losss, dim=1)
  lossf = torch.stack(lossf, dim=1)

  if elbo:

    lossg = nn.Softmax(dim=1)(logits)
    losss = lossg*losss
    lossf = lossg*lossf

    losss = losss.sum(dim=1)
    lossf = lossf.sum(dim=1)

  else:

    lossg = nn.LogSoftmax(dim=1)(logits)
    losss = lossg + losss
    lossf = lossg + lossf

    losss = torch.logsumexp(losss, dim=1)
    lossf = torch.logsumexp(lossf, dim=1)

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() == 0)[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/x.shape[0]


def _conditional_weibull_loss(model, x, t, e, elbo=True):

  alpha = model.discount
  shape, scale, logits = model.forward(x)

  k_ = shape
  b_ = scale

  lossf = []
  losss = []

  for g in range(model.k):

    k = k_[:, g]
    b = b_[:, g]

    s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
    f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
    f = f + s

    lossf.append(f)
    losss.append(s)

  losss = torch.stack(losss, dim=1)
  lossf = torch.stack(lossf, dim=1)

  if elbo:

    lossg = nn.Softmax(dim=1)(logits)
    losss = lossg*losss
    lossf = lossg*lossf
    losss = losss.sum(dim=1)
    lossf = lossf.sum(dim=1)

  else:

    lossg = nn.LogSoftmax(dim=1)(logits)
    losss = lossg + losss
    lossf = lossg + lossf
    losss = torch.logsumexp(losss, dim=1)
    lossf = torch.logsumexp(lossf, dim=1)

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() == 0)[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/x.shape[0]


def conditional_loss(model, x, t, e, elbo=True):

  if model.dist == 'Weibull':
    return _conditional_weibull_loss(model, x, t, e, elbo)

  elif model.dist == 'LogNormal':
    return _conditional_lognormal_loss(model, x, t, e, elbo)


def _weibull_cdf(model, x, t_horizon):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double()
  t_horz = t_horz.repeat(x.shape[0], 1)

  cdfs = []
  for j in range(len(t_horizon)):

    t = t_horz[:, j]
    lcdfs = []

    for g in range(model.k):

      k = k_[:, g]
      b = b_[:, g]
      s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
      lcdfs.append(s)

    lcdfs = torch.stack(lcdfs, dim=1)
    lcdfs = lcdfs+logits
    lcdfs = torch.logsumexp(lcdfs, dim=1)
    cdfs.append(lcdfs.detach().numpy())

  return cdfs


def _lognormal_cdf(model, x, t_horizon):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double()
  t_horz = t_horz.repeat(x.shape[0], 1)

  cdfs = []

  for j in range(len(t_horizon)):

    t = t_horz[:, j]
    lcdfs = []

    for g in range(model.k):

      mu = k_[:, g]
      sigma = b_[:, g]

      s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
      s = 0.5 - 0.5*torch.erf(s)
      s = torch.log(s)
      lcdfs.append(s)

    lcdfs = torch.stack(lcdfs, dim=1)
    lcdfs = lcdfs+logits
    lcdfs = torch.logsumexp(lcdfs, dim=1)
    cdfs.append(lcdfs.detach().numpy())

  return cdfs


def predict_cdf(model, x, t_horizon):
  torch.no_grad()
  if model.dist == 'Weibull':
    return _weibull_cdf(model, x, t_horizon)

  if model.dist == 'LogNormal':
    return _lognormal_cdf(model, x, t_horizon)
