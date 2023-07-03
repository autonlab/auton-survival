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

def _normal_loss(model, t, e, risk='1'):

  shape, scale = model.get_shape_scale(risk)

  k_ = shape.expand(t.shape[0], -1)
  b_ = scale.expand(t.shape[0], -1)

  ll = 0.
  for g in range(model.k):

    mu = k_[:, g]
    sigma = b_[:, g]

    f = - sigma - 0.5*np.log(2*np.pi)
    f = f - 0.5*torch.div((t - mu)**2, torch.exp(2*sigma))
    s = torch.div(t - mu, torch.exp(sigma)*np.sqrt(2))
    s = 0.5 - 0.5*torch.erf(s)
    s = torch.log(s)

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def _lognormal_loss(model, t, e, risk='1'):

  shape, scale = model.get_shape_scale(risk)

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

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def _weibull_loss(model, t, e, risk='1'):

  shape, scale = model.get_shape_scale(risk)

  k_ = shape.expand(t.shape[0], -1)
  b_ = scale.expand(t.shape[0], -1)

  ll = 0.
  for g in range(model.k):

    k = k_[:, g]
    b = b_[:, g]

    s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
    f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
    f = f + s

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def unconditional_loss(model, t, e, risk='1'):

  if model.dist == 'Weibull':
    return _weibull_loss(model, t, e, risk)
  elif model.dist == 'LogNormal':
    return _lognormal_loss(model, t, e, risk)
  elif model.dist == 'Normal':
    return _normal_loss(model, t, e, risk)
  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')

def _conditional_normal_loss(model, x, t, e, elbo=True, risk='1'):

  alpha = model.discount
  shape, scale, logits = model.forward(x, risk)

  lossf = []
  losss = []

  k_ = shape
  b_ = scale

  for g in range(model.k):

    mu = k_[:, g]
    sigma = b_[:, g]

    f = - sigma - 0.5*np.log(2*np.pi)
    f = f - 0.5*torch.div((t - mu)**2, torch.exp(2*sigma))
    s = torch.div(t - mu, torch.exp(sigma)*np.sqrt(2))
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

  uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
  cens = np.where(e.cpu().data.numpy() != int(risk))[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))

def _conditional_lognormal_loss(model, x, t, e, elbo=True, risk='1'):

  alpha = model.discount
  shape, scale, logits = model.forward(x, risk)

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

  uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
  cens = np.where(e.cpu().data.numpy() != int(risk))[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))


def _conditional_weibull_loss(model, x, t, e, elbo=True, risk='1'):

  alpha = model.discount
  shape, scale, logits = model.forward(x, risk)

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

  uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
  cens = np.where(e.cpu().data.numpy() != int(risk))[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))


def conditional_loss(model, x, t, e, elbo=True, risk='1'):

  if model.dist == 'Weibull':
    return _conditional_weibull_loss(model, x, t, e, elbo, risk)
  elif model.dist == 'LogNormal':
    return _conditional_lognormal_loss(model, x, t, e, elbo, risk)
  elif model.dist == 'Normal':
    return _conditional_normal_loss(model, x, t, e, elbo, risk)
  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')

def _weibull_pdf(model, x, t_horizon, risk='1'):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x, risk)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double().to(logits.device)
  t_horz = t_horz.repeat(shape.shape[0], 1)

  pdfs = []
  for j in range(len(t_horizon)):

    t = t_horz[:, j]
    lpdfs = []

    for g in range(model.k):

      k = k_[:, g]
      b = b_[:, g]
      s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
      f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
      f = f + s
      lpdfs.append(f)

    lpdfs = torch.stack(lpdfs, dim=1)
    lpdfs = lpdfs+logits
    lpdfs = torch.logsumexp(lpdfs, dim=1)
    pdfs.append(lpdfs.detach().cpu().numpy())

  return pdfs

def _weibull_cdf(model, x, t_horizon, risk='1'):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x, risk)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double().to(logits.device)
  t_horz = t_horz.repeat(shape.shape[0], 1)

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
    cdfs.append(lcdfs.detach().cpu().numpy())

  return cdfs

def _weibull_mean(model, x, risk='1'):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x, risk)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  lmeans = []

  for g in range(model.k):

    k = k_[:, g]
    b = b_[:, g]

    one_over_k = torch.reciprocal(torch.exp(k))
    lmean = -(one_over_k*b) + torch.lgamma(1+one_over_k)
    lmeans.append(lmean)

  lmeans = torch.stack(lmeans, dim=1)
  lmeans = lmeans+logits
  lmeans = torch.logsumexp(lmeans, dim=1)

  return torch.exp(lmeans).detach().numpy()




def _lognormal_cdf(model, x, t_horizon, risk='1'):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x, risk)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double().to(logits.device)
  t_horz = t_horz.repeat(shape.shape[0], 1)

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
    cdfs.append(lcdfs.detach().cpu().numpy())

  return cdfs

def _normal_cdf(model, x, t_horizon, risk='1'):

  squish = nn.LogSoftmax(dim=1)

  shape, scale, logits = model.forward(x, risk)
  logits = squish(logits)

  k_ = shape
  b_ = scale

  t_horz = torch.tensor(t_horizon).double().to(logits.device)
  t_horz = t_horz.repeat(shape.shape[0], 1)

  cdfs = []

  for j in range(len(t_horizon)):

    t = t_horz[:, j]
    lcdfs = []

    for g in range(model.k):

      mu = k_[:, g]
      sigma = b_[:, g]

      s = torch.div(t - mu, torch.exp(sigma)*np.sqrt(2))
      s = 0.5 - 0.5*torch.erf(s)
      s = torch.log(s)
      lcdfs.append(s)

    lcdfs = torch.stack(lcdfs, dim=1)
    lcdfs = lcdfs+logits
    lcdfs = torch.logsumexp(lcdfs, dim=1)
    cdfs.append(lcdfs.detach().cpu().numpy())

  return cdfs

def _normal_mean(model, x, risk='1'):

  squish = nn.Softmax(dim=1)
  shape, scale, logits = model.forward(x, risk)

  logits = squish(logits)
  k_ = shape
  b_ = scale

  lmeans = []
  for g in range(model.k):

    mu = k_[:, g]
    sigma = b_[:, g]
    lmeans.append(mu)

  lmeans = torch.stack(lmeans, dim=1)
  lmeans = lmeans*logits
  lmeans = torch.sum(lmeans, dim=1)

  return lmeans.detach().cpu().numpy()


def predict_mean(model, x, risk='1'):
  torch.no_grad()
  if model.dist == 'Normal':
    return _normal_mean(model, x, risk)
  elif model.dist == 'Weibull':
    return _weibull_mean(model, x, risk)
  else:
    raise NotImplementedError('Mean of Distribution: '+model.dist+
                              ' not implemented yet.')


def predict_pdf(model, x, t_horizon, risk='1'):
  torch.no_grad()
  if model.dist == 'Weibull':
    return _weibull_pdf(model, x, t_horizon, risk)
  # if model.dist == 'LogNormal':
  #   return _lognormal_pdf(model, x, t_horizon, risk)
  # if model.dist == 'Normal':
  #   return _normal_pdf(model, x, t_horizon, risk)
  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')


def predict_cdf(model, x, t_horizon, risk='1'):
  torch.no_grad()
  if model.dist == 'Weibull':
    return _weibull_cdf(model, x, t_horizon, risk)
  if model.dist == 'LogNormal':
    return _lognormal_cdf(model, x, t_horizon, risk)
  if model.dist == 'Normal':
    return _normal_cdf(model, x, t_horizon, risk)
  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')
