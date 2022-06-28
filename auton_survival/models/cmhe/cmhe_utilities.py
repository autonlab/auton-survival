
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
import numpy as np

from scipy.interpolate import UnivariateSpline
from sksurv.linear_model.coxph import BreslowEstimator

from tqdm import tqdm

def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)

def partial_ll_loss(lrisks, tb, eb, eps=1e-2):

  tb = tb + eps*np.random.random(len(tb))
  sindex = np.argsort(-tb)

  tb = tb[sindex]
  eb = eb[sindex]

  lrisks = lrisks[sindex] # lrisks = tf.gather(lrisks, sindex)

  lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)
  plls = lrisks - lrisksdenom
  pll = plls[eb == 1]

  pll = torch.sum(pll) # pll = tf.reduce_sum(pll)

  return -pll

def fit_spline(t, surv, smoothing_factor=1e-4):
  return UnivariateSpline(t, surv, s=smoothing_factor, ext=3)

def smooth_bl_survival(breslow, smoothing_factor):

  blsurvival = breslow.baseline_survival_
  x, y = blsurvival.x, blsurvival.y
  return fit_spline(x, y, smoothing_factor=smoothing_factor)

def get_probability_(lrisks, ts, spl):
  risks = np.exp(lrisks)
  s0ts = (-risks)*(spl(ts)**(risks-1))
  return s0ts * spl.derivative()(ts)

def get_survival_(lrisks, ts, spl):
  risks = np.exp(lrisks)
  return spl(ts)**risks

def get_probability(lrisks, breslow_splines, t):
  psurv = []
  for i in range(lrisks.shape[1]):
    p = get_probability_(lrisks[:, i], t, breslow_splines[i])
    psurv.append(p)
  psurv = np.array(psurv).T
  return psurv

def get_survival(lrisks, breslow_splines, t):
  psurv = []
  for i in range(lrisks.shape[1]):
    p = get_survival_(lrisks[:, i], t, breslow_splines[i])
    psurv.append(p)
  psurv = np.array(psurv).T
  return psurv

def get_posteriors(probs):
  probs_ = probs+1e-8
  return probs-torch.logsumexp(probs, dim=1).reshape(-1,1)

def get_hard_z(gates_prob):
  return torch.argmax(gates_prob, dim=1)

def sample_hard_z(gates_prob):
  return torch.multinomial(gates_prob.exp(), num_samples=1)[:, 0]

def repair_probs(probs):
  probs[torch.isnan(probs)] = -20
  probs[probs<-20] = -20
  return probs

def get_likelihood(model, breslow_splines, x, t, e, a):

  # Function requires numpy/torch

  gates, lrisks = model(x, a=a)
  lrisks = lrisks.numpy()
  e, t = e.numpy(), t.numpy()

  probs = []

  for i in range(model.g):

    survivals = get_survival(lrisks[:, :, i], breslow_splines, t)
    probability = get_probability(lrisks[:, :, i], breslow_splines, t)

    event_probs = np.array([survivals,  probability])
    event_probs = event_probs[e.astype('int'), range(len(e)), :]
    probs.append(np.log(event_probs))

  probs = np.array(probs).transpose(1, 2, 0)
  event_probs = gates+probs

  return event_probs

def q_function(model, x, t, e, a, log_likelihoods, typ='soft'):

  z_posteriors = repair_probs(
                    get_posteriors(
                      torch.logsumexp(log_likelihoods, dim=2)))
  zeta_posteriors = repair_probs(
                      get_posteriors(
                        torch.logsumexp(log_likelihoods, dim=1)))

  if typ == 'hard':
    z = get_hard_z(z_posteriors)
    zeta = get_hard_z(zeta_posteriors)
  else:
    z = sample_hard_z(z_posteriors)
    zeta = sample_hard_z(zeta_posteriors)

  gates, lrisks = model(x, a=a)

  loss = 0
  for i in range(model.k):
    lrisks_ = lrisks[:, i, :][range(len(zeta)), zeta]
    loss += partial_ll_loss(lrisks_[z == i], t[z == i], e[z == i])

  #log_smax_loss = -torch.nn.LogSoftmax(dim=1)(gates) # tf.nn.log_softmax(gates)

  posteriors = repair_probs(
                get_posteriors(
                  log_likelihoods.reshape(-1, model.k*model.g))).exp()

  gate_loss = posteriors*gates.reshape(-1, model.k*model.g)
  gate_loss = -torch.sum(gate_loss)
  loss+=gate_loss

  return loss

def e_step(model, breslow_splines, x, t, e, a):

  # TODO: Do this in `Log Space`
  # If Breslow splines are not available, like in the first
  # iteration of learning, we randomly compute posteriors.
  if breslow_splines is None: log_likelihoods = torch.rand(len(x), model.k, model.g)
  else: log_likelihoods = get_likelihood(model, breslow_splines, x, t, e, a)

  return log_likelihoods

def m_step(model, optimizer, x, t, e, a, log_likelihoods, typ='soft'):

  optimizer.zero_grad()
  loss = q_function(model, x, t, e, a, log_likelihoods, typ)
  gate_regularization_loss = (model.phi_gate.weight**2).sum()
  gate_regularization_loss += (model.z_gate.weight**2).sum()
  loss += (model.gate_l2_penalty)*gate_regularization_loss
  loss.backward()
  optimizer.step()

  return float(loss)

def fit_breslow(model, x, t, e, a, log_likelihoods=None, smoothing_factor=1e-4, typ='soft'):

  gates, lrisks = model(x, a=a)

  lrisks = lrisks.numpy()

  e = e.numpy()
  t = t.numpy()

  if log_likelihoods is None:
    z_posteriors = torch.logsumexp(gates, dim=2)
    zeta_posteriors = torch.logsumexp(gates, dim=1)
  else:
    z_posteriors = repair_probs(get_posteriors(torch.logsumexp(log_likelihoods, dim=2)))
    zeta_posteriors = repair_probs(get_posteriors(torch.logsumexp(log_likelihoods, dim=1)))

  if typ == 'soft':
    z = sample_hard_z(z_posteriors)
    zeta = sample_hard_z(zeta_posteriors)
  else:
    z = get_hard_z(z_posteriors)
    zeta = get_hard_z(zeta_posteriors)

  breslow_splines = {}
  for i in range(model.k):
    breslowk = BreslowEstimator().fit(lrisks[:, i, :][range(len(zeta)), zeta][z==i], e[z==i], t[z==i])
    breslow_splines[i] = smooth_bl_survival(breslowk, smoothing_factor=smoothing_factor)

  return breslow_splines


def train_step(model, x, t, e, a, breslow_splines, optimizer,
               bs=256, seed=100, typ='soft', use_posteriors=False,
               update_splines_after=10, smoothing_factor=1e-4):

  from sklearn.utils import shuffle

  x, t, e, a = shuffle(x, t, e, a, random_state=seed)

  n = x.shape[0]
  batches = (n // bs) + 1

  epoch_loss = 0
  for i in range(batches):

    xb = x[i*bs:(i+1)*bs]
    tb = t[i*bs:(i+1)*bs]
    eb = e[i*bs:(i+1)*bs]
    ab = a[i*bs:(i+1)*bs]

    # E-Step !!!
    # e_step_start = time.time()
    with torch.no_grad():
      log_likelihoods = e_step(model, breslow_splines, xb, tb, eb, ab)

    torch.enable_grad()
    loss = m_step(model, optimizer, xb, tb, eb, ab, log_likelihoods, typ=typ)
    epoch_loss += loss

    with torch.no_grad():
      if i%update_splines_after == 0:
        if use_posteriors:
          log_likelihoods = e_step(model, breslow_splines, x, t, e, a)
          breslow_splines = fit_breslow(model, x, t, e, a,
                                        log_likelihoods=log_likelihoods,
                                        typ='soft',
                                        smoothing_factor=smoothing_factor)
        else:
          breslow_splines = fit_breslow(model, x, t, e, a,
                                        log_likelihoods=None,
                                        typ='soft',
                                        smoothing_factor=smoothing_factor)
          # print(f'Duration of Breslow spline estimation: {time.time() - estimate_breslow_start}')
      # except Exception as exce:
      #   print("Exception!!!:", exce)
      #   logging.warning("Couldn't fit splines, reusing from previous epoch")
  #print (epoch_loss/n)
  return breslow_splines


def test_step(model, x, t, e, a, breslow_splines, loss='q', typ='soft'):

  if loss == 'q':
    with torch.no_grad():
      posteriors = e_step(model, breslow_splines, x, t, e, a)
      loss = q_function(model, x, t, e, a, posteriors, typ=typ)

  return float(loss/x.shape[0])


def train_cmhe(model, train_data, val_data, epochs=50,
               patience=2, vloss='q', bs=256, typ='soft', lr=1e-3,
               use_posteriors=False, debug=False,
               return_losses=False, update_splines_after=10,
               smoothing_factor=1e-4, random_seed=0):

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  if val_data is None: val_data = train_data

  xt, tt, et, at = train_data
  xv, tv, ev, av = val_data

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  valc = np.inf
  patience_ = 0

  breslow_splines = None

  losses = []

  for epoch in tqdm(range(epochs)):

    # train_step_start = time.time()
    breslow_splines = train_step(model, xt, tt, et, at, breslow_splines,
                                 optimizer, bs=bs, seed=epoch, typ=typ,
                                 use_posteriors=use_posteriors,
                                 update_splines_after=update_splines_after,
                                 smoothing_factor=smoothing_factor)
    # print(f'Duration of train-step: {time.time() - train_step_start}')
    # test_step_start = time.time()
    valcn = test_step(model, xv, tv, ev, av, breslow_splines,
                      loss=vloss, typ=typ)
    # print(f'Duration of test-step: {time.time() - test_step_start}')

    losses.append(valcn)

    if epoch % 1 == 0:
      if debug: print(patience_, epoch, valcn)

    if valcn > valc:
      patience_ += 1
    else:
      patience_ = 0

    if patience_ == patience:
        if return_losses: return (model, breslow_splines), losses
        else: return (model, breslow_splines)

    valc = valcn

  if return_losses: return (model, breslow_splines), losses
  else: return (model, breslow_splines)

def predict_survival(model, x, a, t):

  if isinstance(t, (int, float)): t = [t]

  model, breslow_splines = model
  
  gates, lrisks = model(x, a=a)

  lrisks = lrisks.detach().numpy()
  gates = gates.exp().reshape(-1, model.k*model.g).detach().numpy()

  predictions = []
  for t_ in t:
    expert_outputs = []
    for i in range(model.g):
      expert_output = get_survival(lrisks[:, :, i], breslow_splines, t_)
      expert_outputs.append(expert_output)
    expert_outputs = np.array(expert_outputs).transpose(1, 2, 0).reshape(-1, model.k*model.g)

    predictions.append((gates*expert_outputs).sum(axis=1))
  return np.array(predictions).T

def predict_latent_z(model, x):

  model, _ = model
  gates = model.model.embedding(x)

  z_gate_probs = torch.exp(gates).sum(axis=2).detach().numpy()

  return z_gate_probs

def predict_latent_phi(model, x):

  model, _ = model
  x = model.embedding(x)

  p_phi_gate = torch.nn.Softmax(dim=1)(model.phi_gate(x)).detach().numpy()

  return p_phi_gate
