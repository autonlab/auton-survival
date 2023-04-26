
import logging
from matplotlib.pyplot import get

import torch
import numpy as np

from sksurv.linear_model.coxph import BreslowEstimator

from sklearn.utils import shuffle


from tqdm import tqdm


from auton_survival.models.dsm.utilities import get_optimizer

def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)

def weighted_partial_ll_loss(lrisks, t, e, eps=1e-10, weights=None, type='naive'):

  if weights is None:
    weights = np.zeros_like(t)

  t = t + eps*np.random.random(len(t))

  sidx = np.argsort(-t)

  t, e, lrisks, weights = t[sidx], e[sidx], lrisks[sidx], weights[sidx]

  if type == 'naive':
    lrisksdenom = torch.logcumsumexp(lrisks+weights , dim = 0)
    plls = weights.exp()*(lrisks - lrisksdenom)
  else:
    lrisks = lrisks+weights
    lrisksdenom = torch.logcumsumexp(lrisks , dim = 0)
    plls = (lrisks - lrisksdenom)

  pll = plls[e == 1]
  pll = torch.sum(pll)

  return -pll

# def partial_ll_loss(lrisks, tb, eb, eps=1e-2):

#   tb = tb + eps*np.random.random(len(tb))
#   sindex = np.argsort(-tb)

#   tb = tb[sindex]
#   eb = eb[sindex]

#   lrisks = lrisks[sindex] # lrisks = tf.gather(lrisks, sindex)
#   # lrisksdenom = tf.math.cumulative_logsumexp(lrisks)
#   lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)

#   plls = lrisks - lrisksdenom
#   pll = plls[eb == 1]

#   pll = torch.sum(pll) # pll = tf.reduce_sum(pll)

#   return -pll

def fit_spline(t, surv, s=1e-4):
  # from scipy.interpolate import PchipInterpolator
  # return PchipInterpolator(t, surv)# s=s, ext=3, k=1)


  from scipy.interpolate import UnivariateSpline
  return UnivariateSpline(t, surv, s=s, ext=3, k=1)



def smooth_bl_survival(times, bl_survival, smoothing_factor):

  # blsurvival = breslow.baseline_survival_
  # x, y = blsurvival.x, blsurvival.y
  return fit_spline(times, bl_survival, s=smoothing_factor)

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
  #probs_ = probs+1e-8
  return probs-torch.logsumexp(probs, dim=1).reshape(-1,1)

def get_hard_z(gates_prob):
  return torch.argmax(gates_prob, dim=1)

def sample_hard_z(gates_prob):
  return torch.multinomial(gates_prob.exp(), num_samples=1)[:, 0]

def repair_probs(probs):
  probs[probs>10] = 10
  probs[probs<-10] = -10
  probs[torch.isnan(probs)] = 0

  return probs

def get_likelihood(model, breslow_splines, x, t, e):

  # Function requires numpy/torch

  gates, lrisks = model(x)
  lrisks = lrisks.numpy()
  e, t = e.numpy(), t.numpy()

  survivals = get_survival(lrisks, breslow_splines, t)
  probability = get_probability(lrisks, breslow_splines, t)

  event_probs = np.array([survivals, probability])
  event_probs = event_probs[e.astype('int'), range(len(e)), :]
  #event_probs[event_probs<1e-10] = 1e-10
  probs = gates+np.log(event_probs)
  # else:
  #   gates_prob = torch.nn.Softmax(dim = 1)(gates)
  #   probs = gates_prob*event_probs
  return probs

def q_function(model, x, t, e, posteriors, typ='soft'):

  # if typ == 'hard': z = get_hard_z(posteriors)
  # else: z = sample_hard_z(posteriors)

  gates, lrisks = model(x)

  k = model.k

  loss = 0
  for i in range(k):
    # lrisks_ = lrisks[z == i][:, i]
    #loss += partial_ll_loss(lrisks_, t[z == i], e[z == i])
    loss += weighted_partial_ll_loss(lrisks[:, i], t, e, weights=posteriors[:,i])

  #log_smax_loss = -torch.nn.LogSoftmax(dim=1)(gates) # tf.nn.log_softmax(gates)

  gate_loss = posteriors.exp()*gates
  gate_loss = -torch.sum(gate_loss)
  loss+=gate_loss

  return loss

def e_step(model, breslow_splines, x, t, e):

  # TODO: Do this in `Log Space`
  if breslow_splines is None:
      # If Breslow splines are not available, like in the first
      # iteration of learning, we randomly compute posteriors.
    posteriors = get_posteriors(torch.rand(len(x), model.k))
    pass
  else:
    probs = get_likelihood(model, breslow_splines, x, t, e)
    posteriors = get_posteriors(repair_probs(probs))

  return posteriors

def m_step(model, optimizer, x, t, e, posteriors, typ='soft'):

  optimizer.zero_grad()
  loss = q_function(model, x, t, e, posteriors, typ)
  loss.backward()
  optimizer.step()

  return float(loss)

def weighted_breslow_estimator(t, e, risks=None, weights=None):

	if risks is None:
		risks = np.ones_like(t)
	
	if weights is None:
		weights = np.ones_like(t)

	# sort the data by time
	idx = np.argsort(t)
	t, e, risks, weights = t[idx], e[idx], risks[idx], weights[idx]

	# compute changepoints all t
	diff = np.where((t[1:] - t[:-1]).astype(bool))[0] + 1
	diff = np.array([0] + list(diff))

	# compute changepoints for uncensored t
	diff_uncensored = np.where((t[e==1][1:] - t[e==1][:-1]).astype(bool))[0]
	diff_uncensored = np.array(list(diff_uncensored) + [len(t[e==1])-1])


	t_failed_counts_ = np.cumsum(weights[e==1])

	# print("t uniq failed:", t[e==1])
	# print("weights uniq failed:", weights[e==1])
	# print("cumulative weights fail:", t_failed_counts_)
	# print("cps: ", diff_uncensored)


	t_failed_counts_ = np.insert(t_failed_counts_[diff_uncensored], 0, 0)
	t_failed_counts_ = t_failed_counts_[1:] - t_failed_counts_[:-1]

	# print(np.unique(t[e==1]))
	# print(t_failed_counts_)


	# compute the number of deaths at each time point.
	t_failed_unique, t_failed_counts = np.unique(t[e==1], return_counts=True)
	t_all_unique, _ = np.unique(t, return_counts=True)


	# find the indices of the failed times
	_, _, idx = np.intersect1d(t_failed_unique, t_all_unique, return_indices=True)

	t_failed_all = np.zeros_like(t_all_unique)
	t_failed_all[idx] = t_failed_counts_
	
	# compute the number of patients at-risk.
	adjusted_at_risk_counts = np.cumsum(weights[::-1]*risks[::-1])[::-1][diff]

	estimate = np.cumsum(t_failed_all/adjusted_at_risk_counts)

	return t_all_unique, estimate


def fit_breslow(model, x, t, e, posteriors=None,
                smoothing_factor=1e-4, typ='soft'):

  # TODO: Make Breslow in Torch !!!

  gates, lrisks = model(x)

  lrisks = lrisks.numpy()

  e = e.numpy()
  t = t.numpy()

  if posteriors is None: z_probs = gates
  else: z_probs = posteriors

  # if typ == 'soft': z = sample_hard_z(z_probs)
  # else: z = get_hard_z(z_probs)

  z_probs = z_probs.exp().detach().numpy()

  breslow_splines = {}
  for i in range(model.k):

#    breslowkn = weighted_breslow_estimator(t[z==i], e[z==i], np.exp(lrisks[:, i][z==i]))

    breslowkn = weighted_breslow_estimator(t, e, np.exp(lrisks[:, i]),
                                           weights=z_probs[:, i])

    #breslowk = BreslowEstimator().fit(lrisks[:, i][z==i], e[z==i], t[z==i])
    # breslow_splines[i] = smooth_bl_survival(breslowk.baseline_survival_.x,
    #                                         breslowk.baseline_survival_.y,
    #                                         smoothing_factor=smoothing_factor)

    breslow_splines[i] = smooth_bl_survival(breslowkn[0], np.exp(-breslowkn[1]),
                                            smoothing_factor=smoothing_factor)

  return breslow_splines


def train_step(model, x, t, e, breslow_splines, optimizer,
               bs=256, seed=100, typ='soft', use_posteriors=False,
               update_splines_after=10, smoothing_factor=1e-4):

  x, t, e = shuffle(x, t, e, random_state=seed)

  n = x.shape[0]

  batches = (n // bs) + 1

  epoch_loss = 0
  for i in range(batches):

    xb = x[i*bs:(i+1)*bs]
    tb = t[i*bs:(i+1)*bs]
    eb = e[i*bs:(i+1)*bs]
    #ab = a[i*bs:(i+1)*bs]

    # E-Step !!!
    # e_step_start = time.time()
    with torch.no_grad():
      posteriors = e_step(model, breslow_splines, xb, tb, eb)

    torch.enable_grad()
    loss = m_step(model, optimizer, xb, tb, eb, posteriors, typ=typ)

    with torch.no_grad():
      try:
        if i%update_splines_after == 0:
          if use_posteriors:

            posteriors = e_step(model, breslow_splines, x, t, e)
            breslow_splines = fit_breslow(model, x, t, e, 
                                          posteriors=posteriors,
                                          typ='soft',
                                          smoothing_factor=smoothing_factor)
          else:
            breslow_splines = fit_breslow(model, x, t, e,
                                          posteriors=None,
                                          typ='soft',
                                          smoothing_factor=smoothing_factor)
          # print(f'Duration of Breslow spline estimation: {time.time() - estimate_breslow_start}')
      except Exception as exce:
        print("Exception!!!:", exce)
        logging.warning("Couldn't fit splines, reusing from previous epoch")
    epoch_loss += loss
  #print (epoch_loss/n)
  return breslow_splines


def test_step(model, x, t, e, breslow_splines, loss='q', typ='soft'):

  if loss == 'q':
    with torch.no_grad():
      posteriors = e_step(model, breslow_splines, x, t, e)
      loss = q_function(model, x, t, e, posteriors, typ=typ)

  return float(loss/x.shape[0])


def train_dcm(model, train_data, val_data, epochs=50,
              patience=3, vloss='q', bs=256, typ='soft', lr=1e-3,
              use_posteriors=True, debug=False, random_seed=0,
              return_losses=False, update_splines_after=10,
              smoothing_factor=1e-2):

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  if val_data is None:
    val_data = train_data

  xt, tt, et = train_data
  xv, tv, ev = val_data

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  optimizer = get_optimizer(model, lr)

  valc = np.inf
  patience_ = 0

  breslow_splines = None

  losses = []

  for epoch in tqdm(range(epochs)):

    # train_step_start = time.time()
    breslow_splines = train_step(model, xt, tt, et, breslow_splines,
                                 optimizer, bs=bs, seed=epoch, typ=typ,
                                 use_posteriors=use_posteriors,
                                 update_splines_after=update_splines_after,
                                 smoothing_factor=smoothing_factor)
    # print(f'Duration of train-step: {time.time() - train_step_start}')
    # test_step_start = time.time()
    valcn = test_step(model, xv, tv, ev, breslow_splines, loss=vloss, typ=typ)
    # print(f'Duration of test-step: {time.time() - test_step_start}')

    losses.append(valcn)

    if epoch % 1 == 0:
      if debug: print(patience_, epoch, valcn)

    if valcn > valc: patience_ += 1
    else: patience_ = 0

    if patience_ == patience:
      if return_losses: return (model, breslow_splines), losses
      else: return (model, breslow_splines)

    valc = valcn

  if return_losses: return (model, breslow_splines), losses
  else: return (model, breslow_splines)


def predict_survival(model, x, t):

  if isinstance(t, int) or isinstance(t, float): t = [t]

  model, breslow_splines = model
  gates, lrisks = model(x)

  lrisks = lrisks.detach().numpy()
  gate_probs = torch.exp(gates).detach().numpy()

  predictions = []

  for t_ in t:
    expert_output = get_survival(lrisks, breslow_splines, t_)
    predictions.append((gate_probs*expert_output).sum(axis=1))

  return np.array(predictions).T

def predict_latent_z(model, x):

  model, _ = model
  gates, _ = model(x)

  gate_probs = torch.exp(gates).detach().numpy()

  return gate_probs
