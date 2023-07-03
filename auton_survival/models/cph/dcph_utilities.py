import torch
import numpy as np
import pandas as pd

from sksurv.linear_model.coxph import BreslowEstimator

from sklearn.utils import shuffle

from tqdm import tqdm

from auton_survival.models.dsm.utilities import get_optimizer, _reshape_tensor_with_nans

from copy import deepcopy

def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)

def partial_ll_loss(lrisks, tb, eb, eps=1e-3):

  tb = tb + eps*np.random.random(len(tb))
  sindex = np.argsort(-tb)

  tb = tb[sindex]
  eb = eb[sindex]

  lrisks = lrisks[sindex]
  lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)

  plls = lrisks - lrisksdenom
  pll = plls[eb == 1]

  pll = torch.sum(pll)

  return -pll

def fit_breslow(model, x, t, e):
  return BreslowEstimator().fit(model(x).detach().cpu().numpy(),
                                e.numpy(), t.numpy())

def train_step(model, x, t, e, optimizer, bs=256, seed=100):

  x, t, e = shuffle(x, t, e, random_state=seed)

  n = x.shape[0]

  batches = (n // bs) + 1

  epoch_loss = 0

  for i in range(batches):

    xb = x[i*bs:(i+1)*bs]
    tb = t[i*bs:(i+1)*bs]
    eb = e[i*bs:(i+1)*bs]

    # Training Step
    torch.enable_grad()
    optimizer.zero_grad()
    loss = partial_ll_loss(model(xb),
                          _reshape_tensor_with_nans(tb),
                          _reshape_tensor_with_nans(eb))
    loss.backward()
    optimizer.step()

    epoch_loss += float(loss)

  return epoch_loss/n

def test_step(model, x, t, e):

  with torch.no_grad():
    loss = float(partial_ll_loss(model(x), t, e))

  return loss/x.shape[0]


def train_dcph(model, train_data, val_data, epochs=50,
               patience=3, bs=256, lr=1e-3, debug=False,
               random_seed=0, return_losses=False):

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  if val_data is None:
    val_data = train_data

  xt, tt, et = train_data
  xv, tv, ev = val_data

  tt_ = _reshape_tensor_with_nans(tt)
  et_ = _reshape_tensor_with_nans(et)
  tv_ = _reshape_tensor_with_nans(tv)
  ev_ = _reshape_tensor_with_nans(ev)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  optimizer = get_optimizer(model, lr)

  valc = np.inf
  patience_ = 0

  breslow_spline = None

  losses = []
  dics = []
  
  for epoch in tqdm(range(epochs)):

    # train_step_start = time.time()
    _ = train_step(model, xt, tt, et, optimizer, bs, seed=epoch)
    # print(f'Duration of train-step: {time.time() - train_step_start}')
    # test_step_start = time.time()
    valcn = test_step(model, xv, tv_, ev_)
    # print(f'Duration of test-step: {time.time() - test_step_start}')

    losses.append(float(valcn))
    
    dics.append(deepcopy(model.state_dict()))

    if epoch % 1 == 0:
      if debug: print(patience_, epoch, valcn)

    if valcn > valc:
      patience_ += 1
    else:
      patience_ = 0

    if patience_ == patience:
      
      minm = np.argmin(losses)
      model.load_state_dict(dics[minm])

      breslow_spline = fit_breslow(model, xt, tt_, et_)

      if return_losses:
        return (model, breslow_spline), losses
      else:
        return (model, breslow_spline)

    valc = valcn
    
  minm = np.argmin(losses)
  model.load_state_dict(dics[minm])
  
  breslow_spline = fit_breslow(model, xt, tt_, et_)

  if return_losses:
    return (model, breslow_spline), losses
  else:
    return (model, breslow_spline)

def predict_survival(model, x, t=None):

  if isinstance(t, (int, float)): t = [t]

  model, breslow_spline = model
  lrisks = model(x).detach().cpu().numpy()

  unique_times = breslow_spline.baseline_survival_.x

  raw_predictions = breslow_spline.get_survival_function(lrisks)
  raw_predictions = np.array([pred.y for pred in raw_predictions])

  predictions = pd.DataFrame(data=raw_predictions, columns=unique_times)

  if t is None:
    return predictions
  else:
    return __interpolate_missing_times(predictions.T, t)
    #return np.array(predictions).T

def __interpolate_missing_times(survival_predictions, times):

  nans = np.full(survival_predictions.shape[1], np.nan)
  not_in_index = list(set(times) - set(survival_predictions.index))

  for idx in not_in_index:
    survival_predictions.loc[idx] = nans
  return survival_predictions.sort_index(axis=0).interpolate(method='bfill').T[times].values
