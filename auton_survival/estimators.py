import numpy as np
import pandas as pd

def _get_valid_idx(n, size, random_seed):

  import numpy as np
  np.random.seed(random_seed)


  validx = sorted(np.random.choice(n, size=(int(size*n)), replace=False))
  vidx = np.zeros(n).astype('bool')
  vidx[validx] = True

  return vidx

def _fit_dcm(features, outcomes, random_seed, **hyperparams):

  from sdcm.dcm import DeepCoxMixture, CoxMixture
  from sdcm.dcm_utils import train

  import torch
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  k = hyperparams.get("k", 3) 
  layers = hyperparams.get("layers", [100])
  bs = hyperparams.get("bs", 100)
  lr = hyperparams.get("lr", 1e-3)
  epochs = hyperparams.get("epochs", 50)
  smoothing_factor = hyperparams.get("smoothing_factor", 0)

  if len(layers): model = DeepCoxMixture(k=k, inputdim=features.shape[1], hidden=layers[0])
  else: model = CoxMixture(k=k, inputdim=features.shape[1])

  x = torch.from_numpy(features.values.astype('float32'))
  t = torch.from_numpy(outcomes['time'].values.astype('float32'))
  e = torch.from_numpy(outcomes['event'].values.astype('float32'))

  vidx = _get_valid_idx(x.shape[0], 0.15, random_seed)

  train_data = (x[~vidx], t[~vidx], e[~vidx])
  val_data = (x[vidx], t[vidx], e[vidx])

  (model, breslow_splines, unique_times) = train(model, train_data, val_data, 
                                                  epochs=epochs, lr=lr, bs=bs,
                                                  use_posteriors=True, patience=5, return_losses=False,
                                                  smoothing_factor=smoothing_factor)

  return (model, breslow_splines, unique_times)

def _predict_dcm(model, features, times):

  from sdcm.dcm_utils import predict_scores

  import torch
  x = torch.from_numpy(features.values.astype('float32'))

  survival_predictions = predict_scores(model, x, None, model[-1])
  if len(times)>1:
    survival_predictions = pd.DataFrame(survival_predictions, columns=times).T
    return __interpolate_missing_times(survival_predictions, times)
  else:
    return survival_predictions

def _fit_dcph(features, outcomes, random_seed, **hyperparams):

  import torch
  import torchtuples as ttup

  from pycox.models import CoxPH

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  layers = hyperparams.get('layers', [100])
  lr = hyperparams.get('lr', 1e-3)
  bs = hyperparams.get('bs', 100)
  epochs = hyperparams.get('epochs', 50)
  activation = hyperparams.get('activation', 'relu')

  if activation == 'relu': activation = torch.nn.ReLU()
  elif activation == 'relu6': activation = torch.nn.ReLU6() 
  elif activation == 'tanh': activation = torch.nn.Tanh() 
  else: raise NotImplementedError("Activation function not implemented")

  x = features.values.astype('float32')
  t = outcomes['time'].values.astype('float32')
  e = outcomes['event'].values.astype('bool')

  in_features = x.shape[1]
  out_features = 1
  batch_norm = False
  dropout = 0.0

  net = ttup.practical.MLPVanilla(in_features, layers,
                                  out_features, batch_norm, dropout,
                                  activation=activation,
                                  output_bias=False)

  model = CoxPH(net, torch.optim.Adam)

  vidx = _get_valid_idx(x.shape[0], 0.15, random_seed)

  y_train, y_val = (t[~vidx], e[~vidx]), (t[vidx], e[vidx])
  val_data = x[vidx], y_val

  callbacks = [ttup.callbacks.EarlyStopping()]
  model.fit(x[~vidx], y_train, bs, epochs, callbacks, True,
            val_data=val_data,
            val_batch_size=bs)
  model.compute_baseline_hazards()

  return model

def __interpolate_missing_times(survival_predictions, times):

  nans = np.full(survival_predictions.shape[1], np.nan)
  not_in_index = list(set(times) - set(survival_predictions.index))

  for idx in not_in_index:
    survival_predictions.loc[idx] = nans
  return survival_predictions.sort_index(axis=0).interpolate().interpolate(method='bfill').T[times].values


def _predict_dcph(model, features, times):
  if isinstance(times, float) or isinstance(times, int):
    times = [float(times)]

  survival_predictions = model.predict_surv_df(features.values.astype('float32'))

  return __interpolate_missing_times(survival_predictions, times)


def _fit_cph(features, outcomes, random_seed, **hyperparams):
  
  from lifelines import CoxPHFitter

  data = outcomes.join(features)
  penalizer = hyperparams.get('l2', 1e-3)

  return CoxPHFitter(penalizer=penalizer).fit(data, duration_col='time', event_col='event')

def _fit_rsf(features, outcomes, random_seed, **hyperparams):

  from sksurv.ensemble import RandomSurvivalForest
  from sksurv.util import Surv

  n_estimators = hyperparams.get("n_estimators", 50)
  max_depth = hyperparams.get("max_depth", 5)
  max_features = hyperparams.get("max_features", 'sqrt')

  # Initialize an RSF model. 
  rsf = RandomSurvivalForest(n_estimators=n_estimators,
                             max_depth=max_depth,
                             max_features=max_features,
                             random_state=random_seed, 
                             n_jobs=-1)

  y =  Surv.from_dataframe('event', 'time', outcomes)
  # Fit the RSF model
  rsf.fit(features.values, y)
  return rsf


def _fit_dsm(features, outcomes, random_seed, **hyperparams):

  from .models.dsm import DeepSurvivalMachines

  k = hyperparams.get("k", 3) 
  layers = hyperparams.get("layers", [100])
  iters = hyperparams.get("iters", 10)
  distribution = hyperparams.get("distribution", "Weibull")
  temperature = hyperparams.get("temperature", 1.0)

  model = DeepSurvivalMachines(k=k, layers=layers,
                               distribution=distribution,
                               temp=temperature)

  model.fit(features.values, 
            outcomes['time'].values,
            outcomes['event'].values,
            iters=iters)

  return model

def _predict_dsm(model, features, times):
  return model.predict_survival(features.values, times)

def _predict_cph(model, features, times):
  if isinstance(times, float): times = [times] 
  return model.predict_survival_function(features, times=times).values.T

def _predict_rsf(model, features, times):
  if isinstance(times, float) or isinstance(times, int):
    times = [float(times)]

  survival_predictions = model.predict_survival_function(features.values, return_array=True)
  survival_predictions = pd.DataFrame(survival_predictions, columns=model.event_times_).T

  return __interpolate_missing_times(survival_predictions, times)

def _predict_dcm(model, features, times):
  if isinstance(times, float) or isinstance(times, int):
    times = [float(times)]

  from sdcm.dcm_utils import predict_scores

  import torch
  x = torch.from_numpy(features.values.astype('float32'))

  survival_predictions = predict_scores(model, x, times)
  survival_predictions = pd.DataFrame(survival_predictions, columns=times).T

  return __interpolate_missing_times(survival_predictions, times)


class SurvivalModel:

  _VALID_MODELS = ['rsf', 'cph', 'dsm', 'dcph', 'dcm']

  def __init__(self, model, random_seed=0, **hyperparams):

    assert model in SurvivalModel._VALID_MODELS

    self.model = model
    self.hyperparams = hyperparams
    self.random_seed = random_seed
    self.fitted = False

  def fit(self, features, outcomes):
    
    if self.model == 'cph': self._model = _fit_cph(features, outcomes, self.random_seed, **self.hyperparams)
    elif self.model == 'rsf': self._model = _fit_rsf(features, outcomes, self.random_seed, **self.hyperparams)
    elif self.model == 'dsm': self._model = _fit_dsm(features, outcomes, self.random_seed, **self.hyperparams)
    elif self.model == 'dcph': self._model = _fit_dcph(features, outcomes, self.random_seed, **self.hyperparams)
    elif self.model == 'dcm': self._model = _fit_dcm(features, outcomes, self.random_seed, **self.hyperparams)
    else : raise NotImplementedError()
    self.fitted = True
    return self

  def predict_survival(self, features, times):

    if self.model == 'cph': return _predict_cph(self._model, features, times)
    elif self.model == 'rsf': return _predict_rsf(self._model, features, times)
    elif self.model == 'dsm': return _predict_dsm(self._model, features, times) 
    elif self.model == 'dcph': return _predict_dcph(self._model, features, times) 
    elif self.model == 'dcm': return _predict_dcm(self._model, features, times) 
    else : raise NotImplementedError()

  def predict_risk(self, features, times):

    return 1 - self.predict_survival(features, times)

class CounterfactualSurvivalModel:

  _VALID_MODELS = ['rsf', 'cph', 'dsm']

  def __init__(self, treated_model, control_model):

    assert isinstance(treated_model, SurvivalModel)
    assert isinstance(control_model, SurvivalModel)
    assert treated_model.fitted
    assert control_model.fitted

    self.treated_model = treated_model
    self.control_model = control_model

  def predict(self, features, times ):
    raise NotImplementedError()

  def predict_counterfactual(self, features, times):
    
    control_outcomes = self.control_model.predict(features, times)
    treated_outcomes = self.treated_model.predict(features, times)

    return treated_outcomes, control_outcomes

class DCMSubgroupModel(CounterfactualSurvivalModel):

  def __init__(self, random_seed=0, **hyperparams):

    self.fitted = False
    self.random_seed = random_seed
    self.hyperparams = hyperparams

  def fit(self, features, outcomes, intervention_col):
    
    assert intervention_col in features.columns, "The Intervention is not in features."
    self.intervention_col = intervention_col

    from sdcm.dcm_subgroup import DeepCoxSubgroupMixture, CoxSubgroupMixture
    from sdcm.dcm_subgroup_utils import train

    import torch
    torch.manual_seed(self.random_seed)
    np.random.seed(self.random_seed)

    k = self.hyperparams.get("k", 3) 
    g = self.hyperparams.get("g", 2)
    layers = self.hyperparams.get("layers", [100])
    bs = self.hyperparams.get("bs", 100)
    lr = self.hyperparams.get("lr", 1e-3)
    epochs = self.hyperparams.get("epochs", 50)

    if len(layers): model = DeepCoxSubgroupMixture(k=k, g=g, inputdim=features.shape[1]-1, hidden=layers[0])
    else: model = CoxSubgroupMixture(k=k, g=g, inputdim=features.shape[1]-1)
    
    feature_cols = list(set(features.columns) - set([intervention_col]))
    x = torch.from_numpy(features[feature_cols].values.astype('float32'))
    t = torch.from_numpy(outcomes['time'].values.astype('float32'))
    e = torch.from_numpy(outcomes['event'].values.astype('float32'))
    a = torch.from_numpy(features[intervention_col].values.astype('float32'))

    vidx = _get_valid_idx(x.shape[0], 0.15, self.random_seed)

    train_data = (x[~vidx], t[~vidx], e[~vidx], a[~vidx])
    val_data = (x[vidx], t[vidx], e[vidx], a[vidx])

    (model, breslow_splines) = train(model, train_data, val_data, 
                                    epochs=epochs, lr=lr, bs=bs,
                                    use_posteriors=True, patience=5, return_losses=False)

    self._model = (model, breslow_splines)
    return self
    
  def predict_survival(self, features, times):
    
    from sdcm.dcm_subgroup_utils import predict_scores
    import torch

    feature_cols = list(set(features.columns) - set([self.intervention_col]))
    x = torch.from_numpy(features[feature_cols].values.astype('float32'))
    a = torch.from_numpy(features[self.intervention_col].values.astype('float32'))

    return predict_scores(self._model, x, a, times)

  def predict_counterfactual(self, features, times):

    from sdcm.dcm_subgroup_utils import predict_scores
    import torch

    feature_cols = list(set(features.columns) - set([self.intervention_col]))
    x = torch.from_numpy(features[feature_cols].values.astype('float32'))
     
    a1 = torch.from_numpy(np.ones(len(features)).astype('float32'))
    a0 = torch.from_numpy(np.zeros(len(features)).astype('float32'))

    return predict_scores(self._model, x, a0, times), predict_scores(self._model, x, a1, times)
  
