import numpy as np
from sklearn.utils import shuffle

from auton_survival.estimators import SurvivalModel, CounterfactualSurvivalModel
from auton_survival.metrics import survival_regression_metric

from sklearn.model_selection import ParameterGrid 

from tqdm import tqdm

class SurvivalRegressionCV:

  def __init__(self, model, cv_folds=5, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    self.random_seed = random_seed
    self.cv_folds = cv_folds

  def fit(self, features, outcomes, ret_trained_model=True):

    n = len(features)

    np.random.seed(self.random_seed)

    folds = np.array(list(range(self.cv_folds))*n)[:n]
    np.random.shuffle(folds)

    self.folds = folds

    unique_times = np.unique(outcomes['time'].values)

    time_min, time_max = unique_times.min(), unique_times.max()

    for fold in range(self.cv_folds):

      fold_outcomes = outcomes.loc[folds==fold, 'time'] 

      if fold_outcomes.min() > time_min: time_min = fold_outcomes.min() 
      if fold_outcomes.max() < time_max: time_max = fold_outcomes.max() 
      
    unique_times = unique_times[unique_times>=time_min]
    unique_times = unique_times[unique_times<time_max]

    scores = []

    best_model = {}
    best_score = np.inf

    for hyper_param in tqdm(self.hyperparam_grid):    

      predictions = np.zeros((len(features), len(unique_times)))

      fold_models = {}
      for fold in tqdm(range(self.cv_folds)):
        # Fit the model
        fold_model = SurvivalModel(model=self.model, random_seed=self.random_seed, **hyper_param)      
        fold_model.fit(features.loc[folds!=fold], outcomes.loc[folds!=fold])
        fold_models[fold] = fold_model 

        # Predict risk scores
        predictions[folds==fold] = fold_model.predict(features.loc[folds==fold], times=unique_times)
        # Evaluate IBS
      score_per_fold = [] 
      for fold in range(self.cv_folds):
        score = survival_regression_metric('ibs', predictions, outcomes, unique_times, folds, fold)
        score_per_fold.append(score)

      current_score = np.mean(score_per_fold)

      if current_score < best_score:
        best_score = current_score
        best_model = fold_models    
        best_hyper_param = hyper_param 
        best_predictions = predictions

    self.best_hyperparameter = best_hyper_param
    self.best_model_per_fold = best_model
    self.best_predictions = best_predictions

    if ret_trained_model:

      model = SurvivalModel(model=self.model, random_seed=self.random_seed, **self.best_hyperparameter)
      model.fit(features, outcomes)

      return model
    
    else:
      return self

  def evaluate(self, features, outcomes, metrics=['auc', 'ctd'], horizons=[]):

    raise NotImplementedError()

    results = {}

    for metric in metrics:
      results[metric] = {}
      for horizon in horizons:
        results[metric][horizon] = {}
        for fold in range(self.cv_folds):
          results[metric][horizon][fold] = {}

    for fold in range(self.cv_folds):

      fold_model = self.best_model_per_fold[fold]
      fold_predictions = fold_model.predict(features.loc[self.folds==fold], times=horizons) 

      for i, horizon in enumerate(horizons):
        for metric in metrics:
          raise NotImplementedError()

        
class CounterfactualSurvivalRegressionCV:
  
  def __init__(self, model, cv_folds=5, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    self.random_seed = random_seed
    self.cv_folds = cv_folds

    self.treated_experiment = SurvivalRegressionCV(model=model, cv_folds=cv_folds, random_seed=random_seed, hyperparam_grid=hyperparam_grid)
    self.control_experiment = SurvivalRegressionCV(model=model, cv_folds=cv_folds, random_seed=random_seed, hyperparam_grid=hyperparam_grid)

  def fit(self, features, outcomes, interventions):
    
    treated, control = interventions==1, interventions!=1
    treated_model = self.treated_experiment.fit(features.loc[treated], outcomes.loc[treated])
    control_model = self.control_experiment.fit(features.loc[control], outcomes.loc[control])

    return CounterfactualSurvivalModel(treated_model, control_model)
