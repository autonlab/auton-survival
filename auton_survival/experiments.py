import numpy as np
from sklearn.utils import shuffle

from auton_survival.estimators import SurvivalModel, CounterfactualSurvivalModel
from auton_survival.metrics import survival_regression_metric

from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

class SurvivalRegressionCV:
  """Universal interface to train Survival Analysis models in a Cross Validation fashion.
  
  Each of the model is trained in a CV fashion over the user specified
  hyperparameter grid. The best model (in terms of integrated brier score)
  is then selected.
  
  Parameters
  -----------
  model : str
      A string that determines the choice of the surival analysis model.
      Survival model choices include:

      - 'dsm' : Deep Survival Machines [3] model
      - 'dcph' : Deep Cox Proportional Hazards [2] model
      - 'dcm' : Deep Cox Mixtures [4] model
      - 'rsf' : Random Survival Forests [1] model
      - 'cph' : Cox Proportional Hazards [2] model
  cv_folds : int
      Number of folds in the cross validation.
  random_seed : int
      Random seed for reproducibility.
  hyperparam_grid : dict
      A dictionary that contains the hyperparameters for grid search.
      The keys of the dictionary are the hyperparameter names and the
      values are lists of hyperparameter values.

  References
  -----------

  [1] Hemant Ishwaran et al. Random survival forests.
  The annals of applied statistics, 2(3):841–860, 2008.

  [2] Cox, D. R. (1972). Regression models and life-tables.
  Journal of the Royal Statistical Society: Series B (Methodological).

  [3] Chirag Nagpal, Xinyu Li, and Artur Dubrawski.
  Deep survival machines: Fully parametric survival regression and
  representation learning for censored data with competing risks. 2020.

  [4] Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K. (2021c).
  Deep cox mixtures for survival regression.
  In Machine Learning for Healthcare Conference, pages 674–708. PMLR

  """

  def __init__(self, model, cv_folds=5, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    self.random_seed = random_seed
    self.cv_folds = cv_folds

  def fit(self, features, outcomes, ret_trained_model=True):

    r"""Fits the Survival Regression Model to the data in a Cross Validation fashion.

    Parameters
    -----------
    features : pandas.DataFrame
        a pandas dataframe containing the features to use as covariates.
    outcomes : pandas.DataFrame
        a pandas dataframe containing the survival outcomes. The index of the
        dataframe should be the same as the index of the features dataframe.
        Should contain a column named 'time' that contains the survival time and
        a column named 'event' that contains the censoring status.
        \( \delta_i = 1 \) if the event is observed.
    ret_trained_model : bool
        If True, the trained model is returned. If False, the fit function returns
        self.

    Returns
    -----------
    auton_survival.estimators.SurvivalModel:
        The selected survival model based on lowest integrated brier score.
    
    """

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
        predictions[folds==fold] = fold_model.predict_survival(features.loc[folds==fold], times=unique_times)
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

    """"Not implemented yet."""

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

  r"""Universal interface to train Counterfactual Survival Analysis models in a
  Cross Validation fashion.

  Each of the model is trained in a CV fashion over the user specified
  hyperparameter grid. The best model (in terms of integrated brier score)
  is then selected.

  Parameters
  -----------
  model : str
      A string that determines the choice of the surival analysis model.
      Survival model choices include:

      - 'dsm' : Deep Survival Machines [3] model
      - 'dcph' : Deep Cox Proportional Hazards [2] model
      - 'dcm' : Deep Cox Mixtures [4] model
      - 'rsf' : Random Survival Forests [1] model
      - 'cph' : Cox Proportional Hazards [2] model
  cv_folds : int
      Number of folds in the cross validation.
  random_seed : int
      Random seed for reproducibility.
  hyperparam_grid : dict
      A dictionary that contains the hyperparameters for grid search.
      The keys of the dictionary are the hyperparameter names and the
      values are lists of hyperparameter values.

  References
  -----------

  [1] Hemant Ishwaran et al. Random survival forests.
  The annals of applied statistics, 2(3):841–860, 2008.

  [2] Cox, D. R. (1972). Regression models and life-tables.
  Journal of the Royal Statistical Society: Series B (Methodological).

  [3] Chirag Nagpal, Xinyu Li, and Artur Dubrawski.
  Deep survival machines: Fully parametric survival regression and
  representation learning for censored data with competing risks. 2020.

  [4] Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K. (2021c).
  Deep cox mixtures for survival regression.
  In Machine Learning for Healthcare Conference, pages 674–708. PMLR

  """

  def __init__(self, model, cv_folds=5, random_seed=0, hyperparam_grid={}):

    self.model = model
    self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    self.random_seed = random_seed
    self.cv_folds = cv_folds

    self.treated_experiment = SurvivalRegressionCV(model=model,
                                                   cv_folds=cv_folds,
                                                   random_seed=random_seed,
                                                   hyperparam_grid=hyperparam_grid)

    self.control_experiment = SurvivalRegressionCV(model=model,
                                                   cv_folds=cv_folds,
                                                   random_seed=random_seed,
                                                   hyperparam_grid=hyperparam_grid)

  def fit(self, features, outcomes, interventions):

    r"""Fits the Survival Regression Model to the data in a Cross Validation fashion.

    Parameters
    -----------
    features : pandas.DataFrame
        a pandas dataframe containing the features to use as covariates.
    outcomes : pandas.DataFrame
        a pandas dataframe containing the survival outcomes. The index of the
        dataframe should be the same as the index of the features dataframe.
        Should contain a column named 'time' that contains the survival time and
        a column named 'event' that contains the censoring status.
        \( \delta_i = 1 \) if the event is observed.
    interventions: pandas.Series
        A pandas series containing the treatment status of each subject.
        \( a_i = 1 \) if the subject is `treated`, else is considered control.

    Returns
    -----------
    auton_survival.estimators.CounterfactualSurvivalModel:
        The trained counterfactual survival model.

    """


    treated, control = interventions==1, interventions!=1
    treated_model = self.treated_experiment.fit(features.loc[treated],
                                                outcomes.loc[treated])
    control_model = self.control_experiment.fit(features.loc[control],
                                                outcomes.loc[control])

    return CounterfactualSurvivalModel(treated_model, control_model)
