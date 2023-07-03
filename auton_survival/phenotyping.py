# coding=utf-8
# MIT License

# Copyright (c) 2022 Carnegie Mellon University, Auton Lab

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

"""Utilities to phenotype individuals based on similar survival
characteristics."""

from random import random
import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn import cluster, decomposition, mixture
from sklearn.metrics import auc

from auton_survival.utils import _get_method_kwargs
from auton_survival.experiments import CounterfactualSurvivalRegressionCV


class Phenotyper:
  """Base class for all phenotyping methods."""

  def __init__(self, random_seed=0):

    self.random_seed = random_seed
    self.fitted = False

class IntersectionalPhenotyper(Phenotyper):

  """A phenotyper that phenotypes by performing an exhaustive cartesian
  product on prespecified set of categorical and numerical variables.

  Parameters
  -----------
  cat_vars : list of python str(s), default=None
      List of column names of categorical variables to phenotype on.
  num_vars : list of python str(s), default=None
      List of column names of continuous variables to phenotype on.
  num_vars_quantiles : tuple of floats, default=(0, .5, 1.0)
      A tuple of quantiles as floats (inclusive of 0 and 1) used to
      discretize continuous variables into equal-sized bins.
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to individual
      samples and columns as covariates.
  phenotypes : list
      List of lists containing all possible combinations of specified
      categorical and numerical variable values.

  """

  def __init__(self, cat_vars=None, num_vars=None,
               num_vars_quantiles=(0, .5, 1.0), random_seed=0):

    super().__init__(random_seed=random_seed)

    if isinstance(cat_vars, str): cat_vars = [cat_vars]
    if isinstance(num_vars, str): num_vars = [num_vars]

    if cat_vars is None: cat_vars = []
    if num_vars is None: num_vars = []

    assert len(cat_vars+num_vars) != 0, "Please specify intersectional Groups"

    self.cat_vars = cat_vars
    self.num_vars = num_vars
    self.num_vars_quantiles = num_vars_quantiles

    self.fitted = False

  def fit(self, features):

    """Fit the phenotyper by finding all possible intersectional groups
    on a passed set of features.

    Parameters
    -----------
    features : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    Trained instance of intersectional phenotyper.

    """

    self.cut_bins = {}
    self.min_max = {}

    for num_var in self.num_vars:

      binmin = float(features[num_var].min())
      binmax = float(features[num_var].max())

      self.min_max[num_var] = binmin, binmax
      _, self.cut_bins[num_var] = pd.qcut(features[num_var],
                                          self.num_vars_quantiles,
                                          retbins=True)

    self.fitted = True
    return self

  def predict(self, features):

    """Phenotype out of sample test data.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    --------
    np.array:
        a numpy array containing a list of strings that define
        subgroups from all possible combinations of specified categorical
        and numerical variables.

    """

    assert self.fitted, "Phenotyper must be `fitted` before calling `phenotype`."
    features = deepcopy(features)

    for num_var in self.num_vars:

      var_min, var_max = self.min_max[num_var]

      features.loc[features[num_var]>=var_max, [num_var]] = var_max
      features.loc[features[num_var]<=var_min, [num_var]] = var_min

      features[num_var] = pd.cut(features[num_var], self.cut_bins[num_var],
                                 include_lowest=True)

    phenotypes = [group.tolist() for group in features[self.cat_vars+self.num_vars].values]
    phenotypes = self._rename(phenotypes)

    phenotypes = np.array(phenotypes)

    return phenotypes

  def _rename(self, phenotypes):

    """Helper function to clean the phenotype names.

    Parameters
    -----------
    phenotypes : list
        List of lists containing all possible combinations of specified
        categorical and numerical variable values.

    Returns
    --------
    list:
        python list of a list of strings that define subgroups.

    """

    ft_names = self.cat_vars + self.num_vars
    renamed = []
    for i in range(len(phenotypes)):
      row = []
      for j in range(len(ft_names)):
        row.append(ft_names[j]+":"+str(phenotypes[i][j]))
      renamed.append(" & ".join(row))
    return renamed

  def fit_predict(self, features):

    """Fit and perform phenotyping on a given dataset.

    Parameters
    -----------
    features : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array:
        A numpy array containing a list of strings that define
        subgroups from all possible combinations of specified categorical
        and numerical variables.

    """

    return self.fit(features).predict(features)

class ClusteringPhenotyper(Phenotyper):

  """Phenotyper that performs dimensionality reduction followed by clustering.
  Learned clusters are considered phenotypes and used to group samples based
  on similarity in the covariate space.

  Parameters
  -----------
  features: pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples
      and columns as covariates.
  clustering_method : str, default='kmeans'
      The clustering method applied for phenotyping.
      Options include:

      - `kmeans`: K-Means Clustering
      - `dbscan`: Density-Based Spatial Clustering of Applications with
         Noise (DBSCAN)
      - `gmm`: Gaussian Mixture
      - `hierarchical`: Agglomerative Clustering
  dim_red_method: str, default=None
      The dimensionality reductions method applied.
      Options include:

      - `pca` : Principal Component Analysis
      - `kpca` : Kernel Principal Component Analysis
      - `nnmf` : Non-Negative Matrix Factorization
      - None : dimensionality reduction is not applied.
  random_seed : int, default=0
      Controls the randomness and reproducibility of called functions
  kwargs: dict
      Additional arguments for dimensionality reduction and clustering
      Please include dictionary key and item pairs specified by the following
      scikit-learn modules:

      - `pca` : sklearn.decomposition.PCA
      - `nnmf` : sklearn.decomposition.NMF
      - `kpca` : sklearn.decomposition.KernelPCA
      - `kmeans` : sklearn.cluster.KMeans
      - `dbscan` : sklearn.cluster.DBSCAN
      - `gmm` : sklearn.mixture.GaussianMixture
      - `hierarchical` : sklearn.cluster.AgglomerativeClustering

  """

  _VALID_DIMRED_METHODS = ['pca', 'kpca', 'nnmf', None]
  _VALID_CLUSTERING_METHODS = ['kmeans', 'dbscan', 'gmm', 'hierarchical']

  def __init__(self, clustering_method = 'kmeans', dim_red_method = None,
               random_seed=0, **kwargs):

    assert clustering_method in ClusteringPhenotyper._VALID_CLUSTERING_METHODS, "Please specify a valid Clustering method"
    assert dim_red_method in ClusteringPhenotyper._VALID_DIMRED_METHODS, "Please specify a valid Dimensionality Reduction method"

    # Raise warning if "hierarchical" is used with dim_redcution
    if (clustering_method in ['hierarchical']) and (dim_red_method is not None):
      print("WARNING: Are you sure you want to run hierarchical clustering on decomposed features?.",
            "Such behaviour is atypical.")

      # Dimensionality Reduction Step:
    if dim_red_method is not None:
      if dim_red_method == 'pca':
        dim_red_model = decomposition.PCA
      elif dim_red_method == 'nnmf':
        dim_red_model = decomposition.NMF
      elif 'kpca' in dim_red_method:
        dim_red_model = decomposition.KernelPCA
      else:
        raise NotImplementedError("Dimensionality Reduction method: "+dim_red_method+ " Not Implemented.")

    if clustering_method == 'kmeans':
      clustering_model=  cluster.KMeans
    elif clustering_method == 'dbscan':
      clustering_model = cluster.DBSCAN
    elif clustering_method == 'gmm':
      clustering_model = mixture.GaussianMixture
    elif clustering_method == 'hierarchical':
      clustering_model = cluster.AgglomerativeClustering
    else:
      raise NotImplementedError("Clustering method: "+clustering_method+ " Not Implemented.")

    self.clustering_method = clustering_method
    self.dim_red_method = dim_red_method

    c_kwargs = _get_method_kwargs(clustering_model, kwargs)
    if clustering_method == 'gmm':
      if 'covariance_type' not in c_kwargs:
        c_kwargs['covariance_type'] = 'diag'
      c_kwargs['n_components'] = c_kwargs.get('n_clusters', 3)

    self.clustering_model = clustering_model(random_state=random_seed,
                                             **c_kwargs)
    if dim_red_method is not None:
      d_kwargs = _get_method_kwargs(dim_red_model, kwargs)
      if dim_red_method == 'kpca':
        if 'kernel' not in d_kwargs:
          d_kwargs['kernel'] = 'rbf'
          d_kwargs['n_jobs'] = -1
          d_kwargs['max_iter'] = 500

      self.dim_red_model = dim_red_model(random_state=random_seed,
                                         **d_kwargs)

  def fit(self, features):

    """Perform dimensionality reduction and train an instance
    of the clustering algorithm.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual
        samples and columns as covariates.

    Returns
    --------
    Trained instance of clustering phenotyper.

    """

    if self.dim_red_method is not None:
      print("Fitting the following Dimensionality Reduction Model:\n",
            self.dim_red_model)
      self.dim_red_model = self.dim_red_model.fit(features)
      features = self.dim_red_model.transform(features)

    else:
      print("No Dimensionaity reduction specified...\n Proceeding to learn clusters with the raw features...")

    print("Fitting the following Clustering Model:\n", self.clustering_model)
    self.clustering_model = self.clustering_model.fit(features)
    self.fitted = True

    return self

  def _predict_proba_kmeans(self, features):

    """Estimate the probability of belonging to a cluster by computing
    the distance to cluster center normalized by the sum of distances
    to other clusters.

    Parameters
    -----------
    features: pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array:
        A numpy array of probability estimates of sample association to
        learned subgroups.

    """

    #TODO:MAYBE DO THIS IN LOG SPACE?

    negative_exp_distances = np.exp(-self.clustering_model.transform(features))
    probs = negative_exp_distances/negative_exp_distances.sum(axis=1).reshape((-1, 1))

    #assert int(np.sum(probs)) == len(probs), 'Not valid probabilities'

    return probs

  def predict_proba(self, features):

    """Peform dimensionality reduction, clustering, and estimate probability
    estimates of sample association to learned clusters, or subgroups.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array:
        a numpy array of the probability estimates of sample association to
        learned subgroups.

    """

    assert self.fitted, "Phenotyper must be `fitted` before calling \
    `phenotype`."

    if self.dim_red_method is not None:
      features =  self.dim_red_model.transform(features)
    if self.clustering_method == 'gmm':
      return self.clustering_model.predict_proba(features)
    elif self.clustering_method == 'kmeans':
      return self._predict_proba_kmeans(features)

  def predict(self, features):

    """Peform dimensionality reduction, clustering, and extract phenogroups
    that maximize the probability estimates of sample association to
    specific learned clusters, or subgroups.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array:
        a numpy array of phenogroup labels

    """

    assert self.fitted, "Phenotyper must be `fitted` before calling \
    `phenotype`."

    return np.argmax(self.predict_proba(features), axis=1)

  def fit_predict(self, features):

    """Fit and perform phenotyping on a given dataset.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array
        a numpy array of the probability estimates of sample association
        to learned clusters.

    """

    return self.fit(features).predict(features)

class SurvivalVirtualTwinsPhenotyper(Phenotyper):

  """Phenotyper that estimates the potential outcomes under treatment and
  control using a counterfactual Deep Cox Proportional Hazards model,
  followed by regressing the difference of the estimated counterfactual
  Restricted Mean Survival Times using a Random Forest regressor."""

  _VALID_PHENO_METHODS = ['rfr']
  _DEFAULT_PHENO_HYPERPARAMS = {}
  _DEFAULT_PHENO_HYPERPARAMS['rfr'] = {'n_estimators': 50,
                                       'max_depth': 5}

  def __init__(self,
               cf_method='dcph',
               phenotyping_method='rfr',
               cf_hyperparams=None,
               random_seed=0,
               **phenotyper_hyperparams):

    assert cf_method in CounterfactualSurvivalRegressionCV._VALID_CF_METHODS, "\
    Invalid Counterfactual Method: "+cf_method
    assert phenotyping_method in self._VALID_PHENO_METHODS, "Invalid Phenotyping Method:\
    "+phenotyping_method

    self.cf_method = cf_method
    self.phenotyping_method = phenotyping_method

    if cf_hyperparams is None:
      cf_hyperparams = {}

    self.phenotyper_hyperparams = phenotyper_hyperparams
    self.cf_hyperparams = cf_hyperparams

    self.random_seed = random_seed

  def fit(self, features, outcomes, interventions, horizons, metric):

    """Fit a counterfactual model and regress the difference of the estimated
    counterfactual Restricted Mean Survival Time using a Random Forest regressor.

    Parameters
    -----------
    features: pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.
    outcomes : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns 'time' and 'event'.
    interventions : np.array
        Boolean numpy array of treatment indicators. True means individual
        was assigned a specific treatment.
    horizons : int or float or list
        Event-horizons at which to evaluate model performance.
    metric : str, default='ibs'
        Metric used to evaluate model performance and tune hyperparameters.
        Options include:
        - 'auc': Dynamic area under the ROC curve
        - 'brs' : Brier Score
        - 'ibs' : Integrated Brier Score
        - 'ctd' : Concordance Index

    Returns
    -----------
    Trained instance of Survival Virtual Twins Phenotyer.

    """

    cf_model = CounterfactualSurvivalRegressionCV(model=self.cf_method,
                                    hyperparam_grid=self.cf_hyperparams)

    self.cf_model = cf_model.fit(features, outcomes, interventions,
                                 horizons, metric)

    times = np.unique(outcomes.time.values)
    cf_predictions = self.cf_model.predict_counterfactual_survival(features,
                                                                   times.tolist())
    horizon = max(horizons)
    ite_estimates = cf_predictions[1] - cf_predictions[0]
    ite_estimates = [estimate[times < horizon] for estimate in ite_estimates]
    times = times[times < horizon]
    # Compute rmst for each sample based on user-specified event-horizon
    rmst = np.array([auc(times, i) for i in ite_estimates])

    if self.phenotyping_method == 'rfr':

      from sklearn.ensemble import RandomForestRegressor

      pheno_model = RandomForestRegressor(**self.phenotyper_hyperparams)
      pheno_model.fit(features.values, rmst)

    self.pheno_model = pheno_model
    self.fitted = True

    return self

  def predict_proba(self, features):

    """Estimate the probability that the Restrictred Mean Survival Time under
    the Treatment group is greater than that under the control group.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array
        a numpy array of the phenogroup probabilties in the format
        [control_group, treated_group].

    """

    phenotype_preds = self.pheno_model.predict(features)
    preds_surv_greater = (phenotype_preds -  phenotype_preds.min()) / (phenotype_preds.max() - phenotype_preds.min())
    preds_surv_less = 1 - preds_surv_greater
    preds = np.array([[preds_surv_less[i], preds_surv_greater[i]]
                      for i in range(len(features))])

    return preds

  def predict(self, features):

    """Extract phenogroups that maximize probability estimates.

    Parameters
    -----------
    features: pd.DataFrame
        a pandas dataframe with rows corresponding to individual samples
        and columns as covariates.

    Returns
    -----------
    np.array
        a numpy array of the phenogroup labels

    """

    return np.argmax(self.predict_proba(features), axis=1)

  def fit_predict(self, features, outcomes, interventions, horizon):

    """Fit and perform phenotyping on a given dataset.

    Parameters
    -----------
    features: pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns as covariates.
    outcomes : pd.DataFrame
        A pandas dataframe with rows corresponding to individual samples
        and columns 'time' and 'event'.
    treatment_indicator : np.array
        Boolean numpy array of treatment indicators. True means individual
        was assigned a specific treatment.
    horizon : np.float
        The event horizon at which to compute the counterfacutal RMST for
        regression.

    Returns
    -----------
    np.array
        a numpy array of the phenogroup labels.

    """

    return self.fit(features, outcomes, interventions, horizon).predict(features)