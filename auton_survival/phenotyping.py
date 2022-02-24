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

"""Functions to identify subgroups based on observable characteristics for use 
in comparing survival probabilities among groups."""

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, NelsonAalenFitter
from copy import deepcopy

from sklearn import cluster, decomposition, mixture

from auton_survival.utils import _get_method_kwargs


class Phenotyper:
  """Base class for all phenotyping methods"""

  def __init__(self, random_state=0):

    self.random_state = random_state
    self.fitted = False

class IntersectionalPhenotyper(Phenotyper):
  """A phenotyper that creates groups based on all possible combinations of specified categorical and numerical variables.
  
  Parameters
  -----------
  cat_vars : list of python str(s), default=None
      The names of categorical independent variables inputed as python strings in a list
  num_vars : list of python str(s), default=None
     The names of continuous independent variables inputed as python strings in a list  
  num_vars_quantiles : int or list-like of float, default=(0, .5, 1.0)
      Either the number of quantiles as an integer or a list-like of quantile floats (inclusive of 0 and 1) 
      used to discretize continuous variables into equal-sized bins.
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables
  phenotypes : list
      List of lists containing all possible combinations of specified categorical and numerical variable values
      
  """

  def __init__(self, cat_vars=None, num_vars=None,
               num_vars_quantiles=(0, .5, 1.0)):
    
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
  """Obtain discretized value bins based on defined quantiles for continous variables. 
  Record continuous variable minimum and maximum for data clipping.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  Trained instance of intersectional phenotyper
        
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

  def phenotype(self, features):
  """Clip and bin continuous values. Create phenotypes from all possible combinations of specified categorical and numerical variables.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  np.array : A numpy array containing a list of strings that define subgroups from all possible combinations 
      of specified categorical and numerical variables.
        
  """

    assert self.fitted, "Phenotyper must be `fitted` before calling `phenotype`."
    features = deepcopy(features)

    for num_var in self.num_vars:
      
      var_min, var_max = self.min_max[num_var]

      features[num_var][features[num_var]>=var_max] = var_max 
      features[num_var][features[num_var]<=var_min] = var_min

      features[num_var] = pd.cut(features[num_var], self.cut_bins[num_var],
                                 include_lowest=True)

    phenotypes = [group.tolist() for group in features[self.cat_vars+self.num_vars].values]
    phenotypes = self._rename(phenotypes)

    phenotypes = np.array(phenotypes)

    return phenotypes

  def _rename(self, phenotypes):
  """Create phenotype category names from all possible combinations of specified categorical and numerical variables.
    
  Parameters
  -----------
  phenotypes : list
      List of lists containing all possible combinations of specified categorical and numerical variable values
        
  Returns
  -----------
  list : python list of a list of strings that define subgroups.
        
  """

    ft_names = self.cat_vars + self.num_vars
    renamed = []
    for i in range(len(demographics)):
        row = []
        for j in range(len(ft_names)):
            row.append(ft_names[j]+":"+str(demographics[i][j]))
        renamed.append(" & ".join(row))
    return renamed

  def fit_phenotype(self, features):
  """Train an instance of the intersectinal phenotyper and return subgroup .
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  np.array : A numpy array containing a list of strings that define subgroups from all possible combinations of specified 
      categorical and numerical variables.
        
  """

    return self.fit(features).phenotype(features)


class ClusteringPhenotyper(Phenotyper):
  """Phenotyper that performs dimensionality reduction followed by clustering. Learned clusters are considered phenotypes and used to 
  group samples based on similar observable characteristics.

  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
  clustering_method : str, default='kmeans'
      The clustering method applied for phenotyping. Options include:
      - 'kmeans' : K-Means clustering'
      - 'dbscan' : Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
      - 'gmm' : Gaussian Mixture
      - 'hierarchical' : AgglomerativeClustering  
  dim_red_method : str, default=None
      The dimensionality reductions method applied. Options include:
      - 'pca' : Principal Component Analysis
      - 'kpca' : Kernel Principal Component Analysis
      - 'nnmf' : Non-Negative Matrix Factorization 
      - None : dimensionality reduction is not applied. 
  random_state : int, default=0
      Controls the randomness and reproducibility of called functions  
  kwargs : dict
      Additional arguments for dimensionality reduction and clustering
      Please include dictionary key and item pairs specified by the following sci-kit learn modules:
      'pca' : sklearn.decomposition.PCA
      'nnmf' : sklearn.decomposition.NMF
      'kpca' : sklearn.decomposition.KernelPCA  
      'kmeans' : sklearn.cluster.KMeans
      'dbscan' : sklearn.cluster.DBSCAN
      'gmm' : sklearn.mixture.GaussianMixture
      'hierarchical' : sklearn.cluster.AgglomerativeClustering
        
  """

  _VALID_DIMRED_METHODS = ['pca', 'kpca', 'nnmf', None]
  _VALID_CLUSTERING_METHODS = ['kmeans', 'dbscan', 'gmm', 'hierarchical']

  def __init__(self, clustering_method = 'kmeans', dim_red_method = None, random_state=0, **kwargs):

    assert clustering_method in ClusteringPhenotyper._VALID_CLUSTERING_METHODS, "Please specify a valid Clustering method"
    assert dim_red_method in ClusteringPhenotyper._VALID_DIMRED_METHODS, "Please specify a valid Dimensionality Reduction method"

    # Raise warning if "hierarchical" is used with dim_redcution
    if (clustering_method in ['hierarchical']) and (dim_red_method is not None):
      print("WARNING: Are you sure you want to run hierarchical clustering on decomposed features?. Such behaviour is atypical.") 

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

    self.clustering_model = clustering_model(**c_kwargs)
    if dim_red_method is not None:
      d_kwargs = _get_method_kwargs(dim_red_model, kwargs)
      if dim_red_method == 'kpca':
        if 'kernel' not in d_kwargs:
          d_kwargs['kernel'] = 'rbf'
          d_kwargs['n_jobs'] = -1
          d_kwargs['max_iter'] = 500

      self.dim_red_model = dim_red_model(**d_kwargs)

  def fit(self, features):
  """Perform dimensionality reduction and train an instance of the clustering algorithm.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  Trained instance of clustering phenotyper
        
  """
    
    if self.dim_red_method is not None: 
      print("Fitting the following Dimensionality Reduction Model:\n", self.dim_red_model)
      self.dim_red_model = self.dim_red_model.fit(features)
      features = self.dim_red_model.transform(features)

    else:
      print("No Dimensionaity reduction specified...\n Proceeding to learn clusters with the raw features...")

    print("Fitting the following Clustering Model:\n", self.clustering_model)
    self.clustering_model = self.clustering_model.fit(features)
    self.fitted = True

    return self

  def _predict_proba_kmeans(self, features):
  """Obtain the distances to the kmeans cluster centers for each sample.
  Compute the fraction of distance to each cluster out of the total distance to all clusters to estimate
  the probability of sample association to learned clusters, or subgroups.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas DataFrame with rows corresponding to samples and columns corresponding to independent variables
        
  Returns
  -----------
  np.array : A numpy array of probability estimates of sample association to learned subgroups. 
        
  """

    #TODO:MAYBE DO THIS IN LOG SPACE?

    negative_exp_distances = np.exp(-self.clustering_model.transform(features))
    probs = negative_exp_distances/negative_exp_distances.sum(axis=1).reshape((-1, 1))
    
    #assert int(np.sum(probs)) == len(probs), 'Not valid probabilities'

    return probs

  def phenotype(self, features):
  """Peform dimensionality reduction, clustering, and create phenotypes based on the probability estimates
  of sample association to learned clusters, or subgroups.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  np.array : A numpy array of the probability estimates of sample association to learned subgroups.
        
  """
 
    assert self.fitted, "Phenotyper must be `fitted` before calling `phenotype`."
 
    if self.dim_red_method is not None:
      features =  self.dim_red_model.transform(features)
    if self.clustering_method == 'gmm': 
      return self.clustering_model.predict_proba(features) 
    elif self.clustering_method == 'kmeans':
      return self._predict_proba_kmeans(features)
 
  def fit_phenotype(self, features):
  """Train an instance of the clustering phenotyper and identify subgroups based on the probability 
  estimates of sample association to learned clusters, or subgroups.
    
  Parameters
  -----------
  features : pd.DataFrame
      A pandas dataframe with rows corresponding to samples and columns corresponding to independent variables.
        
  Returns
  -----------
  np.array : A numpy array of the probability estimates of sample association to learned clusters.
        
  """

    return self.fit(features).phenotype(features)

class CoxMixturePhenotyper(Phenotyper):
"""TO-DO: ADD DESCRIPTION

"""

  def __init__(self):
  """TO-DO: ADD DESCRIPTION
        
  """

    raise NotImplementedError()

