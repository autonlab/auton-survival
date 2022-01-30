import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, NelsonAalenFitter
from copy import deepcopy

from sklearn import cluster, decomposition, mixture
from auton_survival.utils import _get_method_kwargs




class Phenotyper:

  def __init__(self, random_seed=0):

    self.random_seed = random_seed
    self.fitted = False

class IntersectionalPhenotyper(Phenotyper):

  """A phenotyper using all possible combinations of specified variables.
  """

  def __init__(self, cat_vars=None, num_vars=None, num_vars_quantiles=[0, .5, 1.0]):
    
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
    
    self.cut_bins = {}
    self.min_max = {}

    for num_var in self.num_vars:

      binmin, binmax = float(features[num_var].min()), float(features[num_var].max())

      self.min_max[num_var] = binmin, binmax
      _, self.cut_bins[num_var] = pd.qcut(features[num_var], self.num_vars_quantiles, retbins=True)

    self.fitted = True
    return self

  def _rename(self, demographics):

    ft_names = self.cat_vars + self.num_vars
    renamed = []
    for i in range(len(demographics)):
      row = []
      for j in range(len(ft_names)):
        row.append(ft_names[j]+":"+str(demographics[i][j]))
      renamed.append(" & ".join(row))
    return renamed

  def phenotype(self, features):

    assert self.fitted, "Phenotyper must be `fitted` before calling `phenotype`."
    features = deepcopy(features)
    
    for num_var in self.num_vars:
      
      var_min, var_max = self.min_max[num_var]

      features[num_var][features[num_var]>=var_max] = var_max 
      features[num_var][features[num_var]<=var_min] = var_min
      
      features[num_var] = pd.cut(features[num_var], self.cut_bins[num_var], include_lowest=True)

    demographics = [group.tolist() for group in features[self.cat_vars+self.num_vars].values]
    demographics = self._rename(demographics)

    demographics = np.array(demographics)

    return demographics
  
  def fit_phenotype(self, features):
    return self.fit(features).phenotype(features)


class ClusteringPhenotyper(Phenotyper):
  
  """A Phenotyper that first reduces feature dimensionality followed by clustering.
  """

  _VALID_DIMRED_METHODS = ['pca', 'kpca', 'nnmf']
  _VALID_CLUSTERING_METHODS = ['kmeans', 'dbscan', 'gmm', 'hierarchical']

  def __init__(self, clustering_method = 'kmeans', dim_red_method = None, random_seed=0, **kwargs):

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
    d_kwargs = _get_method_kwargs(dim_red_model, kwargs)

    if clustering_method == 'gmm': 
      if 'covariance_type' not in c_kwargs:
        c_kwargs['covariance_type'] = 'diag'
      c_kwargs['n_components'] = c_kwargs.get('n_clusters', 3) 
    if dim_red_method == 'kpca':
      if 'kernel' not in d_kwargs:
        d_kwargs['kernel'] = 'rbf'
        d_kwargs['n_jobs'] = -1
        d_kwargs['max_iter'] = 500

    self.dim_red_model = dim_red_model(**d_kwargs)
    self.clustering_model = clustering_model(**c_kwargs)

  def fit(self, features):
    
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

    #TODO:MAYBE DO THIS IN LOG SPACE?

    negative_exp_distances = np.exp(-self.clustering_model.transform(features))
    probs = negative_exp_distances/negative_exp_distances.sum(axis=1).reshape((-1, 1))
    
    #assert int(np.sum(probs)) == len(probs), 'Not valid probabilities'

    return probs

  def phenotype(self, features):
 
    assert self.fitted, "Phenotyper must be `fitted` before calling `phenotype`."
 
    if self.dim_red_method is not None:
      features =  self.dim_red_model.transform(features)
    if self.clustering_method == 'gmm': 
      return self.clustering_model.predict_proba(features) 
    elif self.clustering_method == 'kmeans':
      return self._predict_proba_kmeans(features)
 
  def fit_phenotype(self, features):
    return self.fit(features).phenotype(features)


class CoxMixturePhenotyper(Phenotyper):

  def __init__(self):
    raise NotImplementedError()

