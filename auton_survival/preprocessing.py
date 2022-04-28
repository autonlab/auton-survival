import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

class Imputer:

  r"""A class to impute missing values in the input features.

    Real world datasets are often subject to missing covariates.
    Imputation replaces the missing values allowing downstream experiments.
    This class allows multiple strategies to impute both categorical and
    numerical/continuous covariates.

    For categorical features, the class allows:

    - **replace**: Replace all null values with a user specificed constant.
    - **ignore**: Keep all missing values as is.
    - **mode**: Replace null values with most commonly occurring variable.

    For numerical/continuous features,
    the user can choose between the following strategies:

    - **mean**: Replace all missing values with the mean in the column.
    - **median**: Replace all missing values with the median in the column.
    - **knn**: Use a k Nearest Neighbour model to predict the missing value.
    - **missforest**: Use the MissForest model to predict the null values.

    Parameters
    ----------
    cat_feat_strat : str
        Strategy for imputing categorical features.
        One of `'replace'`, `'ignore'`, `'mode'`. Default is `ignore`.
    num_feat_strat : str
        Strategy for imputing numerical/continuous features.
        One of `'mean'`, `'median'`, `'knn'`, `'missforest'`. Default is `mean`.
    remaining : str
        Strategy for handling remaining columns.
        One of `'ignore'`, `'drop'`. Default is `drop`.
    """

  _VALID_CAT_IMPUTE_STRAT = ['replace', 'ignore', 'mode']
  _VALID_NUM_IMPUTE_STRAT = ['mean', 'median', 'knn', 'missforest']
  _VALID_REMAINING_STRAT = ['ignore', 'drop']

  def __init__(self, cat_feat_strat='ignore',
                     num_feat_strat='mean',
                     remaining='drop'):

    assert cat_feat_strat in Imputer._VALID_CAT_IMPUTE_STRAT
    assert num_feat_strat in Imputer._VALID_NUM_IMPUTE_STRAT
    assert remaining in Imputer._VALID_REMAINING_STRAT

    self.cat_feat_strat = cat_feat_strat
    self.num_feat_strat = num_feat_strat
    self.remaining = remaining

    self.fitted = False

  def fit(self, data, cat_feats=None, num_feats=None,
          fill_value=-1, n_neighbors=5, **kwargs):

    if cat_feats is None: cat_feats = []
    if num_feats is None: num_feats = []

    assert len(cat_feats + num_feats) != 0, "Please specify \
    categorical and numerical features."

    self._cat_feats = cat_feats
    self._num_feats = num_feats

    df = data.copy()

    ####### REMAINING VARIABLES
    remaining_feats = set(df.columns) - set(cat_feats) - set(num_feats)

    if self.remaining == 'drop':
      df = df.drop(columns=list(remaining_feats))

    ####### CAT VARIABLES
    if self._cat_feats:
      if self.cat_feat_strat == 'replace':
        self._cat_base_imputer = SimpleImputer(strategy='constant',
                                               fill_value=fill_value).fit(df[cat_feats])
      elif self.cat_feat_strat == 'mode':
        self._cat_base_imputer = SimpleImputer(strategy='most_frequent',
                                               fill_value=fill_value).fit(df[cat_feats])

    ####### NUM VARIABLES
    if self._num_feats:
      if self.num_feat_strat == 'mean':
        self._num_base_imputer = SimpleImputer(strategy='mean').fit(df[num_feats])
      elif self.num_feat_strat == 'median':
        self._num_base_imputer = SimpleImputer(strategy='median').fit(df[num_feats])
      elif self.num_feat_strat == 'knn':
        self._num_base_imputer = KNNImputer(n_neighbors=n_neighbors,
                                            **kwargs).fit(df[num_feats])
      elif self.num_feat_strat == 'missforest':
        from missingpy import MissForest
        self._num_base_imputer = MissForest(**kwargs).fit(df[num_feats])

    self.fitted = True
    return self

  def transform(self, data):

    all_feats = self._cat_feats + self._num_feats
    assert len(set(data.columns)^set(all_feats)) == 0, "Passed columns don't \
    match columns trained on !!! "
    assert self.fitted, "Model is not fitted yet !!!"

    df = data.copy()

    if self.cat_feat_strat != 'ignore':
      if len(self._cat_feats):
        df[self._cat_feats] = self._cat_base_imputer.transform(df[self._cat_feats])

    if len(self._num_feats):
      df[self._num_feats] = self._num_base_imputer.transform(df[self._num_feats])

    return df

  def fit_transform(self, data, cat_feats, num_feats,
                          fill_value=-1, n_neighbors=5, **kwargs):

    """Imputes dataset using imputation strategies.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataframe to be imputed.
    cat_feats: list
        List of categorical features.
    num_feats: list
        List of numerical/continuous features.
    fill_value: int
        Value to be filled if `cat_feat_strat='replace'`.
    n_neighbors: int
        Number of neighbors to be used if `num_feat_strat='knn'`.
    **kwargs
        Passed on.

    Returns:
        pandas.DataFrame: Imputed dataset.
    """

    return self.fit(data, cat_feats=cat_feats,
                          num_feats=num_feats,
                          fill_value=fill_value).transform(data)


class Scaler:

  """Scaler to rescale numerical features.

  For scaling, the user can choose between the following strategies:

  - **standard**: Perform the standard scaling method.
  - **minmax**: Perform the minmax scaling method.
  - **none**: Do not perform scaling.

  Parameters
  ----------
  scaling_strategy: str
      Strategy to use for scaling numerical/continuous data.
      One of `'standard'`, `'minmax'`, `'none'`.
      Default is `standard`.
  """

  _VALID_SCALING_STRAT = ['standard', 'minmax', 'none']

  def __init__(self, scaling_strategy='standard'):

    assert scaling_strategy in Scaler._VALID_SCALING_STRAT

    self.scaling_strategy = scaling_strategy

  def fit(self, data, num_feats=None):
    """Fits scaler to dataset using scaling strategy.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe to be scaled.
    feats: list
        List of numerical/continuous features to be scaled.
        **NOTE**: if left empty, all features are interpreted as numerical.
    Returns:
        Fitted instance of scaler.
    """

    self._num_feats = num_feats

    df = data.copy()

    if self.scaling_strategy == 'standard':
      scaler = StandardScaler()
    elif self.scaling_strategy == 'minmax':
      scaler = MinMaxScaler()
    else:
      scaler = None

    if scaler:
      if self._num_feats:
        self.scaler = scaler.fit(df[self._num_feats])
      else:
        self.scaler = scaler.fit(df)

    self.fitted = True
    return self

  def transform(self, data):
    """Scales data using scaling strategy.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe to be scaled.
    feats: list
        List of numerical/continuous features to be scaled.
        **NOTE**: if left empty, all features are interpreted as numerical.
    Returns:
        Fitted instance of scaler.
    """

    df = data.copy()

    if self._num_feats:
      df[self._num_feats] = self.scaler.transform(df[self._num_feats])
    else:
      df[df.columns] = self.scaler.transform(df)

    return df

  def fit_transform(self, data, num_feats=[]):
    """Fits a scaler and rescales a dataset using a standard rescaling
    strategy.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe to be scaled.
    feats: list
        List of numerical/continuous features to be scaled.
        **NOTE**: if left empty, all features are interpreted as numerical.

    Returns:
        pandas.DataFrame: Scaled dataset.
    """

    # if feats is None, scale everything, else only the columns specified.

    df = data.copy()

    if self.scaling_strategy == 'standard':
      scaler = StandardScaler()
    elif self.scaling_strategy == 'minmax':
      scaler = MinMaxScaler()
    else:
      scaler = None

    if scaler is not None:
      if num_feats:
        df[num_feats] = scaler.fit_transform(df[num_feats])
      else: df[df.columns] = scaler.fit_transform(df)

    return df

class Preprocessor:

  """ A composite transform involving both scaling and preprocessing.

  Parameters
  ----------
  cat_feat_strat: str
      Strategy for imputing categorical features.
  num_feat_strat: str
      Strategy for imputing numerical/continuous features.
  scaling_strategy: str
      Strategy to use for scaling numerical/continuous data.
  one_hot: bool
      Whether to apply one hot encoding to the data.
  remaining: str
      Strategy for handling remaining columns.
  """

  def __init__(self, cat_feat_strat='ignore',
                     num_feat_strat='mean',
                     scaling_strategy='standard',
                     one_hot=True,
                     remaining='drop'):

    self.one_hot = one_hot

    self.imputer = Imputer(cat_feat_strat=cat_feat_strat,
                           num_feat_strat=num_feat_strat,
                           remaining=remaining)

    self.scaler = Scaler(scaling_strategy=scaling_strategy)

  def fit(self, data, cat_feats, num_feats,
            fill_value=-1, n_neighbors=5, **kwargs):
    """Fit imputer and scaler to dataset."""

    self._cat_feats = cat_feats
    self._num_feats = num_feats

    self.imputer.fit(data, 
                     cat_feats=self._cat_feats,
                     num_feats=self._num_feats, 
                     fill_value=fill_value,
                     n_neighbors=n_neighbors,
                     **kwargs)

    data_imputed = self.imputer.transform(data)

    self.scaler.fit(data_imputed, num_feats=self._num_feats)

    self.fitted = True
    return self

  def transform(self, data):
    """Impute and scale the dataset."""

    data_imputed = self.imputer.transform(data)
    data_transformed = self.scaler.transform(data_imputed)

    if self.one_hot:
      data_transformed[self._cat_feats] = data_transformed[self._cat_feats].astype('category')
      data_transformed = pd.get_dummies(data_transformed,
                                        dummy_na=False,
                                        drop_first=True)
    return data_transformed

  def fit_transform(self, data, cat_feats, num_feats, 
                    fill_value=-1, n_neighbors=5, **kwargs):
    """Imputes and scales dataset.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataframe to be imputed.
    cat_feats: list
        List of categorical features.
    num_feats: list
        List of numerical/continuous features.
    one_hot: bool
        Indicating whether to perform one-hot encoding.
    fill_value: int
        Value to be filled if `cat_feat_strat='replace'`.
    n_neighbors: int
        Number of neighbors to be used if `num_feat_strat='knn'`.
    **kwargs
        Passed on.

    Returns:
    --------
        pandas.DataFrame: Imputed and scaled dataset.
    """

    imputer_output = self.imputer.fit_transform(data,
                                                cat_feats=cat_feats,
                                                num_feats=num_feats,
                                                fill_value=fill_value,
                                                n_neighbors=n_neighbors,
                                                **kwargs)
    output = self.scaler.fit_transform(imputer_output, num_feats=num_feats)

    if self.one_hot:
      output[cat_feats] = output[cat_feats].astype('category')
      output = pd.get_dummies(output, dummy_na=False, drop_first=True)

    return output
