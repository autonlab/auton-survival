class Reporter:

  def __init__(self, features, outcomes):

    self.features = features
    self.outcomes = outcomes

    self.n = len(features)

  def _get_intersectional_groups(self, cat_vars=None, num_vars=None,
                                 num_vars_quantiles=[0, .5, 1.0]):
    import pandas as pd

    if isinstance(cat_vars, str): cat_vars = [cat_vars]
    if isinstance(num_vars, str): num_vars = [num_vars] 

    assert len(cat_vars+num_vars) != 0, "Please pass intersectional Groups"

    for num_var in num_vars:
      self.features[num_var] = pd.qcut(self.features[num_var], num_vars_quantiles)

    demographics = [str(group) for group in self.features[cat_vars+num_vars].values]
    demographics = np.array(demographics)

    return demographics 

  def report_intersectional(self, intervention, cat_vars=None, num_vars=None,
                            num_vars_quantiles=[0, .5, 1.0]):

    demographics = self._get_intersectional_groups(cat_vars, num_vars, num_vars_quantiles) 

    for demographic in set(demographics):
      
      _outcomes = self.outcomes.loc[demographics==demographic] 
      _interventions = self.features.loc[demographics==demographic][intervention]

      plot_kaplanmeier(_outcomes, _interventions)
