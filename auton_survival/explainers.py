from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, average_precision_score

class Explainer:

  def __init__(self):
    return
	

class DecisionTreeExplainer(Explainer):

  def __init__(self, random_seed=0, **kwargs):

    self.random_seed = random_seed 
    self.fitted = False
    self.kwargs = kwargs

  def fit(self, features, phenotype):
    
    model = DecisionTreeClassifier(random_state=self.random_seed, **self.kwargs)
    self._model = model.fit(features, phenotype)
    self.fitted = True
    self.feature_names = features.columns 
    return self

  def phenotype(self, features):
    return self._model.predict(features)

  def explain(self, return_tree=False, **kwargs):
    tree = plot_tree(self._model, feature_names=self.feature_names, **kwargs)
    if return_tree: return tree

  def evaluate(self, features, phenotypes):
    predicted_phenotypes = self._model.predict_proba(features)
    return roc_auc_score(phenotypes, predicted_phenotypes)



