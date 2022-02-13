import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, NelsonAalenFitter

from auton_survival.phenotyping import ClusteringPhenotyper, CoxMixturePhenotyper
from auton_survival.preprocessing import Preprocessor

from collections import Counter

from lifelines import KaplanMeierFitter, CoxPHFitter

from sklearn.metrics import roc_curve, auc

import os


def plot_kaplanmeier(outcomes, groups=None, **kwargs):

  if groups is None:
    groups = np.array([1]*len(outcomes))

  for group in set(groups):
    if pd.isna(group): continue
    
    KaplanMeierFitter().fit(outcomes[groups==group]['time'], 
                            outcomes[groups==group]['event']).plot(label=group, **kwargs)

  
def plot_nelsonaalen(outcomes, groups=None, **kwargs):

  if groups is None:
    groups = np.array([1]*len(outcomes))

  for group in set(groups):
    if pd.isna(group): continue
    
    print("Group:", group)

    NelsonAalenFitter().fit(outcomes[groups==group]['time'], 
                            outcomes[groups==group]['event']).plot(label=group, **kwargs)


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


class PhenotypeExperiment:

  def __init__(self, features, outcomes, intervention, cat_vars=None, num_vars=None):

    self.features = features
    self.outcomes = outcomes

    self.cat_vars = cat_vars
    self.num_vars = num_vars

    self.intervention = intervention

    self.n = len(features)

    self.results = None

  def experiment(self, phenotyper='CM', k=3, horizon=None, hypers={}, random_seed=0):

    results = {}

    outcomes = self.outcomes.copy()
    features = self.features.copy()
    intervention = self.intervention

    features_preprocessed = Preprocessor().fit_transform(features, self.cat_vars, self.num_vars)

    if phenotyper == 'CM':

      from dsm import DeepCoxMixtures

      model = DeepCoxMixtures(k=k)
      model.fit(features_preprocessed.values, 
                outcomes['time'].values, 
                outcomes['event'].values, 
                iters=25)
      z_probs = model.predict_latent_z(features_preprocessed.values)

    elif phenotyper == 'clustering':
      raise NotImplementedError()

    Z = np.argmax(z_probs, axis=1)    
    phenotype_size = Counter(Z)
    results['phenotype_size'] = phenotype_size

    def f(x):
      return model.predict_latent_z(x.values)

    import shap

    explainer = shap.explainers.Permutation(f, features_preprocessed)
    shap_values = explainer(features_preprocessed)

    results['shap_values'] = shap_values

    cph = {}
    for z in set(Z):
      features_preprocessed_, outcomes_ =  features_preprocessed.loc[Z==z], outcomes.loc[Z==z]
      cph_ = CoxPHFitter(penalizer=1e-3).fit(features_preprocessed_.join(outcomes_), 'time', 'event')
      cph[z] = cph_.summary[['exp(coef)','exp(coef) lower 95%', 'exp(coef) upper 95%']]

    results['cph_hr'] = cph

    

  def report(self, phenotyper='CM', k=3, horizon=None, hypers={}, random_seed=0):

    os.makedirs("reports/", exist_ok=True)

    outcomes = self.outcomes.copy()
    features = self.features.copy()
    intervention = self.intervention

    features_preprocessed = Preprocessor().fit_transform(features, self.cat_vars, self.num_vars)

    if phenotyper == 'CM':

      from dsm import DeepCoxMixtures

      model = DeepCoxMixtures(k=k)
      model.fit(features_preprocessed.values, 
                outcomes['time'].values, 
                outcomes['event'].values, 
                iters=25)
      z_probs = model.predict_latent_z(features_preprocessed.values)
    
    elif phenotyper == 'clustering':
      raise NotImplementedError()

    Z = np.argmax(z_probs, axis=1)    

    phenotype_size = Counter(Z)

    markers = ['X', 'o', '^']
    colors = ['C'+str(x) for x in range(10)]
    pheno = [chr(x) for x in range(65, 80)]

    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    # Population Level Survival Curve

    plt.figure(figsize=(8,6))

    for i, treat in enumerate(set(features[intervention].values)):
    
      outcomes_ = outcomes.loc[features[intervention]==treat]
      
      KaplanMeierFitter().fit(outcomes_['time'], outcomes_['event']).plot(label=treat, lw=0, 
                                                                          marker=markers[i], markersize=12,
                                                                          markerfacecolor='white',
                                                                          color=colors[i])

    plt.legend(fontsize=18)
    plt.title('CRASH-2 Population Survival (Kaplan-Meier)', fontsize=18)
    plt.ylabel('Survival Probability', fontsize=18)
    plt.xlabel('Time in Days', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.text(16, 0.91, r"HR:$0.897\pm0.032$", fontsize=20)

    plt.xlim(0,30)
    plt.ylim(0.75, 1.)
    plt.grid(True, ls='-.', lw=1)
    plt.savefig("reports/population_survival.pdf")

    plt.figure(figsize=(8,6))

    # Phenogroup Level Survival Curve

    for z in set(Z):
        
      outcomes_ = outcomes.loc[Z==z]
      
      kmf = KaplanMeierFitter().fit(outcomes_['time'], outcomes_['event'])
      
      kmf.plot(label= 'Phenogroup: '+pheno[z], lw=0, 
                marker=markers[z], markersize=12, markerfacecolor='white',
                color=colors[z])

      kmf_ = kmf.confidence_interval_survival_function_.iloc[:31]
      
      print("***")
      print(auc(kmf_.index.values, kmf_['KM_estimate_lower_0.95'].values))
      print(auc(kmf_.index.values, kmf_['KM_estimate_upper_0.95'].values))
      print("***")

    plt.title('Survival Rate by Phenogroup', fontsize=18)
    plt.ylabel('Survival Probability', fontsize=18)
    plt.xlabel('Time in Days', fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=16)
    plt.xlim(0, horizon)

    #plt.ylim(0.4, 1.)
    plt.grid(ls='-.')
    plt.savefig("reports/km_phenotype.pdf")


    def f(x):
      return model.predict_latent_z(x.values)

    import shap

    explainer = shap.explainers.Permutation(f, features_preprocessed)
    shap_values = explainer(features_preprocessed)

    cmaps = ["Blues", "Oranges", "Greens"]

    for z in set(Z):
      plt.figure(figsize=(8,6))
      shap.plots.beeswarm(shap_values[:, :, z], color=plt.get_cmap(cmaps[z]), max_display=10, show=False)
      plt.savefig("".join(["reports/shap_", pheno[z], ".pdf"]))

    from lifelines import CoxPHFitter

    cph = {}
    for z in set(Z):
      features_preprocessed_, outcomes_ =  features_preprocessed.loc[Z==z], outcomes.loc[Z==z]
      cph[z] = CoxPHFitter(penalizer=1e-3).fit(features_preprocessed_.join(outcomes_), 'time', 'event')

    for i in set(Z):
      values = cph[i].summary[['exp(coef)','exp(coef) lower 95%', 'exp(coef) upper 95%']]

      from matplotlib import pyplot as plt

      plt.figure(figsize=(4, 10))

      s = 150

      mean = values['exp(coef)'].values
      mins = values['exp(coef) lower 95%'].values
      maxs = values['exp(coef) upper 95%'].values
      
      for j in range(len(maxs)):
        plt.plot([mins[j], maxs[j]],[j, j],  lw=1, ls=':', zorder=-100, color='k' )
      
      
      plt.scatter(values['exp(coef)'], range(len(values['exp(coef)'])), s=s, marker='s', color='k', facecolor='C'+str(i), zorder=300)
      plt.scatter(values['exp(coef) lower 95%'], range(len(values['exp(coef)'])), marker='<', s=s/1.5,  color='k', facecolor='white')
      plt.scatter(values['exp(coef) upper 95%'], range(len(values['exp(coef)'])), marker='>', s=s/1.5 , color='k', facecolor='white')

      if i==0: plt.yticks(range(len(values)), values.index, fontsize=16, rotation=0)
      else: plt.yticks([])
       
      plt.plot([1, 1], [-1,20], ls=':', color='k', lw=2)

      plt.xlim(0, 2)
      plt.ylim(-.5, 14.5)
      
      plt.xticks(fontsize=18)
      plt.xlabel('Hazard Ratio', fontsize=18)
      plt.title('Phenogroup: '+ pheno[i] , fontsize=18)

      plt.savefig("reports/hr_phenotype_"+str(pheno[i])+".pdf")

  def create_report(self):
    import subprocess
    subprocess.run("Rscript -e 'rmarkdown::render(\"template.Rmd\", output_format=\"pdf_document\")'", shell=True)

