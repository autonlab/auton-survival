r'''


[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/autonlab/DeepSurvivalMachines/branch/master/graph/badge.svg?token=FU1HB5O92D)](https://codecov.io/gh/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/auton-survival?style=social)](https://github.com/autonlab/auton-survival)

<br>


The `auton-survival` Package
---------------------------

The python package `auton-survival` is repository of reusable utilities for projects
involving censored Time-to-Event Data. `auton-survival` provides a flexible APIs 
allowing rapid experimentation including dataset preprocessing, regression, 
counterfactual estimation, clustering and phenotyping and propensity adjusted evaluation.

**For complete details on** `auton-survival` **see**: 
<h3>• <a href="https://www.cs.cmu.edu/~chiragn/papers/auton_survival.pdf">White Paper</a> &nbsp;&nbsp; • <a href="https://autonlab.github.io/auton-survival/">Documentation</a> &nbsp;&nbsp; • <a href="https://nbviewer.org/github/autonlab/auton-survival/tree/master/examples/">Demo Notebooks</a></h3>


<a id="what"></a>

What is Survival Analysis?
--------------------------

**Survival Analysis** involves estimating when an event of interest, \( T \)
would take places given some features or covariates \( X \). In statistics
and ML these scenarious are modelled as regression to estimate the conditional
survival distribution, \( P(T>t|X) \). As compared to typical
regression problems, Survival Analysis differs in two major ways:

* The Event distribution, \( T \) has positive support, 
  \( T in [0, inf) \).
* There is presence of censoring (ie. a large number of instances of data are
  lost to follow up.)

<a id="package"></a>

The Auton Survival Package
---------------------------

The package `auton_survival` is repository of reusable utilities for projects
involving censored Time-to-Event Data. `auton_survival` allows rapid
experimentation including dataset preprocessing, regression, counterfactual
estimation, clustering and phenotyping and propensity-adjusted evaluation.

<a id="regression"></a>

Survival Regression
-------------------

### `auton_survival.models`

Currently supported Survival Models include:

- `auton_survival.models.dsm.DeepSurvivalMachines`
- `auton_survival.models.dcm.DeepCoxMixtures`
- `auton_survival.models.cph.DeepCoxPH`
- `auton_survival.models.cmhe.DeepCoxMixturesHeterogenousEffects`

Training a Deep Cox Proportional Hazards Model with `auton-survival`:

```python
from auton_survival import datasets, preprocessing, models 

# Load the SUPPORT Dataset
outcomes, features = datasets.load_dataset("SUPPORT")

# Preprocess (Impute and Scale) the features
features = preprocessing.Preprocessor().fit_transform(features)

# Train a Deep Cox Proportional Hazards (DCPH) model
model = models.cph.DeepCoxPH(layers=[100])
model.fit(features, outcomes.time, outcomes.event)

# Predict risk at specific time horizons.
predictions = model.predict_risk(features, t=[8, 12, 16])
```

### `auton_survival.estimators` [\[Demo Notebook\]](https://nbviewer.org/github/autonlab/auton-survival/blob/master/examples/Survival%20Regression%20with%20Auton-Survival.ipynb)</a>

This module provides a wrapper `auton_survival.estimators.SurvivalModel` to model
survival datasets with standard survival (time-to-event) analysis methods.
The use of the wrapper allows a simple standard interface for multiple different
survival regression methods.

`auton_survival.estimators` also provides convenient wrappers around other popular
python survival analysis packages to experiment with Random Survival Forests and 
Weibull Accelerated Failure Time regression models.

```python
from auton_survival import estimators

# Train a Deep Survival Machines model using the SurvivalModel class.
model = estimators.SurvivalModel(model='dsm')
model.fit(features, outcomes)

# Predict risk at time horizons.
predictions = model.predict_risk(features, times=[8, 12, 16])
```

### `auton_survival.experiments` [\[Demo Notebook\]](https://nbviewer.org/github/autonlab/auton-survival/blob/master/examples/CV%20Survival%20Regression%20on%20SUPPORT%20Dataset.ipynb)</a>

Modules to perform standard survival analysis experiments. This module
provides a top-level interface to run `auton_survival` style experiments
of survival analysis, involving options for cross-validation and
nested cross-validation style experiments with multiple different survival
analysis models.

The module supports multiple model peroformance evaluation metrics and further 
eases evaluation by automatically computing the *censoring adjusted* estimates,
such as **Time Dependent Concordance Index** and **Brier Score** with **IPCW**
adjustment.

```python
# auton_survival cross-validation experiment.
from auton_survival.datasets import load_dataset

outcomes, features = load_dataset(dataset='SUPPORT')
cat_feats = ['sex', 'income', 'race']
num_feats = ['age', 'resp', 'glucose']

from auton_survival.experiments import SurvivalRegressionCV
# Instantiate an auton_survival Experiment 
experiment = SurvivalRegressionCV(model='cph', num_folds=5, 
                                    hyperparam_grid=hyperparam_grid)

# Fit the `experiment` object with the specified Cox model.
model = experiment.fit(features, outcomes, metric='ibs',
                       cat_feats=cat_feats, num_feats=num_feats)

```

<a id="phenotyping"></a>

Phenotyping and Knowledge Discovery
-----------------------------------

### `auton_survival.phenotyping` [\[Demo Notebook\]](https://nbviewer.org/github/autonlab/auton-survival/blob/master/examples/Phenotyping%20Censored%20Time-to-Events.ipynb)</a>

`auton_survival.phenotyping` allows extraction of latent clusters or subgroups
of patients that demonstrate similar outcomes. In the context of this package,
we refer to this task as **phenotyping**. `auton_survival.phenotyping` allows:

- **Intersectional Phenotyping**: Recovers groups, or phenotypes, of individuals 
over exhaustive combinations of user-specified categorical and numerical features. 

```python
from auton_survival.phenotyping import IntersectionalPhenotyper

# ’ca’ is cancer status. ’age’ is binned into two quantiles.
phenotyper = IntersectionalPhenotyper(num_vars_quantiles=(0, .5, 1.0),
cat_vars=['ca'], num_vars=['age'])
phenotypes = phenotyper.fit_predict(features)
```

- **Unsupervised Phenotyping**: Identifies groups of individuals based on structured 
similarity in the fature space by first performing dimensionality reduction of the 
input covariates, followed by clustering. The estimated probability of an individual 
to belong to a latent group is computed as the distance to the cluster normalized by 
the sum of distance to other clusters.

```python
from auton_survival.phenotyping import ClusteringPhenotyper

# Dimensionality reduction using Principal Component Analysis (PCA) to 8 dimensions.
dim_red_method, = 'pca', 
# We use a Gaussian Mixture Model (GMM) with 3 components and diagonal covariance.
clustering_method, n_clusters = 'gmm', 

# Initialize the phenotyper with the above hyperparameters.
phenotyper = ClusteringPhenotyper(clustering_method=clustering_method,
                                  dim_red_method=dim_red_method,
                                  n_components=n_components,
                                  n_clusters=n_clusters)
# Fit and infer the phenogroups.
phenotypes = phenotyper.fit_predict(features)
```

- **Supervised Phenotyping**: Identifies latent groups of individuals with similar
survival outcomes. This approach can be performed as a direct consequence of training 
the `Deep Survival Machines` and `Deep Cox Mixtures` latent variable survival 
regression estimators using the `predict latent z` method. 

```python
from auton_survival.models.dcm import DeepCoxMixtures [\[Demo Notebook\]]

# Instantiate a DCM Model with 3 phenogroups and a single hidden layer with size 100.
model = DeepCoxMixtures(k = 3, layers = [100])
model.fit(features, outcomes.time, outcomes.event, iters = 100, learning_rate = 1e-4)

# Infer the latent Phenotpyes
latent_z_prob = model.predict_latent_z(features)
phenotypings = latent_z_prob.argmax(axis=1)
```

- **Counterfactual Phenotyping**: Identifies groups of individuals that demonstrate
heterogenous treatment effects. That is, the learnt phenogroups have differential
response to a specific intervention. Relies on the specially designed
`auton_survival.models.cmhe.DeepCoxMixturesHeterogenousEffects` latent variable model.

```python
# Instantiate the CMHE model
model = DeepCoxMixturesHeterogenousEffects(random_seed=random_seed, k=k, g=g, layers=layers)

model = model.fit(features, outcomes.time, outcomes.event, intervention)
zeta_probs = model.predict_latent_phi(x_tr)
zeta = np.argmax(zeta_probs, axis=1)
```

- **Virtual Twins Phenotyping**: Phenotyper that estimates the potential outcomes under treatment and
control using a counterfactual Deep Cox Proportional Hazards model,
followed by regressing the difference of the estimated counterfactual
Restricted Mean Survival Times using a Random Forest regressor.

```python
from auton_survival.phenotyping import SurvivalVirtualTwins

# Instantiate the Survival Virtual Twins
model = SurvivalVirtualTwins(horizon=365)
# Infer the estimated counterfactual phenotype probability.
phenotypes = model.fit_predict(features, outcomes.time, outcomes.event, interventions)
```

<a id="evaluation"></a>

Evaluation and Reporting
-------------------------

### `auton_survival.metrics`

Helper functions to generate standard reports for common Survival Analysis tasks with support for bootstrapped confidence intervals.

- **Regression Metric**: Metrics for survival model performance evaluation:
    - Brier Score 
    - Integrated Brier Score
    - Area under the Receiver Operating Characteristic (ROC) Curve
    - Concordance Index

```python
from auton_survival.metrics import survival_regression_metric

# Infer event-free survival probability from model
predictions = model.predict_survival(features, times)
# Compute Brier Score, Integrated Brier Score
# Area Under ROC Curve and Time Dependent Concordance Index
metrics = ['brs', 'ibs', 'auc', 'ctd']
score = survival_regression_metric(metric='brs', outcomes_train, 
                                   outcomes_test, predictions_test,
                                   times=times)
```

- **Treatment Effect**: Used to compare treatment arms by computing the difference in the following metrics for treatment and control groups:
    - **Time at Risk** (TaR)
    - **Risk at Time**
    - **Restricted Mean Survival Time** (RMST)

```python
from auton_survival.metrics import survival_diff_metric

# Compute the difference in RMST, Risk at Time, and TaR between treatment and control groups
metrics = ['restricted_mean', 'survival_at', 'tar']
effect = survival_diff_metric(metric='restricted_mean', outcomes=outcomes
                              treatment_indicator=treatment_indicator, 
                              weights=None, horizon=120, n_bootstrap=500)
```

- **Phenotype Purity**: Used to measure a phenotyper’s ability to extract subgroups, or phenogroups, with differential survival rates by fitting a Kaplan-Meier estimator within each phenogroup followed by estimating the Brier Score or Integrated Brier Score within each phenogroup.

```python
from auton_survival.metrics import phenotype_purity

# Measure phenotype purity using the Brier Score at event horizons of 1, 2 and 5 years.
phenotype_purity(phenotypes, outcomes, strategy='instantaneous', 
                 time=[365,730,1825])
# Measure phenotype purity using the Integrated Brier score at an event horizon of 5 years.
phenotype_purity(phenotypes, outcomes, strategy='integrated', time=1825)
```

### `auton_survival.reporting`

Helper functions to generate plots for Survival Analysis tasks.

```python
# Plot separate Kaplan Meier survival estimates for phenogroups.
auton_survival.reporting.plot_kaplanmeier(outcomes, groups=phenotypes)

# Plot separate Nelson-Aalen estimates for phenogroups.
auton_survival.reporting.plot_nelsonaalen(outcomes, groups=phenotypes)
```

<a id="preprocess"></a>

Dataset Loading and Preprocessing
---------------------------------

Helper functions to load and preprocess various time-to-event data like the
popular `SUPPORT`, `FRAMINGHAM` and `PBC` dataset for survival analysis.


### `auton_survival.datasets`

```python
# Load the SUPPORT Dataset
from auton_survival.datasets import load_dataset
datasets = ['SUPPORT', 'PBC', 'FRAMINGHAM', 'MNIST', 'SYNTHETIC']
features, outcomes = datasets.load_dataset('SUPPORT')
```

### `auton_survival.preprocessing`
This module provides a flexible API to perform imputation and data
normalization for downstream machine learning models. The module has
3 distinct classes, `Scaler`, `Imputer` and `Preprocessor`. The `Preprocessor`
class is a composite transform that does both Imputing ***and*** Scaling with
a single function call.

```python
# Preprocessing loaded Datasets
from auton_survival import datasets
features, outcomes = datasets.load_topcat()

from auton_survival.preprocessing import Preprocessing
features = Preprocessor().fit_transform(features,
                    cat_feats=['GENDER', 'ETHNICITY', 'SMOKE'],
                    num_feats=['height', 'weight'])

# The `cat_feats` and `num_feats` lists would contain all the categorical and numerical features in the dataset.

```

<a id="ref"></a>

Citing and References
----------------------

**Please cite the following paper if you use the `auton-survival` package:**

[1] [auton-survival: 
an Open-Source Package for Regression, Counterfactual Estimation, Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)](https://arxiv.org/abs/2204.07276)</a>

```
  @article{auton-survival,
  title={auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, Evaluation and Phenotyping with Censored Time-to-Event Data},
  author={Nagpal, Chirag and Potosnak, Willa and Dubrawski, Artur},
  year={2022},
  publisher={arXiv}
  }
```

**Additionally, `auton-survival` implements the following methodologies:**

[2] [Deep Survival Machines:
Fully Parametric Survival Regression and
Representation Learning for Censored Data with Competing Risks.
IEEE Journal of Biomedical and Health Informatics (2021)](https://arxiv.org/abs/2003.01176)</a>

```
  @article{nagpal2021dsm,
  title={Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks},
  author={Nagpal, Chirag and Li, Xinyu and Dubrawski, Artur},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={25},
  number={8},
  pages={3163--3175},
  year={2021},
  publisher={IEEE}
  }
```

[3] [Deep Parametric Time-to-Event Regression with Time-Varying Covariates. AAAI
Spring Symposium (2021)](http://proceedings.mlr.press/v146/nagpal21a.html)</a>

```
  @InProceedings{pmlr-v146-nagpal21a,
  title={Deep Parametric Time-to-Event Regression with Time-Varying Covariates},
  author={Nagpal, Chirag and Jeanselme, Vincent and Dubrawski, Artur},
  booktitle={Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
  }
```

[4] [Deep Cox Mixtures for Survival Regression. Conference on Machine Learning for
Healthcare (2021)](https://arxiv.org/abs/2101.06536)</a>

```
  @inproceedings{nagpal2021dcm,
  title={Deep Cox mixtures for survival regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  booktitle={Machine Learning for Healthcare Conference},
  pages={674--708},
  year={2021},
  organization={PMLR}
  }
```

[5] [Counterfactual Phenotyping with Censored Time-to-Events (2022)](https://arxiv.org/abs/2202.11089)</a>

```
  @article{nagpal2022counterfactual,
  title={Counterfactual Phenotyping with Censored Time-to-Events},
  author={Nagpal, Chirag and Goswami, Mononito and Dufendach, Keith and Dubrawski, Artur},
  journal={arXiv preprint arXiv:2202.11089},
  year={2022}
  }
```

<a id="install"></a>

Compatibility and Installation
------------------------------
`auton_survival` requires `python` 3.5+ and `pytorch` 1.1+.

To evaluate performance using standard metrics
`auton_survival` requires `scikit-survival`.

To install `auton_survival`, clone the following git repository:
```console
foo@bar:~$ git clone https://github.com/autonlab/auton-survival.git
foo@bar:~$ pip install -r requirements.txt
```

<a id="contrib"></a>

Contributing
------------
`auton_survival` is [on GitHub]. Bug reports and pull requests are welcome.

[on GitHub]: https://github.com/autonlab/auton-survival.git

<a id="license"></a>

License
-------
MIT License

Copyright (c) 2022 Carnegie Mellon University, [Auton Lab](http://autonlab.org)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<img align="right" height ="120px" src="https://www.cs.cmu.edu/~chiragn/cmu_logo.jpeg">
<img align="right" height ="110px" src="https://www.cs.cmu.edu/~chiragn/auton_logo.png"> 

<br><br><br><br><br>
<br><br><br><br><br>

'''

from .models.dsm import DeepSurvivalMachines
from .models.dcm import DeepCoxMixtures
from .models.cph import DeepCoxPH, DeepRecurrentCoxPH
from .models.cmhe import DeepCoxMixturesHeterogenousEffects
