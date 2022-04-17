
[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/autonlab/DeepSurvivalMachines/branch/master/graph/badge.svg?token=FU1HB5O92D)](https://codecov.io/gh/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/auton-survival?style=social)](https://github.com/autonlab/auton-survival)


<img align=right style="align:right;" src="https://ndownloader.figshare.com/files/34052981" width=30%>

<br>

Python package `auton_survival` provides a flexible API for various problems
in survival analysis, including regression, counterfactual estimation,
and phenotyping.

**For complete details on** `auton-survival` **see**: 
<h3>• <a href="https://www.cs.cmu.edu/~chiragn/papers/auton_survival.pdf">White Paper</a> &nbsp;&nbsp; • <a href="https://autonlab.github.io/auton-survival/">Documentation</a> &nbsp;&nbsp; • <a href="https://nbviewer.org/github/autonlab/auton-survival/tree/master/examples/">Demo Notebooks</a></h3>

What is Survival Analysis?
--------------------------

**Survival Analysis** involves estimating when an event of interest, \( T \)
would take places given some features or covariates \( X \). In statistics
and ML these scenarious are modelled as regression to estimate the conditional
survival distribution, \( \mathbb{P}(T>t|X) \). As compared to typical
regression problems, Survival Analysis differs in two major ways:

* The Event distribution, \( T \) has positive support ie.
  \( T \in [0, \infty) \).
* There is presence of censoring ie. a large number of instances of data are
  lost to follow up.

The Auton Survival Package
---------------------------

The package `auton_survival` is repository of reusable utilities for projects
involving censored Time-to-Event Data. `auton_survival` allows rapid
experimentation including dataset preprocessing, regression, counterfactual
estimation, clustering and phenotyping and propnsity adjusted evaluation.


Survival Regression
-------------------

Currently supported Survival Models are:

### `auton_survival.models.dsm.DeepSurvivalMachines`
### `auton_survival.models.dcm.DeepCoxMixtures`
### `auton_survival.models.cph.DeepCoxPH`


### `auton_survival.estimators`

This module provids a wrapper to model survival datasets with standard
survival (time-to-event) analysis methods. The use of the wrapper allows
a simple standard interface for multiple different survival regression methods.

`auton_survival.estimators` also provides convenient wrappers around other popular
python survival analysis packages to experiment with the following
survival regression estimators

- Random Survival Forests (`pysurvival`):
- Weibull Accelerated Failure Time (`lifelines`) :

### `auton_survival.experiments`

Modules to perform standard survival analysis experiments. This module
provides a top-level interface to run `auton_survival` style experiments
of survival analysis, involving cross-validation style experiments with
multiple different survival analysis models at different horizons of
event times.

The module further eases evaluation by automatically computing the
*censoring adjusted* estimates of the Metrics of interest, like
**Time Dependent Concordance Index** and **Brier Score** with **IPCW**
adjustment.

```python
# auton_survival Style Cross Validation Experiment.
from auton_survival import datasets
features, outcomes = datasets.load_topcat()

from auton_survival.experiments import SurvivalCVRegressionExperiment

# instantiate an auton_survival Experiment by
# specifying the features and outcomes to use.
experiment = SurvivalCVRegressionExperiment(features, outcomes)

# Fit the `experiment` object with a Cox Model
experiment.fit(model='cph')

# Evaluate the performance at time=1 year horizon.
scores = experiment.evaluate(time=1.)

print(scores)
```


Phenotyping and Knowledge Discovery
-----------------------------------

### `auton_survival.phenotyping`

`auton_survival.phenotyping` allows extraction of latent clusters or subgroups
of patients that demonstrate similar outcomes. In the context of this package,
we refer to this task as **phenotyping**. `auton_survival.phenotyping` allows:

- **Unsupervised Phenotyping**: Involves first performing dimensionality
reduction on the inpute covariates \( x \) followed by the use of a clustering
algorithm on this representation.

- **Factual Phenotyping**: Involves the use of structured latent variable
models, `auton_survival.models.dcm.DeepCoxMixtures` or
`auton_survival.models.dsm.DeepSurvivalMachines` to recover phenogroups that
demonstrate differential observed survival rates.

- **Counterfactual Phenotyping**: Involves learning phenotypes that demonstrate
heterogenous treatment effects. That is, the learnt phenogroups have differential
response to a specific intervention. Relies on the specially designed
`auton_survival.models.cmhe.DeepCoxMixturesHeterogenousEffects` latent variable model.

Dataset Loading and Preprocessing
---------------------------------

Helper functions to load and prerocsss various time-to-event data like the
popular `SUPPORT`, `FRAMINGHAM` and `PBC` dataset for survival analysis.


### `auton_survival.datasets`

```python
# Load the SUPPORT Dataset
from auton_survival import dataset
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

# The `cat_feats` and `num_feats` lists would contain all the categorical and
# numerical features in the dataset.

```

Evaluation and Reporting
-------------------------

### `auton_survival.metrics`

Helper functions to generate standard reports for common Survival Analysis tasks.

Citing and References
----------------------

Please cite the following papers if you are using the `auton_survival` package.

[1] [Deep Survival Machines:
Fully Parametric Survival Regression and
Representation Learning for Censored Data with Competing Risks."
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

[2] [Deep Parametric Time-to-Event Regression with Time-Varying Covariates. AAAI
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

[3] [Deep Cox Mixtures for Survival Regression. Conference on Machine Learning for
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

[4] [Counterfactual Phenotyping with Censored Time-to-Events (2022)](https://arxiv.org/abs/2202.11089)</a>

```
  @article{nagpal2022counterfactual,
  title={Counterfactual Phenotyping with Censored Time-to-Events},
  author={Nagpal, Chirag and Goswami, Mononito and Dufendach, Keith and Dubrawski, Artur},
  journal={arXiv preprint arXiv:2202.11089},
  year={2022}
  }
```

## Installation

```console
foo@bar:~$ git clone https://github.com/autonlab/auton_survival
foo@bar:~$ pip install -r requirements.txt
```

Compatibility
-------------
`auton_survival` requires `python` 3.5+ and `pytorch` 1.1+.

To evaluate performance using standard metrics
`auton_survival` requires `scikit-survival`.

Contributing
------------
`auton_survival` is [on GitHub]. Bug reports and pull requests are welcome.

[on GitHub]: https://github.com/autonlab/auton-survival

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
