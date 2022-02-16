'''

[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/autonlab/DeepSurvivalMachines/branch/master/graph/badge.svg?token=FU1HB5O92D)](https://codecov.io/gh/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/DeepSurvivalMachines?style=social)](https://github.com/autonlab/DeepSurvivalMachines)


Python package `auton_survival` provides a flexible API for various problems
in survival analysis, including regression, counterfactual estimation,
and phenotyping.

What is Survival Analysis?
------------------------

**Survival Analysis** involves estimating when an event of interest, \( T \)
would take places given some features or covariates \( X \). In statistics
and ML these scenarious are modelled as regression to estimate the conditional
survival distribution, \( \mathbb{P}(T>t|X) \). As compared to typical
regression problems, Survival Analysis differs in two major ways:

* The Event distribution, \( T \) has positive support ie.
  \( T \in [0, \infty) \).
* There is presence of censoring ie. a large number of instances of data are
  lost to follow up.

# Auton Survival

Repository of reusable code utilities for Survival Analysis projects.

## `auton_survival.datasets`

Helper functions to load various trial data like `TOPCAT`, `BARI2D` and `ALLHAT`.

```python
# Load the TOPCAT Dataset
from auton_survival import dataset
features, outcomes = datasets.load_topcat()
```

## `auton_survival.preprocessing`
This module provides a flexible API to perform imputation and data
normalization for downstream machine learning models. The module has
3 distinct classes, `Scaler`, `Imputer` and `Preprocessor`. The `Preprocessor`
class is a composite transform that does both Imputing ***and*** Scaling.

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

## `auton_survival.estimators`

This module provids a wrapper to model BioLINNC datasets with standard
survival (time-to-event) analysis methods.
The use of the wrapper allows a simple standard interface for multiple different
survival models, and also helps standardize experiments across various differents
research areas.

Currently supported Survival Models are:

- Cox Proportional Hazards Model (`lifelines`):
- Random Survival Forests (`pysurvival`):
- Weibull Accelerated Failure Time (`lifelines`) :
- Deep Survival Machines: **Not Implemented Yet**
- Deep Cox Mixtures: **Not Implemented Yet**


```python
# Preprocessing loaded Datasets
from auton_survival import datasets
features, outcomes = datasets.load_topcat()

from auton_survival.estimators import Preprocessing
features = Preprocessing().fit_transform(features)
```


## `auton_survival.experiments`

Modules to perform standard survival analysis experiments. This module
provides a top-level interface to run `auton_survival` Style experiments
of survival analysis, involving cross-validation style experiments with
multiple different survival analysis models at different horizons of event times.

The module further eases evaluation by automatically computing the
*censoring adjusted* estimates of the Metrics of interest, like
**Time Dependent Concordance Index** and **Brier Score** with **IPCW** adjustment.

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

## `auton_survival.reporting`

Helper functions to generate standard reports for popular Survival Analysis problems.

## Installation

```console
foo@bar:~$ git clone https://github.com/autonlab/auton_survival
foo@bar:~$ pip install -r requirements.txt
```

Compatibility
-------------
`dsm` requires `python` 3.5+ and `pytorch` 1.1+.

To evaluate performance using standard metrics
`dsm` requires `scikit-survival`.

Contributing
------------
`dsm` is [on GitHub]. Bug reports and pull requests are welcome.

[on GitHub]: https://github.com/chiragnagpal/deepsurvivalmachines

License
-------
MIT License

Copyright (c) 2020 Carnegie Mellon University, [Auton Lab](http://autonlab.org)

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


<img style="float: right;" width ="200px" src="https://www.cmu.edu/brand/\
downloads/assets/images/wordmarks-600x600-min.jpg">
<img style="float: right;padding-top:50px" src="https://www.autonlab.org/\
user/themes/auton/images/AutonLogo.png">

<br><br><br><br><br>
<br><br><br><br><br>

'''

from .models.dsm import DeepSurvivalMachines
from .models.dcm import DeepCoxMixtures
from .models.cph import DeepCoxPH, DeepRecurrentCoxPH
from .models.cmhe import DeepCoxMixturesHeterogenousEffects
