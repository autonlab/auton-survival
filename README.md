[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/autonlab/DeepSurvivalMachines/branch/master/graph/badge.svg?token=FU1HB5O92D)](https://codecov.io/gh/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/auton-survival?style=social)](https://github.com/autonlab/auton-survival)

<img align="right" width=50% src=https://ndownloader.figshare.com/files/26367844>

Package: `auton-survival`
-------------

Python package `auton-survival` provides an API to train the Deep Survival Machines
and associated models for problems in survival analysis. The underlying model
is implemented in `pytorch`.

For full documentation of the module, please see https://autonlab.github.io/auton-survival/

What is Survival Analysis?
------------------------

**Survival Analysis** involves estimating when an event of interest, _*T*_
would take place given some features or covariates _*X*_. In statistics
and ML, these scenarios are modelled as regression to estimate the conditional
survival distribution, _*P*_(_T_>t|_X_).   
As compared to typical regression problems, Survival Analysis differs in two major ways:

* The Event distribution, _*T*_ has positive support i.e. _*T*_ ∈ [0, ∞).
* There is presence of censoring i.e. a large number of instances of data are
  lost to follow up.

Deep Survival Machines
----------------------
<img width=100% src=https://ndownloader.figshare.com/files/25259852>


**Deep Survival Machines (DSM)** is a **fully parametric** approach to model
Time-to-Event outcomes in the presence of Censoring, first introduced in
[\[1\]](https://arxiv.org/abs/2003.01176).
In the context of Healthcare ML and Biostatistics, this is known as 'Survival
Analysis'. The key idea behind Deep Survival Machines is to model the
underlying event outcome distribution as a mixure of some fixed \( K \)
parametric distributions. The parameters of these mixture distributions as
well as the mixing weights are modelled using Neural Networks.

#### Usage Example

```python
from dsm import DeepSurvivalMachines
model = DeepSurvivalMachines()
model.fit()
model.predict_risk()
```

Recurrent Deep Survival Machines
--------------------------------
<img width=100% src=https://ndownloader.figshare.com/files/28329918>


**Recurrent Deep Survival Machines (RDSM)** builds on the original **DSM**
model and allows for learning of representations of the input covariates using
**Recurrent Neural Networks** like **LSTMs, GRUs**. Deep Recurrent Survival
Machines is a natural fit to model problems where there are time dependendent
covariates.


Deep Convolutional Survival Machines
------------------------------------

Predictive maintenance and medical imaging sometimes requires to work with
image streams. Deep Convolutional Survival Machines extends **DSM** and
**DRSM** to learn representations of the input image data using
convolutional layers. If working with streaming data, the learnt
representations are then passed through an LSTM to model temporal dependencies
before determining the underlying survival distributions.

> :warning: **Not Implemented Yet!**

Deep Cox Mixtures
------------------
<img width=100% align="center" src=https://ndownloader.figshare.com/files/28316535>

The Cox Mixture involves the assumption that the survival function
of the individual to be a mixture of K Cox Models. Conditioned on each
subgroup Z=k; the PH assumptions are assumed to hold and the baseline
hazard rates is determined non-parametrically using an spline-interpolated
Breslow's estimator.
For full details on Deep Cox Mixture, refer to the paper:

<a href="https://proceedings.mlr.press/v149/nagpal21a">Deep Cox Mixtures
for Survival Regression. Machine Learning in Health Conference (2021)</a>


Installation
------------

```console
foo@bar:~$ git clone https://github.com/autonlab/auton-survival.git
foo@bar:~$ cd auton-survival
foo@bar:~$ pip install -r requirements.txt
```

Examples
---------

1. [Deep Survival Machines on the SUPPORT Dataset](https://nbviewer.jupyter.org/github/autonlab/DeepSurvivalMachines/blob/master/examples/DSM%20on%20SUPPORT%20Dataset.ipynb)
2. [Recurrent Deep Survival Machines on the PBC Dataset](https://nbviewer.org/github/autonlab/DeepSurvivalMachines/blob/master/examples/RDSM%20on%20PBC%20Dataset.ipynb)


References
----------

Please cite the following papers if you are using the `dsm` package.

[1] [Deep Survival Machines:
Fully Parametric Survival Regression and
Representation Learning for Censored Data with Competing Risks.
IEEE Journal of Biomedical \& Health Informatics (2021)](https://arxiv.org/abs/2003.01176)</a>

```
  @article{nagpal2021deep,
  title={Deep Survival Machines: Fully Parametric Survival Regression and\
  Representation Learning for Censored Data with Competing Risks},
  author={Nagpal, Chirag and Li, Xinyu and Dubrawski, Artur},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2021}
  }
```


[2] [Deep Parametric Time-to-Event Regression with Time-Varying Covariates.
AAAI Spring Symposium (2021)](http://proceedings.mlr.press/v146/nagpal21a.html)</a>

```
@InProceedings{pmlr-v146-nagpal21a,
  title = 	 {Deep Parametric Time-to-Event Regression with Time-Varying Covariates},
  author =       {Nagpal, Chirag and Jeanselme, Vincent and Dubrawski, Artur},
  booktitle = 	 {Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  }
```

[3] [Deep Cox Mixtures for Survival Regression. Machine Learning for Healthcare (2021)](https://proceedings.mlr.press/v149/nagpal21a)

```
@InProceedings{nagpal2021dcm,
  title={Deep Cox Mixtures for Survival Regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  booktitle={Proceedings of the 6th Machine Learning for Healthcare Conference},
  pages={674--708},
  year={2021},
  volume={149},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
}
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

Copyright (c) 2020 Carnegie Mellon University, [Auton Lab](http://www.autonlab.org)

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

<img align="right" height ="120px" src="https://www.cs.cmu.edu/~chiragn/cmu_logo.jpeg">
<img align="right" height ="110px" src="https://www.cs.cmu.edu/~chiragn/auton_logo.png"> 
