# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

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


r"""
[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/autonlab/DeepSurvivalMachines/branch/master/graph/badge.svg?token=FU1HB5O92D)](https://codecov.io/gh/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/DeepSurvivalMachines?style=social)](https://github.com/autonlab/DeepSurvivalMachines)


Python package `dsm` provides an API to train the Deep Survival Machines
and associated models for problems in survival analysis. The underlying model
is implemented in `pytorch`.

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

Deep Survival Machines
----------------------

.. figure:: https://ndownloader.figshare.com/files/25259852
   :figwidth: 20 %
   :alt: map to buried treasure

   This is the caption of the figure (a simple paragraph).

**Deep Survival Machines (DSM)** is a fully parametric approach to model
Time-to-Event outcomes in the presence of Censoring first introduced in
[\[1\]](https://arxiv.org/abs/2003.01176).
In the context of Healthcare ML and Biostatistics, this is known as 'Survival
Analysis'. The key idea behind Deep Survival Machines is to model the
underlying event outcome distribution as a mixure of some fixed \( k \)
parametric distributions. The parameters of these mixture distributions as
well as the mixing weights are modelled using Neural Networks.

Example Usage
-------------

>>> from dsm import DeepSurvivalMachines
>>> from dsm import datasets
>>> # load the SUPPORT dataset.
>>> x, t, e = datasets.load_dataset('SUPPORT')
>>> # instantiate a DeepSurvivalMachines model.
>>> model = DeepSurvivalMachines()
>>> # fit the model to the dataset.
>>> model.fit(x, t, e)
>>> # estimate the predicted risks at the time
>>> model.predict_risk(x, 10)


Deep Recurrent Survival Machines
--------------------------------

**Deep Recurrent Survival Machines (DRSM)** builds on the original **DSM**
model and allows for learning of representations of the input covariates using
**Recurrent Neural Networks** like **LSTMs, GRUs**. Deep Recurrent Survival
Machines is a natural fit to model problems where there are time dependendent
covariates. Examples include situations where we are working with streaming
data like vital signs, degradation monitoring signals in predictive
maintainance. **DRSM** allows the learnt representations at each time step to
involve historical context from previous time steps. **DRSM** implementation in
`dsm` is carried out through an easy to use API,
`DeepRecurrentSurvivalMachines` that accepts lists of data streams and
corresponding failure times. The module automatically takes care of appropriate
batching and padding of variable length sequences.


Deep Convolutional Survival Machines
------------------------------------

Predictive maintenance and medical imaging sometimes requires to work with
image streams. Deep Convolutional Survival Machines extends **DSM** and
**DRSM** to learn representations of the input image data using
convolutional layers. If working with streaming data, the learnt
representations are then passed through an LSTM to model temporal dependencies
before determining the underlying survival distributions.

..warning:: Not Implemented Yet!


Deep Cox Mixtures
------------------

`dsm` includes pytorch implementation of the Deep Cox Mixture model proposed in
<a href="https://arxiv.org/abs/2101.06536">Deep Cox Mixtures
for Survival Regression. Machine Learning in Health Conference (2021)</a>

The Deep Cox Mixture involves the assumption that the survival function
of the individual to be a mixture of \( K \)  Cox Models. Conditioned on each
subgroup \( Z=k \) ; the PH assumptions are assumed to hold and the baseline
hazard rates is determined non-parametrically using an spline-interpolated
Breslow's estimator.


References
----------

Please cite the following papers if you are using the `dsm` package.

[1] [Deep Survival Machines:
Fully Parametric Survival Regression and
Representation Learning for Censored Data with Competing Risks."
arXiv preprint arXiv:2003.01176 (2020)](https://arxiv.org/abs/2003.01176)</a>

```
  @article{nagpal2020deep,
  title={Deep Survival Machines: Fully Parametric Survival Regression and\
  Representation Learning for Censored Data with Competing Risks},
  author={Nagpal, Chirag and Li, Xinyu and Dubrawski, Artur},
  journal={arXiv preprint arXiv:2003.01176},
  year={2020}
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
  @article{nagpal2021dcm,
  title={Deep Cox mixtures for survival regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  journal={arXiv preprint arXiv:2101.06536},
  year={2021}
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

"""

from dsm.dsm_api import DeepSurvivalMachines
from dsm.dsm_api import DeepConvolutionalSurvivalMachines
from dsm.dsm_api import DeepRecurrentSurvivalMachines
from dsm.dsm_api import DeepCNNRNNSurvivalMachines
from dsm.contrib.dcm_api import DeepCoxMixtures
