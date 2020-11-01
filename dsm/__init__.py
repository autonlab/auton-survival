# coding=utf-8
# Copyright 2020 Chirag Nagpal
#
# This file is part of Deep Survival Machines.

# Deep Survival Machines is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Deep Survival Machines is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Deep Survival Machines.
# If not, see <https://www.gnu.org/licenses/>.

"""
[![Build Status](https://travis-ci.org/autonlab/DeepSurvivalMachines.svg?\
branch=master)](https://travis-ci.org/autonlab/DeepSurvivalMachines)
&nbsp;&nbsp;&nbsp;\
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)]\
(https://www.gnu.org/licenses/gpl-3.0)
&nbsp;&nbsp;&nbsp;\
[![GitHub Repo stars](https://img.shields.io/github/stars/autonlab/Deep\
SurvivalMachines?style=social)](https://github.com/autonlab/DeepSurvival\
Machines)


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


..warning:: Not Implemented Yet!

Deep Convolutional Survival Machines
------------------------------------

Predictive maintenance and medical imaging sometimes requires to work with
image streams. Deep Convolutional Survival Machines extends **DSM** and
**DRSM** to learn representations of the input image data using
convolutional layers. If working with streaming data, the learnt
representations are then passed through an LSTM to model temporal dependencies
before determining the underlying survival distributions.

..warning:: Not Implemented Yet!


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
Copyright 2020 [Chirag Nagpal](http://cs.cmu.edu/~chiragn),
[Auton Lab](http://www.autonlab.org).

Deep Survival Machines is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Deep Survival Machines is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.



<img style="float: right;" width ="200px" src="https://www.cmu.edu/brand/\
downloads/assets/images/wordmarks-600x600-min.jpg">
<img style="float: right;padding-top:50px" src="https://www.autonlab.org/\
user/themes/auton/images/AutonLogo.png">

<br><br><br><br><br>
<br><br><br><br><br>

"""

from dsm.dsm_api import DeepSurvivalMachines, DeepRecurrentSurvivalMachines
