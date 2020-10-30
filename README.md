
Package: `dsm`
-------------

Python package `dsm` provides an API to train the Deep Survival Machines
and associated models for problems in survival analysis. The underlying model
is implemented in `pytorch`.

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

<img width=80% src=https://ndownloader.figshare.com/files/25259852>


**Deep Survival Machines (DSM)** is a **fully parametric** approach to model
Time-to-Event outcomes in the presence of Censoring, first introduced in
[\[1\]](https://arxiv.org/abs/2003.01176).
In the context of Healthcare ML and Biostatistics, this is known as 'Survival
Analysis'. The key idea behind Deep Survival Machines is to model the
underlying event outcome distribution as a mixure of some fixed \( K \)
parametric distributions. The parameters of these mixture distributions as
well as the mixing weights are modelled using Neural Networks.

#### Usage Example
    >>> from dsm import DeepSurvivalMachines
    >>> model = DeepSurvivalMachines()
    >>> model.fit()
    >>> model.predict_risk()

Deep Recurrent Survival Machines
--------------------------------

**Deep Recurrent Survival Machines (DRSM)** builds on the original **DSM**
model and allows for learning of representations of the input covariates using
**Recurrent Neural Networks** like **LSTMs, GRUs**. Deep Recurrent Survival
Machines is a natural fit to model problems where there are time dependendent
covariates.

> :warning: **Not Implemented Yet!**

Deep Convolutional Survival Machines
------------------------------------

Predictive maintenance and medical imaging sometimes requires to work with
image streams. Deep Convolutional Survival Machines extends **DSM** and
**DRSM** to learn representations of the input image data using
convolutional layers. If working with streaming data, the learnt
representations are then passed through an LSTM to model temporal dependencies
before determining the underlying survival distributions.

> :warning: **Not Implemented Yet!**

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
along with Deep Survival Machines.  If not, see <https://www.gnu.org/licenses/>.

<img style="float: left; padding-top:25px" width ="150px" src="https://www.cmu.edu/brand/downloads/assets/images/wordmarks-600x600-min.jpg">
<img style="float: right; padding-top:25px" width = "150px" src="https://www.autonlab.org/user/themes/auton/images/AutonLogo.png"> 
