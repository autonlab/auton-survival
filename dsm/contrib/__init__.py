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
`dsm` includes extended functionality for survival analysis as part
of `dsm.contrib`.

Contributed Modules
--------------------
This submodule incorporates contributed survival analysis methods.


Deep Cox Mixtures
------------------

The Cox Mixture involves the assumption that the survival function
of the individual to be a mixture of K Cox Models. Conditioned on each
subgroup Z=k; the PH assumptions are assumed to hold and the baseline
hazard rates is determined non-parametrically using an spline-interpolated
Breslow's estimator.

For full details on Deep Cox Mixture, refer to the paper [1].

References
----------
[1] <a href="https://arxiv.org/abs/2101.06536">Deep Cox Mixtures
for Survival Regression. Machine Learning in Health Conference (2021)</a>

```
  @article{nagpal2021dcm,
  title={Deep Cox mixtures for survival regression},
  author={Nagpal, Chirag and Yadlowsky, Steve and Rostamzadeh, Negar and Heller, Katherine},
  journal={arXiv preprint arXiv:2101.06536},
  year={2021}
  }
```

"""

from dsm.contrib.dcm_api import DeepCoxMixtures