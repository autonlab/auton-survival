import torch

class CoxMixtureHeterogenousEffects(torch.nn.Module):
  """PyTorch model definition of the Cox Mixture with Hereogenous Effects Model.

  Cox Mixtures with Heterogenous Effects involves the assuming that the
  base survival rates are independent of the treatment effect.
  of the individual to be a mixture of K Cox Models. Conditioned on each
  subgroup Z=k; the PH assumptions are assumed to hold and the baseline
  hazard rates is determined non-parametrically using an spline-interpolated
  Breslow's estimator.

  """

  def __init__(self, k, g, inputdim, hidden=None):

    super(CoxMixtureHeterogenousEffects, self).__init__()

    assert isinstance(k, int)

    if hidden is None:
      hidden = inputdim

    self.k = k # Base Physiology groups
    self.g = g # Treatment Effect groups

    self.embedding = torch.nn.Identity()

    self.expert = torch.nn.Linear(hidden, k, bias=False)

    self.z_gate = torch.nn.Linear(hidden, k, bias=False)
    self.phi_gate = torch.nn.Linear(hidden, g, bias=False)

    self.omega = torch.nn.Parameter(torch.rand(g)-0.5)

  def forward(self, x, a):

    x = self.embedding(x)
    a = 2*(a -0.5)

    log_hrs = torch.clamp(self.expert(x), min=-100, max=100)

    logp_z_gate = torch.nn.LogSoftmax(dim=1)(self.z_gate(x)) #
    logp_phi_gate = torch.nn.LogSoftmax(dim=1)(self.phi_gate(x))

    logp_jointlatent_gate = torch.zeros(len(x), self.k, self.g)

    for i in range(self.k):
      for j in range(self.g):
        logp_jointlatent_gate[:, i, j] = logp_z_gate[:, i] + logp_phi_gate[:, j]

    logp_joint_hrs = torch.zeros(len(x), self.k, self.g)

    for i in range(self.k):
      for j in range(self.g):
        logp_joint_hrs[:, i, j] = log_hrs[:, i] + (j!=2)*a*self.omega[j]

    return logp_jointlatent_gate, logp_joint_hrs

class DeepCoxMixtureHeterogenousEffects(CoxMixtureHeterogenousEffects):

  def __init__(self, k, g, inputdim, hidden):

    super(DeepCoxMixtureHeterogenousEffects, self).__init__(k, g,
                                                            inputdim,
                                                            hidden)

    # Get rich feature representations of the covariates
    self.embedding = torch.nn.Sequential(torch.nn.Linear(inputdim, hidden),
                                         torch.nn.Tanh(),
                                         torch.nn.Linear(hidden, hidden),
                                         torch.nn.Tanh())
