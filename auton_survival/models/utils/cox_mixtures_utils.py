import numpy as np
from scipy.interpolate import UnivariateSpline


from inspect import signature

import torch


def fit_spline(
    t,
    surv,
    smoothing_factor=1e-4,
    k=signature(UnivariateSpline).parameters["k"].default,
):
    return UnivariateSpline(t, surv, s=smoothing_factor, ext=3, k=k)


def smooth_bl_survival(
    breslow, smoothing_factor, k=signature(fit_spline).parameters["k"].default
):
    blsurvival = breslow.baseline_survival_
    x, y = blsurvival.x, blsurvival.y
    return fit_spline(x, y, smoothing_factor=smoothing_factor, k=k)


def get_probability_(lrisks, ts, spl):
    risks = np.exp(lrisks)
    s0ts = (-risks) * (spl(ts) ** (risks - 1))
    return s0ts * spl.derivative()(ts)


def get_survival_(lrisks, ts, spl):
    risks = np.exp(lrisks)
    return spl(ts) ** risks


def get_probability(lrisks, breslow_splines, t):
    psurv = []
    for i in range(lrisks.shape[1]):
        p = get_probability_(lrisks[:, i], t, breslow_splines[i])
        psurv.append(p)
    psurv = np.array(psurv).T
    return psurv


def get_survival(lrisks, breslow_splines, t):
    psurv = []
    for i in range(lrisks.shape[1]):
        p = get_survival_(lrisks[:, i], t, breslow_splines[i])
        psurv.append(p)
    psurv = np.array(psurv).T
    return psurv


def get_hard_z(gates_prob):
    return torch.argmax(gates_prob, dim=1)


def sample_hard_z(gates_prob):
    return torch.multinomial(gates_prob.exp(), num_samples=1)[:, 0]


def get_posteriors(probs):
    return probs - torch.logsumexp(probs, dim=1).reshape(-1, 1)


def repair_probs(probs, cutoff=-20):
    probs[torch.isnan(probs)] = cutoff
    probs[probs < cutoff] = cutoff
    return probs
