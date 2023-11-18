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


import torch
import numpy as np

from sksurv.linear_model.coxph import BreslowEstimator
from sklearn.utils import shuffle

from tqdm.auto import tqdm
from auton_survival.models.utils.common_utils import partial_ll_loss
from auton_survival.models.utils.cox_mixtures_utils import (
    get_hard_z,
    get_posteriors,
    get_probability,
    get_survival,
    repair_probs,
    sample_hard_z,
    smooth_bl_survival,
)

from auton_survival.logging import logger


def get_likelihood(model, breslow_splines, x, t, e, a):
    # Function requires numpy/torch

    gates, lrisks = model(x, a=a)
    lrisks = lrisks.numpy()
    e, t = e.numpy(), t.numpy()

    probs = []

    for i in range(model.g):
        survivals = get_survival(lrisks[:, :, i], breslow_splines, t)
        probability = get_probability(lrisks[:, :, i], breslow_splines, t)

        event_probs = np.array([survivals, probability])
        event_probs = event_probs[e.astype("int"), range(len(e)), :]
        probs.append(np.log(event_probs))

    probs = np.array(probs).transpose(1, 2, 0)
    event_probs = gates + probs

    return event_probs


def q_function(model, x, t, e, a, log_likelihoods, typ="soft"):
    z_posteriors = repair_probs(
        get_posteriors(torch.logsumexp(log_likelihoods, dim=2))
    )
    zeta_posteriors = repair_probs(
        get_posteriors(torch.logsumexp(log_likelihoods, dim=1))
    )

    if typ == "hard":
        z = get_hard_z(z_posteriors)
        zeta = get_hard_z(zeta_posteriors)
    else:
        z = sample_hard_z(z_posteriors)
        zeta = sample_hard_z(zeta_posteriors)

    gates, lrisks = model(x, a=a)

    loss = 0
    for i in range(model.k):
        lrisks_ = lrisks[:, i, :][range(len(zeta)), zeta]
        loss += partial_ll_loss(lrisks_[z == i], t[z == i], e[z == i])

    # log_smax_loss = -torch.nn.LogSoftmax(dim=1)(gates) # tf.nn.log_softmax(gates)

    posteriors = repair_probs(
        get_posteriors(log_likelihoods.reshape(-1, model.k * model.g))
    ).exp()

    gate_loss = posteriors * gates.reshape(-1, model.k * model.g)
    gate_loss = -torch.sum(gate_loss)
    loss += gate_loss

    return loss


def e_step(model, breslow_splines, x, t, e, a):
    # TODO: Do this in `Log Space`
    # If Breslow splines are not available, like in the first
    # iteration of learning, we randomly compute posteriors.
    if breslow_splines is None:
        log_likelihoods = torch.rand(len(x), model.k, model.g)
    else:
        log_likelihoods = get_likelihood(model, breslow_splines, x, t, e, a)

    return log_likelihoods


def m_step(model, optimizer, x, t, e, a, log_likelihoods, typ="soft"):
    optimizer.zero_grad()
    loss = q_function(model, x, t, e, a, log_likelihoods, typ)
    gate_regularization_loss = (model.phi_gate.weight**2).sum()
    gate_regularization_loss += (model.z_gate.weight**2).sum()
    loss += (model.gate_l2_penalty) * gate_regularization_loss
    loss.backward()
    optimizer.step()

    return float(loss)


def fit_breslow(
    model, x, t, e, a, log_likelihoods=None, smoothing_factor=1e-4, typ="soft"
):
    gates, lrisks = model(x, a=a)

    lrisks = lrisks.numpy()

    e = e.numpy()
    t = t.numpy()

    if log_likelihoods is None:
        z_posteriors = torch.logsumexp(gates, dim=2)
        zeta_posteriors = torch.logsumexp(gates, dim=1)
    else:
        z_posteriors = repair_probs(
            get_posteriors(torch.logsumexp(log_likelihoods, dim=2))
        )
        zeta_posteriors = repair_probs(
            get_posteriors(torch.logsumexp(log_likelihoods, dim=1))
        )

    if typ == "soft":
        z = sample_hard_z(z_posteriors)
        zeta = sample_hard_z(zeta_posteriors)
    else:
        z = get_hard_z(z_posteriors)
        zeta = get_hard_z(zeta_posteriors)

    breslow_splines = {}
    for i in range(model.k):
        breslowk = BreslowEstimator().fit(
            lrisks[:, i, :][range(len(zeta)), zeta][z == i],
            e[z == i],
            t[z == i],
        )
        breslow_splines[i] = smooth_bl_survival(
            breslowk, smoothing_factor=smoothing_factor
        )

    return breslow_splines


def train_step(
    model,
    x,
    t,
    e,
    a,
    breslow_splines,
    optimizer,
    bs=256,
    seed=100,
    typ="soft",
    use_posteriors=False,
    update_splines_after=10,
    smoothing_factor=1e-4,
):
    x, t, e, a = shuffle(x, t, e, a, random_state=seed)

    n = x.shape[0]
    batches = (n // bs) + 1

    epoch_loss = 0
    for i in range(batches):
        xb = x[i * bs : (i + 1) * bs]
        tb = t[i * bs : (i + 1) * bs]
        eb = e[i * bs : (i + 1) * bs]
        ab = a[i * bs : (i + 1) * bs]

        # E-Step !!!
        # e_step_start = time.time()
        with torch.no_grad():
            log_likelihoods = e_step(model, breslow_splines, xb, tb, eb, ab)

        torch.enable_grad()
        loss = m_step(
            model, optimizer, xb, tb, eb, ab, log_likelihoods, typ=typ
        )
        epoch_loss += loss

        with torch.no_grad():
            if i % update_splines_after == 0:
                if use_posteriors:
                    log_likelihoods = e_step(model, breslow_splines, x, t, e, a)
                    breslow_splines = fit_breslow(
                        model,
                        x,
                        t,
                        e,
                        a,
                        log_likelihoods=log_likelihoods,
                        typ="soft",
                        smoothing_factor=smoothing_factor,
                    )
                else:
                    breslow_splines = fit_breslow(
                        model,
                        x,
                        t,
                        e,
                        a,
                        log_likelihoods=None,
                        typ="soft",
                        smoothing_factor=smoothing_factor,
                    )

    return breslow_splines


def test_step(model, x, t, e, a, breslow_splines, loss="q", typ="soft"):
    if loss == "q":
        with torch.no_grad():
            posteriors = e_step(model, breslow_splines, x, t, e, a)
            loss = q_function(model, x, t, e, a, posteriors, typ=typ)

    return float(loss / x.shape[0])


def train_cmhe(
    model,
    train_data,
    val_data,
    epochs=50,
    patience=2,
    vloss="q",
    bs=256,
    typ="soft",
    lr=1e-3,
    use_posteriors=False,
    debug=False,
    return_losses=False,
    update_splines_after=10,
    smoothing_factor=1e-4,
    random_seed=0,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if val_data is None:
        val_data = train_data

    xt, tt, et, at = train_data
    xv, tv, ev, av = val_data

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valc = np.inf
    patience_ = 0

    breslow_splines = None

    losses = []

    for epoch in tqdm(range(epochs)):
        # train_step_start = time.time()
        breslow_splines = train_step(
            model,
            xt,
            tt,
            et,
            at,
            breslow_splines,
            optimizer,
            bs=bs,
            seed=epoch,
            typ=typ,
            use_posteriors=use_posteriors,
            update_splines_after=update_splines_after,
            smoothing_factor=smoothing_factor,
        )

        valcn = test_step(
            model, xv, tv, ev, av, breslow_splines, loss=vloss, typ=typ
        )

        losses.append(valcn)

        logger.debug(f"Patience: {patience_} | Epoch: {epoch} | Loss: {valcn}")

        if valcn > valc:
            patience_ += 1
        else:
            patience_ = 0

        if patience_ == patience:
            break

        valc = valcn

    if return_losses:
        return (model, breslow_splines), losses
    else:
        return (model, breslow_splines)


def predict_survival(model, x, a, t):
    if isinstance(t, (int, float)):
        t = [t]

    model, breslow_splines = model

    gates, lrisks = model(x, a=a)

    lrisks = lrisks.detach().numpy()
    gates = gates.exp().reshape(-1, model.k * model.g).detach().numpy()

    predictions = []
    for t_ in t:
        expert_outputs = []
        for i in range(model.g):
            expert_output = get_survival(lrisks[:, :, i], breslow_splines, t_)
            expert_outputs.append(expert_output)
        expert_outputs = (
            np.array(expert_outputs)
            .transpose(1, 2, 0)
            .reshape(-1, model.k * model.g)
        )

        predictions.append((gates * expert_outputs).sum(axis=1))
    return np.array(predictions).T


def predict_latent_z(model, x):
    model, _ = model
    gates = model.model.embedding(x)

    z_gate_probs = torch.exp(gates).sum(axis=2).detach().numpy()

    return z_gate_probs


def predict_latent_phi(model, x):
    model, _ = model
    x = model.embedding(x)

    p_phi_gate = torch.nn.Softmax(dim=1)(model.phi_gate(x)).detach().numpy()

    return p_phi_gate
