import warnings
import logging
from matplotlib.pyplot import get

import torch
import numpy as np

from sksurv.linear_model.coxph import BreslowEstimator

from sklearn.utils import shuffle

from tqdm.auto import tqdm

from auton_survival.models.utils.common_utils import get_optimizer

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

logger = logging.getLogger(__name__)


def get_likelihood(model, breslow_splines, x, t, e):
    # Function requires numpy/torch

    gates, lrisks = model(x)
    lrisks = lrisks.numpy()
    e, t = e.numpy(), t.numpy()

    survivals = get_survival(lrisks, breslow_splines, t)
    probability = get_probability(lrisks, breslow_splines, t)

    event_probs = np.array([survivals, probability])
    event_probs = event_probs[e.astype("int"), range(len(e)), :]
    # event_probs[event_probs<1e-10] = 1e-10
    probs = gates + np.log(event_probs)
    # else:
    #   gates_prob = torch.nn.Softmax(dim = 1)(gates)
    #   probs = gates_prob*event_probs
    return probs


def q_function(model, x, t, e, posteriors, typ="soft"):
    if typ == "hard":
        z = get_hard_z(posteriors)
    else:
        z = sample_hard_z(posteriors)

    gates, lrisks = model(x)

    k = model.k

    loss = 0
    for i in range(k):
        lrisks_ = lrisks[z == i][:, i]
        loss += partial_ll_loss(lrisks_, t[z == i], e[z == i])

    # log_smax_loss = -torch.nn.LogSoftmax(dim=1)(gates) # tf.nn.log_softmax(gates)

    gate_loss = posteriors.exp() * gates
    gate_loss = -torch.sum(gate_loss)
    loss += gate_loss

    return loss


def e_step(model, breslow_splines, x, t, e):
    # TODO: Do this in `Log Space`
    if breslow_splines is None:
        # If Breslow splines are not available, like in the first
        # iteration of learning, we randomly compute posteriors.
        posteriors = get_posteriors(torch.rand(len(x), model.k))
        pass
    else:
        probs = get_likelihood(model, breslow_splines, x, t, e)
        posteriors = get_posteriors(repair_probs(probs, cutoff=-10))

    return posteriors


def m_step(model, optimizer, x, t, e, posteriors, typ="soft"):
    optimizer.zero_grad()
    loss = q_function(model, x, t, e, posteriors, typ)
    loss.backward()
    optimizer.step()

    return float(loss)


def fit_breslow(
    model, x, t, e, posteriors=None, smoothing_factor=1e-4, typ="soft"
):
    # TODO: Make Breslow in Torch !!!

    gates, lrisks = model(x)

    lrisks = lrisks.numpy()

    e = e.numpy()
    t = t.numpy()

    if posteriors is None:
        z_probs = gates
    else:
        z_probs = posteriors

    if typ == "soft":
        z = sample_hard_z(z_probs)
    else:
        z = get_hard_z(z_probs)

    breslow_splines = {}
    for i in range(model.k):
        breslowk = BreslowEstimator().fit(
            lrisks[:, i][z == i], e[z == i], t[z == i]
        )
        breslow_splines[i] = smooth_bl_survival(
            breslowk, smoothing_factor=smoothing_factor, k=1
        )

    return breslow_splines


def train_step(
    model,
    x,
    t,
    e,
    breslow_splines,
    optimizer,
    bs=256,
    seed=100,
    typ="soft",
    use_posteriors=False,
    update_splines_after=10,
    smoothing_factor=1e-4,
):
    x, t, e = shuffle(x, t, e, random_state=seed)

    n = x.shape[0]

    batches = (n // bs) + 1

    epoch_loss = 0
    for i in range(batches):
        xb = x[i * bs : (i + 1) * bs]
        tb = t[i * bs : (i + 1) * bs]
        eb = e[i * bs : (i + 1) * bs]
        # ab = a[i*bs:(i+1)*bs]

        # E-Step !!!
        # e_step_start = time.time()
        with torch.no_grad():
            posteriors = e_step(model, breslow_splines, xb, tb, eb)

        torch.enable_grad()
        loss = m_step(model, optimizer, xb, tb, eb, posteriors, typ=typ)

        with torch.no_grad():
            try:
                if i % update_splines_after == 0:
                    if use_posteriors:
                        posteriors = e_step(model, breslow_splines, x, t, e)
                        breslow_splines = fit_breslow(
                            model,
                            x,
                            t,
                            e,
                            posteriors=posteriors,
                            typ="soft",
                            smoothing_factor=smoothing_factor,
                        )
                    else:
                        breslow_splines = fit_breslow(
                            model,
                            x,
                            t,
                            e,
                            posteriors=None,
                            typ="soft",
                            smoothing_factor=smoothing_factor,
                        )

            except Exception as exception:
                warnings.warn(
                    "Couldn't fit splines, reusing from previous epoch"
                )
        epoch_loss += loss

    return breslow_splines


def test_step(model, x, t, e, breslow_splines, loss="q", typ="soft"):
    if loss == "q":
        with torch.no_grad():
            posteriors = e_step(model, breslow_splines, x, t, e)
            loss = q_function(model, x, t, e, posteriors, typ=typ)

    return float(loss / x.shape[0])


def train_dcm(
    model,
    train_data,
    val_data,
    epochs=50,
    patience=3,
    vloss="q",
    bs=256,
    typ="soft",
    lr=1e-3,
    use_posteriors=True,
    debug=False,
    random_seed=0,
    return_losses=False,
    update_splines_after=10,
    smoothing_factor=1e-2,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if val_data is None:
        val_data = train_data

    xt, tt, et = train_data
    xv, tv, ev = val_data

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = get_optimizer(model, lr)

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
            model, xv, tv, ev, breslow_splines, loss=vloss, typ=typ
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


def predict_survival(model, x, t):
    if isinstance(t, int) or isinstance(t, float):
        t = [t]

    model, breslow_splines = model
    gates, lrisks = model(x)

    lrisks = lrisks.detach().numpy()
    gate_probs = torch.exp(gates).detach().numpy()

    predictions = []

    for t_ in t:
        expert_output = get_survival(lrisks, breslow_splines, t_)
        predictions.append((gate_probs * expert_output).sum(axis=1))

    return np.array(predictions).T


def predict_latent_z(model, x):
    model, _ = model
    gates, _ = model(x)

    gate_probs = torch.exp(gates).detach().numpy()

    return gate_probs
