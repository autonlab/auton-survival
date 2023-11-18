import torch
import numpy as np
import pandas as pd


from sksurv.linear_model.coxph import BreslowEstimator

from sklearn.utils import shuffle

from tqdm.auto import tqdm

from auton_survival.models.dsm.dsm_utilities import (
    _reshape_tensor_with_nans,
)

from copy import deepcopy
from auton_survival.models.utils.common_utils import (
    get_optimizer,
    partial_ll_loss,
)

from auton_survival.logging import LOGGER

logger = LOGGER.getChild(__name__)


def fit_breslow(model, x, t, e):
    return BreslowEstimator().fit(
        model(x).detach().cpu().numpy(), e.numpy(), t.numpy()
    )


@torch.enable_grad()
def train_step(model: torch.nn.Module, x, t, e, optimizer, bs=256, seed=100):
    x, t, e = shuffle(x, t, e, random_state=seed)

    n = x.shape[0]

    batches = (n // bs) + 1

    epoch_loss = 0

    model.train()

    for i in range(batches):
        xb = x[i * bs : (i + 1) * bs]
        tb = t[i * bs : (i + 1) * bs]
        eb = e[i * bs : (i + 1) * bs]

        loss = partial_ll_loss(
            model(xb),
            _reshape_tensor_with_nans(tb),
            _reshape_tensor_with_nans(eb),
            eps=1e-3,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += float(loss)

    return epoch_loss / n


@torch.inference_mode()
def test_step(model, x, t, e):
    model.eval()

    loss = float(partial_ll_loss(model(x), t, e, eps=1e-3))

    return loss / x.shape[0]


def train_dcph(
    model,
    train_data,
    val_data,
    epochs=50,
    patience=3,
    bs=256,
    lr=1e-3,
    random_seed=0,
    return_losses=False,
    breslow: bool = True,
    weight_decay=0.001,
    momentum=0.9,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if val_data is None:
        val_data = train_data

    xt, tt, et = train_data
    xv, tv, ev = val_data

    tt_ = _reshape_tensor_with_nans(tt)
    et_ = _reshape_tensor_with_nans(et)
    tv_ = _reshape_tensor_with_nans(tv)
    ev_ = _reshape_tensor_with_nans(ev)

    optimizer = get_optimizer(
        model, lr, weight_decay=weight_decay, momentum=momentum
    )

    valc = np.inf
    patience_ = 0

    breslow_spline = None

    losses = []
    dics = []

    for epoch in tqdm(range(epochs), desc="Training"):
        _ = train_step(model, xt, tt, et, optimizer, bs, seed=epoch)

        valcn = test_step(model, xv, tv_, ev_)

        losses.append(float(valcn))

        dics.append(deepcopy(model.state_dict()))

        logger.debug(f"Patience: {patience_} | Epoch: {epoch} | Loss: {valcn}")

        if valcn > valc:
            patience_ += 1
        else:
            patience_ = 0

        if patience_ == patience:
            break

        valc = valcn

    minm = np.argmin(losses)
    model.load_state_dict(dics[minm])

    breslow_spline = fit_breslow(model, xt, tt_, et_) if breslow else None

    if return_losses:
        return (model, breslow_spline), losses
    else:
        return (model, breslow_spline)


@torch.inference_mode()
def predict_survival(model, x, t=None):
    if isinstance(t, (int, float)):
        t = [t]

    model, breslow_spline = model
    model.eval()
    lrisks = model(x).detach().cpu().numpy()

    unique_times = breslow_spline.baseline_survival_.x

    raw_predictions = breslow_spline.get_survival_function(lrisks)
    raw_predictions = np.array([pred.y for pred in raw_predictions])

    predictions = pd.DataFrame(data=raw_predictions, columns=unique_times)

    if t is None:
        return predictions
    else:
        return __interpolate_missing_times(predictions.T, t)


def __interpolate_missing_times(survival_predictions, times):
    nans = np.full(survival_predictions.shape[1], np.nan)
    not_in_index = list(set(times) - set(survival_predictions.index))

    for idx in not_in_index:
        survival_predictions.loc[idx] = nans
    return (
        survival_predictions.sort_index(axis=0)
        .interpolate(method="bfill")
        .T[times]
        .values
    )
