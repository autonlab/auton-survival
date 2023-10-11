import numpy as np
import torch


def randargmax(b, **kw):
    """Random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def partial_ll_loss(lrisks, tb, eb, eps=1e-2):
    tb = tb + eps * np.random.random(len(tb))
    sindex = np.argsort(-tb)

    tb = tb[sindex]
    eb = eb[sindex]

    lrisks = lrisks[sindex]
    lrisksdenom = torch.logcumsumexp(lrisks, dim=0)

    plls = lrisks - lrisksdenom
    pll = plls[eb == 1]

    pll = torch.sum(pll)

    return -pll


def get_optimizer(model, lr):
    if model.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif model.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif model.optimizer == "RMSProp":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(
            "Optimizer " + model.optimizer + " is not implemented"
        )
