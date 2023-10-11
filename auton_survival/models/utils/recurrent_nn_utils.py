import numpy as np


def _get_padded_features(x):
    """Helper function to pad variable length RNN inputs with nans."""
    d = max([len(x_) for x_ in x])
    padx = []
    for i in range(len(x)):
        pads = np.nan * np.ones((d - len(x[i]),) + x[i].shape[1:])
        padx.append(np.concatenate([x[i], pads]))
    return np.array(padx)


def _get_padded_targets(t):
    """Helper function to pad variable length RNN inputs with nans."""
    d = max([len(t_) for t_ in t])
    padt = []
    for i in range(len(t)):
        pads = np.nan * np.ones(d - len(t[i]))
        padt.append(np.concatenate([t[i], pads]))
    return np.array(padt)[:, :, np.newaxis]
