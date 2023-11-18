import torch
import torch.nn as nn

from auton_survival.models.dsm.dsm_torch import create_representation


class DeepCoxPHTorch(nn.Module):
    def _init_coxph_layers(self, lastdim, bias=False):
        self.expert = nn.Linear(lastdim, 1, bias=bias)

    def __init__(
        self,
        inputdim,
        layers=None,
        optimizer="Adam",
        activation="ReLU",
        bias=False,
        dropout=None,
    ):
        super(DeepCoxPHTorch, self).__init__()

        self.inputdim = inputdim
        self.optimizer = optimizer
        self.bias = bias
        self.activation = activation
        self.dropout = dropout

        if layers is None:
            layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self._init_coxph_layers(lastdim)
        self.embedding = create_representation(
            inputdim, layers, activation=activation, bias=bias, dropout=dropout
        )

    def forward(self, x):
        return self.expert(self.embedding(x))


class DeepRecurrentCoxPHTorch(DeepCoxPHTorch):
    def __init__(
        self, inputdim, typ="LSTM", layers=1, hidden=None, optimizer="Adam"
    ):
        super(DeepCoxPHTorch, self).__init__()

        self.typ = typ
        self.layers = layers
        self.hidden = hidden
        self.optimizer = optimizer

        self._init_coxph_layers(hidden)

        if self.typ == "LSTM":
            self.embedding = nn.LSTM(
                inputdim, hidden, layers, bias=False, batch_first=True
            )
        if self.typ == "RNN":
            self.embedding = nn.RNN(
                inputdim,
                hidden,
                layers,
                bias=False,
                batch_first=True,
                nonlinearity="relu",
            )
        if self.typ == "GRU":
            self.embedding = nn.GRU(
                inputdim, hidden, layers, bias=False, batch_first=True
            )

    def forward(self, x):
        x = x.detach().clone()
        inputmask = ~torch.isnan(x[:, :, 0]).reshape(-1)
        x[torch.isnan(x)] = 0

        xrep, _ = self.embedding(x)
        xrep = xrep.contiguous().view(-1, self.hidden)
        xrep = xrep[inputmask]
        xrep = nn.ReLU6()(xrep)

        dim = xrep.shape[0]

        return self.expert(xrep.view(dim, -1))
