import pandas as pd
import numpy as np

import dsm.prognostics.datasets as datasets
import dsm.prognostics.utilities as utilities

from dsm import DeepRecurrentSurvivalMachines


tr_file = '../CMAPSSData/train_FD001.txt'
te_file = ('../CMAPSSData/test_FD001.txt', '../CMAPSSData/RUL_FD001.txt')

train_data, test_data = datasets.load_turbofan(tr_file, te_file)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data


model = DeepRecurrentSurvivalMachines(layers=1, hidden=50, 
                                      typ='GRU', distribution='Normal')
model.fit(x_tr, t_tr, e_tr, batch_size=5, learning_rate=1e-3,
          iters=25, val_data=train_data)

predictions = model.predict_mean(x_te)

print( utilities.mean_squared_error)
