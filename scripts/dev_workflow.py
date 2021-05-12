import sys
sys.path.insert(0, '../DeepSurvivalMachines/')


import pandas as pd
import numpy as np

import dsm.prognostics.datasets as datasets
import dsm.prognostics.utilities as utilities

from dsm import DeepSurvivalMachines

train_data, test_data = datasets.load_turbofan(cmapss_folder='../CMAPSSData/', 
                                               experiment=1, 
                                               windowsize=30,
                                               flatten=True,
                                               sequential=True)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data
x_vl, t_vl, e_vl = x_tr[:10], t_tr[:10], e_tr[:10]

val_data = (x_vl, t_vl, e_vl)

model = DeepSurvivalMachines(layers=[500, 400, 300, 200], distribution='Normal')

model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-4,
          iters=100, val_data=train_data)

predictions = model.predict_mean(x_te)

print(np.sqrt(utilities.mean_squared_error(t_te, predictions)))
