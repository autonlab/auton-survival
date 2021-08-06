import sys
sys.path.insert(0, '../')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pandas as pd
import numpy as np

import dsm.prognostics.datasets as datasets
import dsm.prognostics.utilities as utilities

from dsm import DeepSurvivalMachines



train_data, test_data = datasets.load_turbofan(cmapss_folder='../../CMAPSSData/', 
                                               experiment=1, 
                                               windowsize=30,
                                               flatten=True,
                                               sequential=False,
                                               test_last=True,
                                               censoring=0.2,
                                               return_censored=False)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data
x_vl, t_vl, e_vl = x_tr[:2], t_tr[:2], e_tr[:2] 
val_data = x_vl, t_vl, e_vl


print(x_tr.shape)

t_tr[t_tr>125] = 125

model = DeepSurvivalMachines(k=1, layers=[500], distribution='Normal', temp=.0,
                             cuda=True, precision='float')

model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-3,
          iters=50, val_data=test_data, early_stop=False)

# model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-4,
#           iters=50, val_data=val_data, early_stop=False)

predictions = model.predict_mean(x_te)
print(np.sqrt(utilities.mean_squared_error(t_te, predictions)))

predictions = model.predict_mean(x_tr)
print(np.sqrt(utilities.mean_squared_error(t_tr, predictions)))
