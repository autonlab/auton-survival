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

from torch import nn 


train_data, test_data = datasets.load_turbofan(cmapss_folder='../../CMAPSSData/', 
                                               experiment=1, 
                                               windowsize=30,
                                               flatten=True,
                                               sequential=False,
                                               test_last=True,
                                               censoring=0.0)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data

vsize = 5

print(len(t_tr), len(t_tr), len(e_tr))

#train_data = x_tr[vsize:], t_tr[vsize:], e_tr[vsize:]
val_data = x_tr[:vsize], t_tr[:vsize], e_tr[:vsize] 

x_vl, t_vl, e_vl = val_data

dropout = 0.5

t_tr[t_tr>125] = 125

act = nn.Tanh

representation = []

representation.append(nn.Linear(15*30,500))
representation.append(act())
representation.append(nn.Dropout(dropout))
representation.append(nn.Linear(500,400))
representation.append(act())
representation.append(nn.Dropout(dropout))
representation.append(nn.Linear(400,300))
representation.append(act())
representation.append(nn.Dropout(dropout))
representation.append(nn.Linear(300,100))
representation.append(act())
representation = nn.Sequential(*representation)


model = DeepSurvivalMachines(k=1, layers=[100], distribution='Normal', temp=0.1, embedding=representation,
                             cuda=False)

costs =  model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-3,
                   iters=50, val_data=test_data,
                   early_stop=False)


from matplotlib import pyplot as plt
plt.plot(costs)
plt.savefig("results.pdf")

# model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-4,
#           iters=50, val_data=val_data,
#           early_stop=False)

predictions = model.predict_mean(x_te)
print("Test RMSE:", np.sqrt(utilities.mean_squared_error(t_te, predictions)))

predictions = model.predict_mean(x_vl)
print("Val RMSE:", np.sqrt(utilities.mean_squared_error(t_vl, predictions)))

predictions = model.predict_mean(x_tr)
print("Train RMSE:", np.sqrt(utilities.mean_squared_error(t_tr, predictions)))
