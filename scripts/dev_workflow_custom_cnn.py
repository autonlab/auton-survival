import sys
sys.path.insert(0, '../')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"


import pandas as pd
import numpy as np

import dsm.prognostics.datasets as datasets
import dsm.prognostics.utilities as utilities

from dsm import DeepSurvivalMachines, DeepConvolutionalSurvivalMachines

import torch
from torch import nn 


train_data, test_data = datasets.load_turbofan(cmapss_folder='../../CMAPSSData/', 
                                               experiment=1, 
                                               windowsize=30,
                                               flatten=False,
                                               sequential=False,
                                               test_last=True,
                                               censoring=0.0)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data

vsize = 2

#train_data = x_tr[vsize:], t_tr[vsize:], e_tr[vsize:]
val_data = x_tr[:vsize], t_tr[:vsize], e_tr[:vsize] 

x_vl, t_vl, e_vl = val_data

print (x_tr.shape)

t_tr[t_tr>125] = 125

class CNN(nn.Module):
    
    def __init__(self, hidden, output, dropout=0.5):
        super(CNN, self).__init__()

        self.zeropad = nn.ZeroPad2d((0,0,0,9))
        self.conv1 = nn.Conv2d(1, 10, (10,1), 1,0,1)
        self.conv2 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv3 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv4 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv5 = nn.Conv2d(10,1,(3,1),1,(1,0),1)
        self.fc = nn.Linear(hidden, output)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,input):

#        print("input:", input.shape)
        input = input.unsqueeze(-1)
        input = torch.transpose(input, 1, 3).contiguous()
        #print("transformed:", input.shape)

        out = self.zeropad(input)
        out = self.zeropad(self.act(self.conv1(out)))
        out = self.zeropad(self.act(self.conv2(out)))
        out = self.zeropad(self.act(self.conv3(out)))
        out = self.act(self.conv4(out))
        out = self.act(self.conv5(out))
        out = self.dropout(out.view(out.size(0), -1))
        out = self.act(self.fc(out))
        
        return out


base_model = CNN(450, 100, 0.5)
model = DeepConvolutionalSurvivalMachines(k=1, hidden=100, distribution='Normal', 
                                          temp=0.0, 
                                          embedding=base_model, 
                                          cuda=True, precision='float')

costs = model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-3,
                  iters=200, val_data=test_data, early_stop=False)

from matplotlib import pyplot as plt
plt.plot(costs)
plt.savefig("results_cnn.pdf")

# model.fit(x_tr, t_tr, e_tr, batch_size=512, learning_rate=1e-4,
#           iters=50, val_data=val_data, early_stop=False)

predictions = model.predict_mean(x_te)
print("Test RMSE:", np.sqrt(utilities.mean_squared_error(t_te, predictions)))
predictions = model.predict_mean(x_vl)
print("Val RMSE:", np.sqrt(utilities.mean_squared_error(t_vl, predictions)))
predictions = model.predict_mean(x_tr)
print("Train RMSE:", np.sqrt(utilities.mean_squared_error(t_tr, predictions)))
