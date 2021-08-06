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
                                               test_last=True)

x_tr, t_tr, e_tr = train_data
x_te, t_te, e_te = test_data

dropout = 0.5
activation = "sigmoid"
t_tr[t_tr>125] = 125

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


model = keras.Sequential(
    [
        layers.Dense(500, activation=activation, name="layer1"),
        layers.Dropout(dropout),
        layers.Dense(400, activation=activation, name="layer2"),
        layers.Dropout(dropout),
        layers.Dense(300, activation=activation, name="layer3"),
        layers.Dropout(dropout),
        layers.Dense(100, activation=activation, name="layer4"),
        #layers.Dropout(0.5),
        layers.Dense(1, name="layer5"),

    ]
)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=mean_squared_error, 
              optimizer=opt,  
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.fit(x_tr, t_tr, batch_size=512, epochs=200)

# opt = keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(loss=root_mean_squared_error, 
#               optimizer=opt,  
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])

# model.fit(x_tr, t_tr, batch_size=512, epochs=50)

print(model.evaluate(x_te, t_te.astype('float64')))